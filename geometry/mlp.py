import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, nn
from einops import repeat

from .embedding import Embedding

class MLP(nn.Module):
    def __init__(self, n_freq=6, d_hidden=128, d_out=1, n_hidden=3, skip_in=[], use_float16=False):
        super().__init__()
        self.emb = Embedding(3, n_freq)
        layers = [
            nn.Linear(self.emb.out_channels, d_hidden),
            nn.Softplus(beta=100)
        ]
        count = 2
        self.skip_count = []
        self.skip_in = skip_in
        for i in range(n_hidden):
            if i in skip_in:
                layers.append(nn.Linear(d_hidden + self.emb.out_channels, d_hidden))
                self.skip_count.append(count)
            else:
                layers.append(nn.Linear(d_hidden, d_hidden))
            count += 1
            layers.append(nn.Softplus(beta=100))
            count += 1
        layers.append(nn.Linear(d_hidden, d_out))
        count += 1
        self.net = nn.ModuleList(layers)
        self.use_float16 = use_float16
    
    def forward(self, x):
        emb = self.emb(x)
        x = emb

        with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_float16):
            for i, module in enumerate(self.net):
                if i in self.skip_count:
                    x = module(torch.cat([x, emb], dim=-1))
                else:
                    x = module(x)

        return x
    
class MLP_nonrigid(nn.Module):
    def __init__(self, d_hidden=128, d_out=1, n_hidden=3, use_float16=False):
        super().__init__()
        # Initialize layers
        layers = [
            nn.Linear(72, d_hidden),  # Using the dimension of pose_code
            nn.Softplus(beta=100)
        ]
        count = 2
        for i in range(n_hidden):
            layers.append(nn.Linear(d_hidden, d_hidden))
            count += 1
            layers.append(nn.Softplus(beta=100))
            count += 1
        layers.append(nn.Linear(d_hidden, d_out))
        count += 1
        self.net = nn.ModuleList(layers)
        self.use_float16 = use_float16
    
    def forward(self, pose_code):
        x = pose_code
        
        # Use float16 precision if enabled
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_float16):
            for module in self.net:
                x = module(x)
        
        return x


class MLP_deform(nn.Module):
    def __init__(self, n_freq=6, d_hidden=128, d_out=1, n_hidden=3, skip_in=[], use_float16=False):
        super().__init__()
        self.emb = Embedding(3, n_freq)

        layers = [
            nn.Linear(self.emb.out_channels + 136, d_hidden),
            nn.Softplus(beta=100)
        ]
        count = 2
        self.skip_count = []
        self.skip_in = skip_in
        for i in range(n_hidden):
            if i in skip_in:
                print("-------skip_in")
                layers.append(nn.Linear(d_hidden + self.emb.out_channels , d_hidden))
                self.skip_count.append(count)
            else:
                layers.append(nn.Linear(d_hidden, d_hidden))
            count += 1
            layers.append(nn.Softplus(beta=100))
            count += 1
        layers.append(nn.Linear(d_hidden, d_out))
        count += 1
        self.net = nn.ModuleList(layers)
        self.use_float16 = use_float16
    
    def forward(self, x, pose_code):
        emb = self.emb(x)
        x = emb
        x_num = x.shape[1]
        expanded_pose_latent = pose_code.expand(1, x_num, 136)

        x = torch.cat([expanded_pose_latent, x], dim=-1)

        with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_float16):
            for i, module in enumerate(self.net):
                if i in self.skip_count:
                    x = module(torch.cat([x, emb], dim=-1))
                else:
                    x = module(x)
        return x


class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, c_dim: int, f_dim: int) -> None:
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        self.bn = nn.BatchNorm1d(f_dim, affine=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)  # type: ignore
        nn.init.zeros_(self.conv_beta.bias)  # type: ignore

    def forward(self, x: Tensor, c: Tensor) -> Tensor:

        assert x.shape[0] == c.shape[0]  # batch size
        assert c.shape[1] == self.c_dim  # embedding dim
        assert x.shape[2] == c.shape[2]  # num points

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class ConditionalResnetBlock1d(nn.Module):
    def __init__(self, c_dim: int, size_in: int) -> None:
        super().__init__()
        self.size_in = size_in
        self.bn_0 = ConditionalBatchNorm1d(c_dim, size_in)
        self.bn_1 = ConditionalBatchNorm1d(c_dim, size_in)

        self.fc_0 = nn.Conv1d(size_in, size_in, 1)
        self.fc_1 = nn.Conv1d(size_in, size_in, 1)

        self.actvn = nn.ReLU()

        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        return x + dx

class DecoderConditionalBatchNorm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim_condition_embedding: int,
        dim_hidden_layers: int,
        num_hidden_layers: int,
        dim_out: int,
    ):
        super().__init__()

        self.fc_p = nn.Conv1d(input_dim, dim_hidden_layers, 1)

        self.num_blocks = num_hidden_layers
        self.blocks = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.blocks.append(
                ConditionalResnetBlock1d(
                    dim_condition_embedding,
                    dim_hidden_layers,
                )
            )

        self.bn = ConditionalBatchNorm1d(dim_condition_embedding, dim_hidden_layers)

        self.fc_out = nn.Conv1d(dim_hidden_layers, dim_out, 1)

        # self.actvn = nn.ReLU()

    def forward(self, points: Tensor, conditions: Tensor) -> Tensor:
        p = points.transpose(1, 2)
        c = conditions.transpose(1, 2)
        net = self.fc_p(p)

        for i in range(self.num_blocks):
            net = self.blocks[i](net, c)

        # out = self.fc_out(self.actvn(self.bn(net, c)))
        out = self.fc_out(self.bn(net, c))
        out = out.squeeze(1)

        return out

class DisNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        out_dim: int = 3,
    ) -> None:
        super().__init__()
        self.decoder = DecoderConditionalBatchNorm(
            input_dim,
            latent_dim,
            hidden_dim,
            num_hidden_layers,
            out_dim,
        )

    def forward(self, coords: Tensor, latent_codes: Tensor) -> Tensor:
        # coords -> (B, N, 3)
        # latent_codes -> (B, D) or (B, N, D)
        if len(latent_codes.shape) == 2:
            latent_codes = repeat(latent_codes, "b d -> b r d", r=coords.shape[1])

        out = self.decoder(coords, latent_codes)

        return out
