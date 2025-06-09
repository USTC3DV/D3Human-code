# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import tinycudann as tcnn
import numpy as np

#######################################################################################################################################################
# Small MLP using PyTorch primitives, internal helper class
#######################################################################################################################################################

class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
        self.net = torch.nn.Sequential(*net).cuda()
        
        self.net.apply(self._init_weights)
        
        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))

    def forward(self, x):
        return self.net(x.to(torch.float32))

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

############################################



#######################################################################################################################################################
# Outward visible MLP class
#######################################################################################################################################################

class MLPTexture3D(torch.nn.Module):
    def __init__(self, AABB, channels = 3, internal_dims = 32, hidden = 2, min_max = None, use_float16=False):
        super(MLPTexture3D, self).__init__()

        self.channels = channels
        self.internal_dims = internal_dims
        self.AABB = AABB
        self.min_max = min_max
        self.use_float16 = use_float16

        # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
        desired_resolution = 4096
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))


        enc_cfg =  {
            "otype": "HashGrid",
            "n_levels": 5, # 16
            "n_features_per_level": 2, #2
            "log2_hashmap_size": 21, # 21
            "base_resolution": base_grid_resolution, # 16
            "per_level_scale" : per_level_scale # 1.4472692374403782
	    }


        gradient_scaling = 128.0
        self.encoder = tcnn.Encoding(3, enc_cfg)

        # Setup MLP
        mlp_cfg = {
            "n_input_dims" : self.encoder.n_output_dims,
            "n_output_dims" : self.channels,
            "n_hidden_layers" : hidden,
            "n_neurons" : self.internal_dims
        }
        self.net = _MLP(mlp_cfg, gradient_scaling)
        print("Encoder output: %d dims" % (self.encoder.n_output_dims))

    def sample(self, texc, frame_id):

        ###################################
        bbox = torch.tensor([0.6, 0.6, 0.2]).cuda(), torch.tensor([-0.8, -1.2, -0.2]).cuda()
        _texc = (texc.view(-1, 3) - bbox[0][None, ...]) / (bbox[1][None, ...] - bbox[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        
        p_enc = self.encoder(_texc.contiguous())

        with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_float16):
            out = self.net.forward(p_enc)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]


        return out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, c]


    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        pass

    def cleanup(self):
        tcnn.free_temporary_memory()



class MeshTexture3D(torch.nn.Module):
    def __init__(self, v):
        pass

    def __get_load_Texture3d(self, mesh, FLAGS):

        kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
        ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')

        mlp_min = torch.cat((kd_min[0:3], ks_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max), dim=0)
        
        self.mlp_material = MLPTexture3D(mesh.getAABB(), channels=6, min_max=[mlp_min, mlp_max], use_float16=FLAGS.use_float16)

        

