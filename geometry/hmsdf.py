# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
from tqdm import trange
import random
from torchvision import transforms

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF

import pysdf
import trimesh

from PIL import Image

from render import mesh
from render import render
from render import render_mask
import render.optixutils as ou
from render import regularizer
from ssim_loss import ssim

from .gshell_tets import GShell_Tets
from .hmsdf_tets_split import hmSDF_Tets

import kaolin

from .mlp import MLP, MLP_deform
from deform.smplx_exavatar_deformer import SMPLX_Deformer
from script.get_tet_smpl import get_tet_mesh
from scipy.spatial import cKDTree

from pytorch3d.ops import knn_points
from lap_loss import body_laplacian_loss, body_normal_loss
###############################################################################
# Regularizer
###############################################################################

def write_pc(path_mesh, v, vn=None, f=None):
    # print("---------v:", v.shape)
    assert v.ndim == 2 and v.shape[1] == 3
    with open(path_mesh, 'w') as fp:
        fp.write(('v {:f} {:f} {:f}\n' * v.shape[0]).format(*v.reshape(-1)))
        if vn is not None:
            fp.write(('vn {:f} {:f} {:f}\n' * vn.shape[0]).format(*vn.reshape(-1)))
        if f is not None:
            fp.write(('f {:d} {:d} {:d}\n' * f.shape[0]).format(*f.reshape(-1) + 1))

def write_obj(path_mesh, v, f):
    with open(path_mesh, 'w') as obj_file:
        for vertex in v:
            obj_file.write('v {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))
        
        for face in f:
            obj_file.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))

def crop_image(image1, image2, h, w, crop_size):
    max_w = w - crop_size
    max_h = h - crop_size
    start_w = random.randint(0, max_w)
    start_h = random.randint(0, max_h)

    cropped_tensor1 = image1[..., start_h:start_h+crop_size, start_w:start_w+crop_size]
    cropped_tensor2 = image2[..., start_h:start_h+crop_size, start_w:start_w+crop_size]
    return cropped_tensor1, cropped_tensor2

class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice = nn.Sequential()
        for x in range(15):  # 停止在 relu3_3
            self.slice.add_module(str(x), vgg[x])
        if not requires_grad:
            for parameter in self.slice.parameters():
                parameter.requires_grad = False

    def forward(self, x, y):
        x_features = self.slice(x)
        y_features = self.slice(y)

        loss = nn.functional.l1_loss(x_features, y_features)
        return loss


# ori push_eps=0.001
def collision_loss(cloth_pos, body_pos, body_faces, push_eps=0.005):
    device = cloth_pos.device

    cloth_pos_batch = cloth_pos.unsqueeze(0)       # (1, N, 3)
    body_pos_batch = body_pos.unsqueeze(0)         # (1, M, 3)

    if body_faces.shape[0] == 3 and body_faces.ndim == 2:
        body_faces = body_faces.T  

    body_face_vertices = body_pos[body_faces]      # (F, 3, 3)
    body_face_vertices = body_face_vertices.unsqueeze(0)  # (1, F, 3, 3)
    body_face_centers = body_face_vertices.mean(dim=2)  # (1, F, 3)

    v0 = body_face_vertices[:, :, 1] - body_face_vertices[:, :, 0]  # (1, F, 3)
    v1 = body_face_vertices[:, :, 2] - body_face_vertices[:, :, 0]  # (1, F, 3)
    body_face_normals = torch.cross(v0, v1, dim=2)                  # (1, F, 3)
    body_face_normals = torch.nn.functional.normalize(body_face_normals, dim=2)

    body_face_centers = body_face_centers.to(device)
    body_face_normals = body_face_normals.to(device)

    knn_output = knn_points(cloth_pos_batch, body_face_centers, K=1, return_nn=True)
    nn_points = knn_output.knn[:, :, 0, :]  # (1, N, 3)
    nn_idx = knn_output.idx[:, :, 0]        # (1, N)

    nn_normals = body_face_normals[0][nn_idx[0]]  # (N, 3)
    nn_normals = nn_normals.unsqueeze(0)          # (1, N, 3)

    direction = cloth_pos_batch - nn_points  # (1, N, 3)
    distance = (direction * nn_normals).sum(dim=2)  # (1, N)

    interpenetration = torch.relu(push_eps - distance)  # (1, N)
    loss = (interpenetration ** 2).mean()

    return loss




class MobileNetPerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 4, 7], use_gpu=True):
        super(MobileNetPerceptualLoss, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True).features
        # mobilenet = models.mobilenet_v3_large(pretrained=True).features
        # mobilenet = models.mobilenet_v3_small(pretrained=True).features
        self.layers = layers
        self.features = nn.ModuleList(mobilenet).eval()
        if use_gpu:
            self.features = self.features.cuda()
        for param in self.features.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        loss = 0
        for i, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += self.criterion(x, y)
        loss = loss/3
        return loss


def compute_sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)

    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])

    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

###############################################################################
#  Geometry interface
###############################################################################



class HmSDFTetsGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS, offset=None):
        super(HmSDFTetsGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.gshell_tets   = GShell_Tets()
        self.hmsdf_tets   = hmSDF_Tets()
        self.scale         = scale
        self.batch_point_num = 100000
        self.boxscale      = torch.tensor(FLAGS.boxscale).view(1, 3).cuda()
        self.perceptual_loss = PerceptualLoss().cuda()
        self.mobileNet_perceptual_loss = MobileNetPerceptualLoss().cuda()
        self.smplx_deform   = SMPLX_Deformer(model_path = 'smplx', gender=self.FLAGS.gender)
        
        self._init_tet()
        self._init_sdf()
        self._init_use_nonrigid_deform()
        self._init_msdf()
        self._init_deform()
        self._init_cond(img_num=self.FLAGS.n_images)
        self._init_render_cond(img_num=self.FLAGS.n_images)

        self.fix_code = torch.nn.Parameter(0.1*torch.randn((1, 1, 136)).cuda(), requires_grad=True)

    def _init_tet(self):
        with torch.no_grad():
            self.optix_ctx = ou.OptiXContext()

            tets = np.load('data/tets/tet_grid.npz')
            print(f'using resolution {self.grid_res}')
            self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda')
            self.verts[:,1] = self.verts[:,1]- 0.1919
            self.verts *= 1.2
            self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
            self.generate_edges()

    def _init_sdf(self):
        ckp_root = os.path.join(self.FLAGS.out_dir, "ckp")
        os.makedirs(ckp_root, exist_ok=True)
        model_save_path = os.path.join(ckp_root, 'init_smpl_deform_convex_{}.pth'.format(self.grid_res))

        if self.FLAGS.pretrain_smpl:
            self.sdf    = torch.nn.Parameter(torch.zeros_like(self.verts[:, 0]), requires_grad=True) ## placeholder
            self.register_parameter('sdf', self.sdf)
            self.sdf_net = MLP(
                skip_in=self.FLAGS.skip_in,
                n_freq=self.FLAGS.n_freq,
                n_hidden=self.FLAGS.n_hidden,
                d_hidden=self.FLAGS.d_hidden,
                use_float16=self.FLAGS.use_float16
            )

            init_mesh_path = os.path.join(self.FLAGS.out_dir, "smpl_template_{}.obj".format(self.grid_res))

            self.smplx_deform.initialize(betas=self.FLAGS.shape_param.cuda(), save_path=init_mesh_path)
            init_mesh = trimesh.load(init_mesh_path, process=False)

            sdf_gt_fn = pysdf.SDF(init_mesh.vertices, init_mesh.faces)
            sdf_gt = -torch.from_numpy(sdf_gt_fn(np.array(self.verts.cpu()))[:,None]).cuda()

            tet_mesh_path = os.path.join(self.FLAGS.out_dir, "smpl_template_tet_{}.npz".format(self.grid_res))
            tet_smpl_v, tet_smpl_f = get_tet_mesh(init_mesh_path, tet_mesh_path)
            self.sdf_tet_gt = -torch.from_numpy(sdf_gt_fn(np.array(tet_smpl_v))[:,None]).cuda()
            self.tet_smpl_v = torch.from_numpy(tet_smpl_v).cuda()
            self.tet_smpl_f = torch.from_numpy(tet_smpl_f).long().cuda()

            self.generate_edges_smpl()

            msdf = (torch.rand_like(self.tet_smpl_v[:,0]) - 0.01).clamp(-1, 1)
            self.smpl_msdf = torch.nn.Parameter(msdf.clone().detach(), requires_grad=True)
            self.register_parameter('smpl_msdf', self.smpl_msdf)

            if os.path.exists(model_save_path):
                self.sdf_net.load_state_dict(torch.load(model_save_path))
                self.sdf_net.cuda()
            else:
                self.sdf_net.cuda()
                optimizer = torch.optim.Adam(self.sdf_net.parameters(), lr=1e-3)

                num_batches = (self.verts.shape[0] + self.batch_point_num - 1) // self.batch_point_num

                for _ in trange(self.FLAGS.sdf_mlp_pretrain_smpl_steps):
                    sdf_out_list = []
                    for i in range(num_batches):
                        batch_verts = self.verts[i * self.batch_point_num:(i + 1) * self.batch_point_num]
                        batch_sdf_out = self.sdf_net(batch_verts)
                        sdf_out_list.append(batch_sdf_out)
                    sdf_out = torch.cat(sdf_out_list, dim=0)
                    loss = (sdf_out - sdf_gt).pow(2).mean()
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                print('init smpl sdf net trained with loss:', loss)
                torch.save(self.sdf_net.state_dict(), model_save_path)


    def _init_use_nonrigid_deform(self):
        if self.FLAGS.use_nonrigid_deform:
            deform_model_save_path = 'checkpoints/init_deform_deform_cond_pe8.pth'
            pose_latent = torch.zeros(1, 1, 136).cuda()
            self.nonrigid = MLP_deform(
                skip_in=self.FLAGS.skip_in,
                n_freq=8,
                n_hidden=self.FLAGS.n_hidden,
                d_hidden=self.FLAGS.d_hidden,
                use_float16=self.FLAGS.use_float16,
                d_out=3
            )

            if os.path.exists(deform_model_save_path):
                self.nonrigid.load_state_dict(torch.load(deform_model_save_path))
                self.nonrigid.cuda()
            else:
                self.nonrigid.cuda()

                target = torch.zeros_like(self.verts) 
                optimizer = torch.optim.Adam(self.nonrigid.parameters(), lr=1e-3)

                for _ in trange(self.FLAGS.sdf_deform_pretrain_steps):
                    scaled_verts = self.verts / self.boxscale
                    nonrigid_v_pred = self.nonrigid(self.verts.reshape(1,-1,3), pose_latent)
                    loss = (nonrigid_v_pred - target).pow(2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                print('use_nonrigid_deform trained with loss:', loss)
                torch.save(self.nonrigid.state_dict(), deform_model_save_path)

    def _init_msdf(self):
        msdf         = (torch.rand_like(self.verts[:,0]) - 0.01).clamp(-1, 1)
        self.msdf    = torch.nn.Parameter(msdf.clone().detach(), requires_grad=True)
        self.register_parameter('msdf', self.msdf)


    def _init_deform(self):
        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)
        self.clamp_deform()
        
    def _init_basedeform(self, v, f, body_v=None, cloth_v=None):

        self.base_v = v
        self.base_f = f
        if body_v is not None:
            self.body_v = body_v
        if cloth_v is not None:
            self.cloth_v = cloth_v


    def _init_cond(self, img_num):
        cond = torch.rand((img_num+1, 64), device=self.verts.device)
        self.cond = torch.nn.Parameter(cond, requires_grad=True)
        self.register_parameter('cond', self.cond)

    def _init_render_cond(self, img_num):
        render_cond = torch.rand((img_num+1, 64), device=self.verts.device)
        self.render_cond = torch.nn.Parameter(render_cond, requires_grad=True)
        self.register_parameter('render_cond', self.render_cond)



    def _init_use_body_nonrigid_deform(self):
        if self.FLAGS.use_nonrigid_deform:
            deform_model_save_path = 'checkpoints/init_deform_deform_cond_pe8.pth'
            pose_latent = torch.zeros(1, 1, 136).cuda()

            self.body_nonrigid = MLP_deform(
                skip_in=self.FLAGS.skip_in,
                n_freq=8,
                n_hidden=self.FLAGS.n_hidden,
                d_hidden=self.FLAGS.d_hidden,
                use_float16=self.FLAGS.use_float16,
                d_out=3
            )

            if os.path.exists(deform_model_save_path):
                self.body_nonrigid.load_state_dict(torch.load(deform_model_save_path))
                self.body_nonrigid.cuda()
            else:
                self.body_nonrigid.cuda()

                target = torch.zeros_like(self.verts) 
                optimizer = torch.optim.Adam(self.body_nonrigid.parameters(), lr=1e-3)

                for _ in trange(self.FLAGS.sdf_deform_pretrain_steps):
                    scaled_verts = self.verts / self.boxscale

                    nonrigid_v_pred = self.body_nonrigid(self.verts.reshape(1,-1,3), pose_latent)
                    loss = (nonrigid_v_pred - target).pow(2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                print('use_nonrigid_deform trained with loss:', loss)
                torch.save(self.body_nonrigid.state_dict(), deform_model_save_path)



    @torch.no_grad()
    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)
            self.max_displacement = 1.0 / self.grid_res * self.scale / 2.1

    @torch.no_grad()
    def generate_edges_smpl(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.tet_smpl_f[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges_smpl = torch.unique(all_edges_sorted, dim=0)
            self.max_displacement_smpl = 1.0 / self.grid_res * self.scale / 2.1


    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    @torch.no_grad()
    def clamp_deform(self):
        if not self.FLAGS.use_tanh_deform:
            self.deform.data[:] = self.deform.clamp(-1.0, 1.0)
        self.msdf.data[:] = self.msdf.clamp(-2.0, 2.0)

    @torch.no_grad()
    def new_smpl_pose(self, betas, body_pose, global_orient, transl):
        v, f = self.smpl_deform.smpl_forward_seq(betas, body_pose, global_orient, transl)
        return v, f


    def getMesh_init(self, material, target=None, it=None):

        smplx_param = {"shape": self.FLAGS.shape_param,
                       "face_offset": self.FLAGS.face_offset,
                       "joint_offset": self.FLAGS.joint_offset,
                       "locator_offset": self.FLAGS.locator_offset,
                       "trans": self.FLAGS.trans_optim,
                       "rhand_pose": self.FLAGS.rhand_pose_optim,
                       "jaw_pose": self.FLAGS.jaw_pose_optim,
                       "expr": self.FLAGS.expr_optim,
                       "body_pose": self.FLAGS.body_pose_optim,
                       "root_pose": self.FLAGS.root_pose_optim,
                       "lhand_pose": self.FLAGS.lhand_pose_optim,
                       "leye_pose": self.FLAGS.leye_pose_optim,
                       "reye_pose": self.FLAGS.reye_pose_optim,
                       }

        v_deformed = self.verts + self.max_displacement * self.deform
        if self.FLAGS.use_sdf_mlp:

            num_batches = (v_deformed.shape[0] + self.batch_point_num - 1) // self.batch_point_num
            sdf_outputs = [] 

            for i in range(num_batches):
                batch_v_deformed = v_deformed[i * self.batch_point_num:(i + 1) * self.batch_point_num]
                batch_sdf = self.sdf_net(batch_v_deformed)
                sdf_outputs.append(batch_sdf)

            sdf = torch.cat(sdf_outputs, dim=0)

        else:
            sdf = self.sdf

        if self.FLAGS.use_msdf_mlp:
            msdf = self.msdf_net(v_deformed)
        else:
            msdf = self.msdf

        verts, faces, uvs, uv_idx, v_tng, extra = self.gshell_tets(
            v_deformed, sdf, msdf, self.indices)
        
        deform_imesh = None
        return_dict = {}

        template_imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        tmp_nodeform_mesh = mesh.Mesh(verts, faces, v_tex=None, t_tex_idx=None, material=material)
        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, tmp_nodeform_mesh.v_pos.contiguous(), tmp_nodeform_mesh.t_pos_idx.int(), rebuild=1)
        tmp_nodeform_mesh = mesh.auto_normals(tmp_nodeform_mesh)
        return_dict['tmp_nodeform_mesh'] = tmp_nodeform_mesh


        if target is not None:

            verts_deform = self.smplx_deform.lbs_forward(verts.reshape(1, -1, 3), smplx_param, idx=target["idx"][0], face=faces)
            deform_imesh = mesh.Mesh(verts_deform, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)
            
            with torch.no_grad():
                ou.optix_build_bvh(self.optix_ctx, deform_imesh.v_pos.contiguous(), deform_imesh.t_pos_idx.int(), rebuild=1)
            deform_imesh = mesh.auto_normals(deform_imesh)
            
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, imesh.v_pos.contiguous(), imesh.t_pos_idx.int(), rebuild=1)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        
        return_dict['imesh'] = imesh
        return_dict['deform_imesh'] = deform_imesh
        return_dict['template_imesh'] = template_imesh
        return_dict['sdf'] = sdf
        return_dict['msdf'] = extra['msdf']
        return_dict['msdf_watertight'] = extra['msdf_watertight']
        return_dict['msdf_boundary'] = extra['msdf_boundary']
        return_dict['n_verts_watertight'] = extra['n_verts_watertight']

        if self.FLAGS.visualize_watertight:

            imesh_watertight = mesh.Mesh(extra['vertices_watertight'], extra['faces_watertight'], v_tex=None, t_tex_idx=None, material=material)
            imesh_watertight = mesh.auto_normals(imesh_watertight)

            if target is not None:

                tmp_nodeform_wt_mesh = mesh.Mesh(extra['vertices_watertight'], extra['faces_watertight'], v_tex=None, t_tex_idx=None, material=material)
                with torch.no_grad():
                    ou.optix_build_bvh(self.optix_ctx, tmp_nodeform_wt_mesh.v_pos.contiguous(), tmp_nodeform_wt_mesh.t_pos_idx.int(), rebuild=1)
                tmp_nodeform_wt_mesh = mesh.auto_normals(tmp_nodeform_wt_mesh)
                return_dict['tmp_nodeform_wt_mesh'] = tmp_nodeform_wt_mesh

                verts_deform_wt = self.smplx_deform.lbs_forward(extra['vertices_watertight'].reshape(1, -1, 3),
                                        smplx_param, 
                                        idx=target["idx"][0])

                
                deform_imesh_wt = mesh.Mesh(verts_deform_wt, extra['faces_watertight'], v_tex=None, t_tex_idx=None, material=material)
                
                with torch.no_grad():
                    ou.optix_build_bvh(self.optix_ctx, deform_imesh_wt.v_pos.contiguous(), deform_imesh_wt.t_pos_idx.int(), rebuild=1)
                deform_imesh_wt = mesh.auto_normals(deform_imesh_wt)

                return_dict['deform_imesh_wt'] = deform_imesh_wt

            return_dict['imesh_watertight'] = imesh_watertight

        return return_dict


    def getMesh_split(self, material, type, target=None, it=None):
        v_deformed = self.verts + self.max_displacement * self.deform
        if self.FLAGS.use_sdf_mlp:

            num_batches = (v_deformed.shape[0] + self.batch_point_num - 1) // self.batch_point_num
            sdf_outputs = [] 

            for i in range(num_batches):
                batch_v_deformed = v_deformed[i * self.batch_point_num:(i + 1) * self.batch_point_num]
                batch_sdf = self.sdf_net(batch_v_deformed)
                sdf_outputs.append(batch_sdf)

            sdf = torch.cat(sdf_outputs, dim=0)

        else:
            sdf = self.sdf

        if self.FLAGS.use_msdf_mlp:
            msdf = self.msdf_net(v_deformed)
        else:
            msdf = self.msdf

        verts, faces, uvs, uv_idx, v_tng, extra = self.hmsdf_tets(
            v_deformed, sdf, msdf, self.indices, type)
        
        deform_imesh = None
        return_dict = {}

        template_imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        if target is not None:

            tmp_nodeform_mesh = mesh.Mesh(verts, faces, v_tex=None, t_tex_idx=None, material=material)
            with torch.no_grad():
                ou.optix_build_bvh(self.optix_ctx, tmp_nodeform_mesh.v_pos.contiguous(), tmp_nodeform_mesh.t_pos_idx.int(), rebuild=1)
            tmp_nodeform_mesh = mesh.auto_normals(tmp_nodeform_mesh)
            return_dict['tmp_nodeform_mesh'] = tmp_nodeform_mesh


            smplx_param = {"shape": self.FLAGS.shape_param,
                       "face_offset": self.FLAGS.face_offset,
                       "joint_offset": self.FLAGS.joint_offset,
                       "locator_offset": self.FLAGS.locator_offset,
                       
                       "trans": self.FLAGS.trans_optim,
                       "rhand_pose": self.FLAGS.rhand_pose_optim,
                       "jaw_pose": self.FLAGS.jaw_pose_optim,
                       "expr": self.FLAGS.expr_optim,
                       "body_pose": self.FLAGS.body_pose_optim,
                       "root_pose": self.FLAGS.root_pose_optim,
                       "lhand_pose": self.FLAGS.lhand_pose_optim,
                       "leye_pose": self.FLAGS.leye_pose_optim,
                       "reye_pose": self.FLAGS.reye_pose_optim,
                       }


            verts_deform = self.smplx_deform.lbs_forward(verts.reshape(1, -1, 3), smplx_param, idx=target["idx"][0], face=faces)
            deform_imesh = mesh.Mesh(verts_deform, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)
            
            with torch.no_grad():
                ou.optix_build_bvh(self.optix_ctx, deform_imesh.v_pos.contiguous(), deform_imesh.t_pos_idx.int(), rebuild=1)
            deform_imesh = mesh.auto_normals(deform_imesh)
            
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, imesh.v_pos.contiguous(), imesh.t_pos_idx.int(), rebuild=1)
        imesh = mesh.auto_normals(imesh)
        
        return_dict['imesh'] = imesh
        return_dict['deform_imesh'] = deform_imesh
        return_dict['template_imesh'] = template_imesh
        return_dict['sdf'] = sdf
        return_dict['msdf'] = extra['msdf']
        return_dict['msdf_watertight'] = extra['msdf_watertight']
        return_dict['msdf_boundary'] = extra['msdf_boundary']
        return_dict['n_verts_watertight'] = extra['n_verts_watertight']



        if self.FLAGS.visualize_watertight:
            imesh_watertight = mesh.Mesh(extra['vertices_watertight'], extra['faces_watertight'], v_tex=None, t_tex_idx=None, material=material)
            imesh_watertight = mesh.auto_normals(imesh_watertight)

            if target is not None:

                tmp_nodeform_wt_mesh = mesh.Mesh(extra['vertices_watertight'], extra['faces_watertight'], v_tex=None, t_tex_idx=None, material=material)
                with torch.no_grad():
                    ou.optix_build_bvh(self.optix_ctx, tmp_nodeform_wt_mesh.v_pos.contiguous(), tmp_nodeform_wt_mesh.t_pos_idx.int(), rebuild=1)
                tmp_nodeform_wt_mesh = mesh.auto_normals(tmp_nodeform_wt_mesh)
                return_dict['tmp_nodeform_wt_mesh'] = tmp_nodeform_wt_mesh

                verts_deform_wt = self.smplx_deform.lbs_forward(extra['vertices_watertight'].reshape(1, -1, 3), smplx_param, idx=target["idx"][0])
                deform_imesh_wt = mesh.Mesh(verts_deform_wt, extra['faces_watertight'], v_tex=None, t_tex_idx=None, material=material)
                
                with torch.no_grad():
                    ou.optix_build_bvh(self.optix_ctx, deform_imesh_wt.v_pos.contiguous(), deform_imesh_wt.t_pos_idx.int(), rebuild=1)
                deform_imesh_wt = mesh.auto_normals(deform_imesh_wt)

                return_dict['deform_imesh_wt'] = deform_imesh_wt

            return_dict['imesh_watertight'] = imesh_watertight

        return return_dict


    def getMesh_seq(self, material, target=None, it=None, save_tmp=False, t="all"):

        smplx_param = {"shape": self.FLAGS.shape_param,
                       "face_offset": self.FLAGS.face_offset,
                       "joint_offset": self.FLAGS.joint_offset,
                       "locator_offset": self.FLAGS.locator_offset,
                       "trans": self.FLAGS.trans_optim,
                       "rhand_pose": self.FLAGS.rhand_pose_optim,
                       "jaw_pose": self.FLAGS.jaw_pose_optim,
                       "expr": self.FLAGS.expr_optim,
                       "body_pose": self.FLAGS.body_pose_optim,
                       "root_pose": self.FLAGS.root_pose_optim,
                       "lhand_pose": self.FLAGS.lhand_pose_optim,
                       "leye_pose": self.FLAGS.leye_pose_optim,
                       "reye_pose": self.FLAGS.reye_pose_optim,
                       }

        base_deform_v = self.base_v #+ base_d
        face_labels = self.FLAGS.face_labels

        f = self.base_f
        return_dict = {}

        if target is not None:


            delta = torch.zeros_like(self.base_v)
            pose_code = self.fix_code

            cloth_delta = self.nonrigid(self.cloth_v.reshape(1, -1, 3), pose_code).reshape(-1, 3)
            body_delta = self.nonrigid(self.body_v.reshape(1, -1, 3), pose_code).reshape(-1, 3)

            delta[self.FLAGS.v_labels==1] = cloth_delta # cloth
            delta[self.FLAGS.v_labels==0] = body_delta # body

            delta_v = base_deform_v + delta
            return_dict["delta"] = delta

            tmp_nodeform_mesh = mesh.Mesh(base_deform_v, f, v_tex=None, t_tex_idx=None, material=material, v_labels=self.FLAGS.v_labels, face_labels=self.FLAGS.face_labels)
            with torch.no_grad():
                ou.optix_build_bvh(self.optix_ctx, tmp_nodeform_mesh.v_pos.contiguous(), tmp_nodeform_mesh.t_pos_idx.int(), rebuild=1)
            tmp_nodeform_mesh = mesh.auto_normals(tmp_nodeform_mesh)
            return_dict['tmp_nodeform_mesh'] = tmp_nodeform_mesh

            if save_tmp == True:

                tmp_all_mesh = mesh.Mesh(delta_v, f, v_tex=None, t_tex_idx=None, material=material)

                with torch.no_grad():

                    ou.optix_build_bvh(self.optix_ctx, tmp_all_mesh.v_pos.contiguous(), tmp_all_mesh.t_pos_idx.int(), rebuild=1)

                tmp_all_mesh = mesh.auto_normals(tmp_all_mesh)
                return_dict['tmp_all_mesh'] = tmp_all_mesh

            verts_deform = self.smplx_deform.lbs_forward(delta_v.reshape(1, -1, 3), smplx_param, idx=target["idx"][0], face=f)


            v1 = verts_deform
        else:
            v1 = self.base_v

        all_mesh = mesh.Mesh(v1, f, v_tex=None, t_tex_idx=None, material=material, v_labels=self.FLAGS.v_labels, 
                             face_labels=self.FLAGS.face_labels, connected_faces=self.FLAGS.connected_faces, edges=self.FLAGS.edges)

        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, all_mesh.v_pos.contiguous(), all_mesh.t_pos_idx.int(), rebuild=1)

        all_mesh = mesh.auto_normals(all_mesh)
        return_dict['all_mesh'] = all_mesh

        return return_dict


    def render_init(self, glctx, target, lgt, opt_material, bsdf=None, denoiser=None, shadow_scale=1.0,
            use_uv=False, iteration=None):
        opt_mesh_dict = self.getMesh_init(opt_material, target=target, it=iteration)
        opt_mesh = opt_mesh_dict['deform_imesh']
        original_mesh = opt_mesh_dict['tmp_nodeform_mesh']
        opt_mesh_watertight = opt_mesh_dict['deform_imesh_wt'] if 'deform_imesh_wt' in opt_mesh_dict else None

        if opt_mesh.v_pos.size(0) != 0:
            sampled_pts = kaolin.ops.mesh.sample_points(opt_mesh.v_pos[None,...], opt_mesh.t_pos_idx, 50000)[0][0]
            opt_mesh_dict['sampled_pts'] = sampled_pts
        else:
            opt_mesh_dict['sampled_pts'] = None
    
        extra_dict = {
            'msdf': opt_mesh_dict['msdf'],
        }

        opt_mesh_dict['buffers'] = render.render_mesh(
            self.FLAGS, target["idx"][0], glctx, opt_mesh, original_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
            msaa=True, background=target['background'], bsdf=bsdf, use_uv=use_uv,
            optix_ctx=self.optix_ctx, denoiser=denoiser, shadow_scale=shadow_scale,
            extra_dict=extra_dict)
        
        if self.FLAGS.visualize_watertight:
            original_wt_mesh = opt_mesh_dict['tmp_nodeform_wt_mesh']
            opt_mesh_dict['buffers_watertight'] = render.render_mesh(
                self.FLAGS, target["idx"][0], glctx, opt_mesh_watertight, original_wt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                msaa=True, background=target['background'], bsdf=bsdf, use_uv=use_uv,
                optix_ctx=self.optix_ctx, denoiser=denoiser, shadow_scale=shadow_scale,
                extra_dict=extra_dict)

        return opt_mesh_dict


    def render_split(self, glctx, target, lgt, opt_material, type, bsdf=None, denoiser=None, shadow_scale=1.0,
            use_uv=False, iteration=None):
        opt_mesh_dict = self.getMesh_split(opt_material, type, target=target, it=iteration)
        opt_mesh = opt_mesh_dict['deform_imesh']
        opt_mesh_watertight = opt_mesh_dict['deform_imesh_wt'] if 'deform_imesh_wt' in opt_mesh_dict else None


        original_deform_mesh = opt_mesh_dict['tmp_nodeform_mesh']

        if opt_mesh.v_pos.size(0) != 0:
            sampled_pts = kaolin.ops.mesh.sample_points(opt_mesh.v_pos[None,...], opt_mesh.t_pos_idx, 50000)[0][0]
            opt_mesh_dict['sampled_pts'] = sampled_pts
        else:
            opt_mesh_dict['sampled_pts'] = None
    
        extra_dict = {
            'msdf': opt_mesh_dict['msdf'],
        }

        opt_mesh_dict['buffers'] = render.render_mesh(
            self.FLAGS, target["idx"][0], glctx, opt_mesh, original_deform_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
            msaa=True, background=target['background'], bsdf=bsdf, use_uv=use_uv,
            optix_ctx=self.optix_ctx, denoiser=denoiser, shadow_scale=shadow_scale,
            extra_dict=extra_dict)
        
        if self.FLAGS.visualize_watertight:
            original_deform_wt_mesh = opt_mesh_dict['tmp_nodeform_wt_mesh']
            opt_mesh_dict['buffers_watertight'] = render.render_mesh(
                self.FLAGS, target["idx"][0], glctx, opt_mesh_watertight, original_deform_wt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                msaa=True, background=target['background'], bsdf=bsdf, use_uv=use_uv,
                optix_ctx=self.optix_ctx, denoiser=denoiser, shadow_scale=shadow_scale,
                extra_dict=extra_dict)

        return opt_mesh_dict


    def render_seq(self, glctx, target, lgt, opt_material, bsdf=None, denoiser=None, shadow_scale=1.0,
            use_uv=False, iteration=None, t="all"):

        opt_mesh_dict = self.getMesh_seq(opt_material, target=target, it=iteration, save_tmp=True, t=t)
        all_mesh = opt_mesh_dict['all_mesh']
        original_mesh = opt_mesh_dict['tmp_nodeform_mesh']

        opt_mesh_dict['all_mesh_buffers'] = render_mask.render_mesh(
            self.FLAGS, target["idx"][0], glctx, all_mesh, original_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
            msaa=True, background=target['background'], bsdf=bsdf, use_uv=use_uv,
            optix_ctx=self.optix_ctx, denoiser=denoiser, shadow_scale=shadow_scale)

        mesh_id = opt_mesh_dict["all_mesh_buffers"]['mesh_id']

        v_label_render = mesh_id[:, :, :, 0]
        alpha = opt_mesh_dict["all_mesh_buffers"]['geometric_normal'][:, :, :, -1]

        cloth_mask = v_label_render * alpha 
        body_mask = (1-v_label_render)  * alpha 

        opt_mesh_dict["body_mask"] = body_mask
        opt_mesh_dict["cloth_mask"] = cloth_mask
        opt_mesh_dict["all_mask"] = alpha

        all_mesh_v = all_mesh.v_pos
        body_v = all_mesh_v[self.FLAGS.v_labels==0]
        cloth_v = all_mesh_v[self.FLAGS.v_labels==1]
        body_f = self.FLAGS.body_f
        cloth_f = self.FLAGS.cloth_f
        colli_loss = collision_loss(cloth_v, body_v, body_f)
        opt_mesh_dict["colli_loss"] = colli_loss

        return opt_mesh_dict

    def tick_init(self, glctx, target, lgt, opt_material, loss_fn, iteration, denoiser):

        t_iter = iteration / self.FLAGS.iter

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        shadow_ramp = min(iteration / 1000, 1.0) ### set occlusion ray influence
        if denoiser is not None: denoiser.set_influence(shadow_ramp)
        opt_mesh_dict = self.render_init(glctx, target, lgt, opt_material, 
            denoiser=denoiser,
            shadow_scale=shadow_ramp,
            iteration=iteration)
        buffers = opt_mesh_dict['buffers']

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================


        with torch.no_grad():
            # Image-space loss, split into a coverage component and a color component
            color_ref = target['all_img']
            gt_mask = color_ref[..., 3:]

        msk_loss = 100 * F.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss = loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        img_loss = img_loss + 5e-1 * F.l1_loss(buffers['msdf_image'].clamp(min=0) * (gt_mask == 0).float(), torch.zeros_like(gt_mask))
        img_loss = img_loss + 5e-1 * F.l1_loss(buffers['msdf_image'].clamp(max=0) * (gt_mask == 1).float(), torch.ones_like(gt_mask))

        if self.FLAGS.use_img_2nd_layer:
            color_ref_2nd = target['img_second']
            img_loss = img_loss + F.mse_loss(buffers['shaded_second'][..., 3:], color_ref_2nd[..., 3:]) 
            img_loss = img_loss + loss_fn(buffers['shaded_second'][..., 0:3] * color_ref_2nd[..., 3:], color_ref_2nd[..., 0:3] * color_ref_2nd[..., 3:])

        if self.FLAGS.use_depth:
            depth_loss_scale = 100.
            depth_loss = depth_loss_scale * ((buffers['invdepth'][:, :, :, :1] - target['invdepth'][:, :, :, :1]).abs()).mean()

            if self.FLAGS.use_depth_2nd_layer:
                depth_loss += 0.1 * depth_loss_scale * ((buffers['invdepth_second'][:, :, :, :1] - target['invdepth_second'][:, :, :, :1]).abs()).mean()
        else:
            depth_loss = torch.tensor(0., device=img_loss.device)

        # Eikonal
        if self.FLAGS.use_sdf_mlp and self.FLAGS.use_eikonal and opt_mesh_dict['sampled_pts'] is not None:
            v = opt_mesh_dict['sampled_pts'].detach()
            v.requires_grad = True

            sdf_eik = self.sdf_net(v)
            if self.FLAGS.eikonal_scale is None:
                ### Default hardcoded Eikonal loss schedule
                if iteration < 500:
                    eik_coeff = 3e-1
                elif iteration < 1000:
                    eik_coeff = 1e-1
                elif iteration < 2000:
                    eik_coeff = 1e-1
                else:
                    eik_coeff = 1e-2
            else:
                eik_coeff = self.FLAGS.eikonal_scale

            eik_loss = eik_coeff * (
                torch.autograd.grad(sdf_eik.sum(), v, create_graph=True)[0].pow(2).sum(dim=-1).sqrt() - 1
            ).pow(2).mean()
        else:
            eik_loss = torch.tensor(0., device=img_loss.device)

        # SDF regularizer
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)
        sdf_reg_loss = compute_sdf_reg_loss(opt_mesh_dict['sdf'], self.all_edges).mean() * sdf_weight
        geo_reg_loss = sdf_reg_loss + eik_loss
        reg_loss = geo_reg_loss 

        if iteration is not None and iteration>self.FLAGS.nonrigid_begin:
            delta = opt_mesh_dict['delta']
            delta_norms = torch.norm(delta, dim=1)
            delta_loss = torch.sum(delta_norms**2)
            reg_loss += delta_loss * 0.1
        else:
            delta_loss = torch.tensor(0., device=img_loss.device)


        unit_out_normal = F.normalize(buffers['geometric_normal'][...,0:3], p=2, dim=-1).reshape(-1, 3)
        unit_out_normal[..., 1] = -unit_out_normal[..., 1]
        unit_out_normal[..., 2] = -unit_out_normal[..., 2]
        unit_gt_normal = F.normalize(target['all_normal'][...,0:3], p=2, dim=-1).reshape(-1, 3)
        unit_out_normal = ((unit_out_normal + 1.0) / 2.0).reshape(1, self.FLAGS.train_res[0], self.FLAGS.train_res[1], 3).permute(0, 3, 1, 2)
        unit_gt_normal = ((unit_gt_normal + 1.0) / 2.0).reshape(1, self.FLAGS.train_res[0], self.FLAGS.train_res[1], 3).permute(0, 3, 1, 2)
        mobile_net_loss = self.mobileNet_perceptual_loss(unit_out_normal, unit_gt_normal)
        normal_loss =  50 * mobile_net_loss

        return {
            "img_loss": img_loss,
            "depth_loss": depth_loss,
            "sdf_reg_loss": sdf_reg_loss,
            "eik_loss": eik_loss,
            "msk_loss": msk_loss,
            "delta_loss": delta_loss,
            "reg_loss": reg_loss,
            "geo_reg_loss": geo_reg_loss,
            "normal_loss": normal_loss,

        }

    def tick_split(self, glctx, target, lgt, opt_material, loss_fn, iteration, denoiser, type):

        t_iter = iteration / self.FLAGS.iter

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        shadow_ramp = min(iteration / 1000, 1.0) ### set occlusion ray influence
        if denoiser is not None: denoiser.set_influence(shadow_ramp)
        opt_mesh_dict = self.render_split(glctx, target, lgt, opt_material, type,
            denoiser=denoiser,
            shadow_scale=shadow_ramp,
            iteration=iteration)
        buffers = opt_mesh_dict['buffers']

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================


        with torch.no_grad():
            # Image-space loss, split into a coverage component and a color component
            if type == "cloth":
                color_ref = target['cloth_img']
                normal_ref = target['cloth_normal']
            elif type == "body":
                color_ref = target['body_img']
                normal_ref = target['body_normal']
            elif type == "all":
                color_ref = target['all_img']
                normal_ref = target['all_normal']
            gt_mask = color_ref[..., 3:]

        msk_loss =  F.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss =  loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        img_loss = img_loss + 5e-1 * F.l1_loss(buffers['msdf_image'].clamp(min=0) * (gt_mask == 0).float(), torch.zeros_like(gt_mask))
        img_loss = img_loss + 5e-1 * F.l1_loss(buffers['msdf_image'].clamp(max=0) * (gt_mask == 1).float(), torch.ones_like(gt_mask))

        if self.FLAGS.use_img_2nd_layer:
            color_ref_2nd = target['img_second']
            img_loss = img_loss + F.mse_loss(buffers['shaded_second'][..., 3:], color_ref_2nd[..., 3:]) 
            img_loss = img_loss + loss_fn(buffers['shaded_second'][..., 0:3] * color_ref_2nd[..., 3:], color_ref_2nd[..., 0:3] * color_ref_2nd[..., 3:])


        if self.FLAGS.use_depth:
            depth_loss_scale = 100.
            depth_loss = depth_loss_scale * ((buffers['invdepth'][:, :, :, :1] - target['invdepth'][:, :, :, :1]).abs()).mean()

            if self.FLAGS.use_depth_2nd_layer:
                depth_loss += 0.1 * depth_loss_scale * ((buffers['invdepth_second'][:, :, :, :1] - target['invdepth_second'][:, :, :, :1]).abs()).mean()
        else:
            depth_loss = torch.tensor(0., device=img_loss.device)

        # Eikonal
        if self.FLAGS.use_sdf_mlp and self.FLAGS.use_eikonal and opt_mesh_dict['sampled_pts'] is not None:
            v = opt_mesh_dict['sampled_pts'].detach()
            v.requires_grad = True

            sdf_eik = self.sdf_net(v)
            if self.FLAGS.eikonal_scale is None:
                ### Default hardcoded Eikonal loss schedule
                if iteration < 500:
                    eik_coeff = 3e-1
                elif iteration < 1000:
                    eik_coeff = 1e-1
                elif iteration < 2000:
                    eik_coeff = 1e-1
                else:
                    eik_coeff = 1e-2
            else:
                eik_coeff = self.FLAGS.eikonal_scale

            eik_loss = eik_coeff * (
                torch.autograd.grad(sdf_eik.sum(), v, create_graph=True)[0].pow(2).sum(dim=-1).sqrt() - 1
            ).pow(2).mean()
        else:
            eik_loss = torch.tensor(0., device=img_loss.device)

        if self.FLAGS.use_mesh_msdf_reg:
            mesh_msdf_regscale = (64 / self.grid_res) ** 3 # scale inversely proportional to grid_res^3
            eps = 1e-3
            open_scale = self.FLAGS.msdf_reg_open_scale
            close_scale = self.FLAGS.msdf_reg_close_scale
            eps = torch.tensor([eps]).cuda()

            if open_scale > 0:
                mesh_msdf_reg_loss = open_scale * mesh_msdf_regscale * F.huber_loss(
                    opt_mesh_dict['msdf'].clamp(min=-eps).squeeze(), 
                    -eps.expand(opt_mesh_dict['msdf'].size(0)), 
                    reduction='sum'
                )
            else:
                mesh_msdf_reg_loss = torch.tensor(0., device=img_loss.device)

            if close_scale != 0:
                with torch.no_grad():
                    visible_verts = (opt_mesh_dict['imesh'].t_pos_idx[buffers['visible_triangles']]).unique()
                    visible_boundary_verts = visible_verts[visible_verts >= opt_mesh_dict['n_verts_watertight']] - opt_mesh_dict['n_verts_watertight']
                    visible_boundary_mask = torch.zeros(opt_mesh_dict['msdf_boundary'].size(0)).cuda()
                    visible_boundary_mask[visible_boundary_verts] = 1
                    visible_boundary_mask = visible_boundary_mask.bool()

                boundary_msdf = opt_mesh_dict['msdf_boundary']
                boundary_msdf = boundary_msdf[visible_boundary_mask]
                mesh_msdf_reg_loss += close_scale * mesh_msdf_regscale * F.huber_loss(
                    boundary_msdf.clamp(max=eps).squeeze(), 
                    eps.expand(boundary_msdf.size(0)), 
                    reduction='sum'
                )
        else:
            mesh_msdf_reg_loss = torch.tensor(0., device=img_loss.device)

        # SDF regularizer
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)
        sdf_reg_loss = compute_sdf_reg_loss(opt_mesh_dict['sdf'], self.all_edges).mean() * sdf_weight

        # Monochrome shading regularizer
        if 'diffuse_light' not in buffers:
            monochrome_loss = torch.zeros_like(img_loss)
        else:
            monochrome_loss = regularizer.shading_loss(buffers['diffuse_light'], buffers['specular_light'], color_ref, self.FLAGS.lambda_diffuse, self.FLAGS.lambda_specular)

        # Material smoothness regularizer
        mtl_smooth_loss = regularizer.material_smoothness_grad(
            buffers['kd_grad'], buffers['ks_grad'], buffers['normal_grad'], 
            lambda_kd=self.FLAGS.lambda_kd, lambda_ks=self.FLAGS.lambda_ks, lambda_nrm=self.FLAGS.lambda_nrm)

        # Chroma regularizer
        chroma_loss = regularizer.chroma_loss(buffers['kd'], color_ref, self.FLAGS.lambda_chroma)
        assert 'perturbed_nrm' not in buffers # disable normal map in first pass

        geo_reg_loss = sdf_reg_loss + eik_loss 
        shading_reg_loss =  monochrome_loss + mtl_smooth_loss + chroma_loss
        reg_loss = geo_reg_loss + shading_reg_loss

        if iteration is not None and iteration>self.FLAGS.nonrigid_begin:
            delta = opt_mesh_dict['delta']
            delta_norms = torch.norm(delta, dim=1)
            delta_loss = torch.sum(delta_norms**2)
            reg_loss += delta_loss * 0.1
        else:
            delta_loss = torch.tensor(0., device=img_loss.device)


        unit_out_normal = F.normalize(buffers['geometric_normal'][...,0:3], p=2, dim=-1).reshape(-1, 3)
        unit_out_normal[..., 1] = -unit_out_normal[..., 1]
        unit_out_normal[..., 2] = -unit_out_normal[..., 2]
        unit_gt_normal = F.normalize(normal_ref[...,0:3], p=2, dim=-1).reshape(-1, 3)

        normal_loss_mse =  F.mse_loss(unit_out_normal, unit_gt_normal)
        normal_loss_cos = 0.1 * (1 - F.cosine_similarity(unit_out_normal, unit_gt_normal, dim=1).mean())
        unit_out_normal = ((unit_out_normal + 1.0) / 2.0).reshape(1, self.FLAGS.train_res[0], self.FLAGS.train_res[1], 3).permute(0, 3, 1, 2)
        unit_gt_normal = ((unit_gt_normal + 1.0) / 2.0).reshape(1, self.FLAGS.train_res[0], self.FLAGS.train_res[1], 3).permute(0, 3, 1, 2)

        unit_out_normal_crop, unit_gt_normal_crop = crop_image(unit_out_normal, unit_gt_normal, self.FLAGS.texture_res[0], self.FLAGS.texture_res[1], crop_size=448)
        mobile_net_loss = self.mobileNet_perceptual_loss(unit_out_normal_crop, unit_gt_normal_crop)
        normal_loss =  5 * mobile_net_loss

        return {
            "img_loss": img_loss,
            "msk_loss": msk_loss,
            "depth_loss": depth_loss,
            "sdf_reg_loss": sdf_reg_loss,
            "eik_loss": eik_loss,
            "mesh_msdf_reg_loss": mesh_msdf_reg_loss,
            "monochrome_loss": monochrome_loss,
            "mtl_smooth_loss": mtl_smooth_loss,
            "chroma_loss": chroma_loss,
            "delta_loss": delta_loss,

            "reg_loss": reg_loss,
            "geo_reg_loss": geo_reg_loss,
            "shading_reg_loss": shading_reg_loss,

            "normal_loss_mse": normal_loss_mse,
            "normal_loss_cos": normal_loss_cos,
            "normal_loss": normal_loss,

        }


    def tick_seq(self, glctx, target, lgt, opt_material, loss_fn, iteration, denoiser, t="all"):

        t_iter = iteration / self.FLAGS.iter
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        shadow_ramp = min(iteration / 1000, 1.0) ### set occlusion ray influence
        if denoiser is not None: denoiser.set_influence(shadow_ramp)

        opt_mesh_dict = self.render_seq(glctx, target, lgt, opt_material, use_uv=False, denoiser=denoiser, t=t)
        all_mesh_buffers = opt_mesh_dict["all_mesh_buffers"]
        all_mesh = opt_mesh_dict["all_mesh"]
        visible_triangles = all_mesh_buffers["visible_triangles"]

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================

        with torch.no_grad():

            gt_cloth_img = target['cloth_img']
            gt_body_img = target['body_img']
            gt_all_img = target['all_img']
            gt_all_normal = target['all_normal']

        render_all_msk = opt_mesh_dict["all_mask"][...,None]
        render_cloth_msk = opt_mesh_dict["cloth_mask"][...,None]
        render_body_msk = opt_mesh_dict["body_mask"][...,None]

        all_msk_loss = 200*F.mse_loss(render_all_msk, gt_all_img[..., 3:]) 
        cloth_msk_loss = 200*F.mse_loss(render_cloth_msk, gt_cloth_img[..., 3:]) 
        body_msk_loss = 200*F.mse_loss(render_body_msk, gt_body_img[..., 3:]) 

        all_img_loss = loss_fn(all_mesh_buffers['shaded'][..., 0:3] * render_all_msk, gt_all_img[..., 0:3])
        cloth_img_loss = loss_fn(all_mesh_buffers['shaded'][..., 0:3] * render_cloth_msk, gt_cloth_img[..., 0:3])
        body_img_loss = loss_fn(all_mesh_buffers['shaded'][..., 0:3] * render_body_msk, gt_body_img[..., 0:3])

        all_mtl_smooth_loss = regularizer.material_smoothness_grad(
            all_mesh_buffers['kd_grad'], all_mesh_buffers['ks_grad'], all_mesh_buffers['normal_grad'], 
            lambda_kd=self.FLAGS.lambda_kd, lambda_ks=self.FLAGS.lambda_ks, lambda_nrm=self.FLAGS.lambda_nrm)

        # Chroma regularizer
        all_chroma_loss = regularizer.chroma_loss(all_mesh_buffers['kd'], gt_all_img, self.FLAGS.lambda_chroma)
        assert 'perturbed_nrm' not in all_mesh_buffers # disable normal map in first pass

        shading_reg_loss =  all_mtl_smooth_loss + all_chroma_loss
        reg_loss = shading_reg_loss
        delta = opt_mesh_dict['delta']
        delta_norms = torch.norm(delta, dim=1)
        delta_loss = torch.sum(delta_norms**2)

        unit_out_normal = F.normalize(all_mesh_buffers['geometric_normal'][...,0:3], p=2, dim=-1).reshape(-1, 3)
        unit_out_normal[..., 1] = -unit_out_normal[..., 1]
        unit_out_normal[..., 2] = -unit_out_normal[..., 2]
        unit_gt_normal = F.normalize(gt_all_normal[...,0:3], p=2, dim=-1).reshape(-1, 3)

        unit_out_normal = ((unit_out_normal + 1.0) / 2.0).reshape(1, self.FLAGS.train_res[0], self.FLAGS.train_res[1], 3).permute(0, 3, 1, 2)
        unit_gt_normal = ((unit_gt_normal + 1.0) / 2.0).reshape(1, self.FLAGS.train_res[0], self.FLAGS.train_res[1], 3).permute(0, 3, 1, 2)
        mobile_net_loss = self.mobileNet_perceptual_loss(unit_out_normal, unit_gt_normal)
        normal_loss = 20 * mobile_net_loss 

        nds_laplacian_loss = body_laplacian_loss(all_mesh)
        nds_normal_loss = body_normal_loss(all_mesh)

        return {
            "visible_triangles": visible_triangles,
            "delta": delta,
            "all_img_loss": all_img_loss,
            "all_msk_loss": all_msk_loss,
            "cloth_img_loss": cloth_img_loss,
            "cloth_msk_loss": cloth_msk_loss,
            "body_img_loss": body_img_loss,
            "body_msk_loss": body_msk_loss,
            "laplacian_loss": nds_laplacian_loss,
            "mtl_smooth_loss": all_mtl_smooth_loss,
            "chroma_loss": all_chroma_loss,
            "delta_loss": delta_loss,
            "reg_loss": reg_loss,
            "shading_reg_loss": shading_reg_loss,
            "normal_loss": normal_loss,
            "colli_loss": opt_mesh_dict["colli_loss"],
            "nds_normal_loss": nds_normal_loss,

        }
