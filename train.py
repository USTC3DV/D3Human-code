# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os

import sys
import time
import argparse
import json

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

import cv2
import openmesh as om

from dataset.dataset_split import Dataset_split
from geometry.hmsdf import HmSDFTetsGeometry

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render

from tensorboardX import SummaryWriter

from denoiser.denoiser import BilateralDenoiser
from script.process_body_cloth_head_msdfcut import process_body_msdf_distance_bodyedge
from script.connet_face_head import process_close_hole

import os
from lap_loss import find_connected_faces


RADIUS = 3.0

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Loss setup
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

def om_loadmesh(path):
    mesh = om.read_trimesh(path)
    v = np.array(mesh.points())
    f = np.array(mesh.face_vertex_indices())

    return v, f

@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch_init(target, bg_type='black', cloth_flag=True):

    if cloth_flag == True:
        assert len(target['cloth_img'].shape) == 4, "Image shape should be [n, h, w, c]"
        if bg_type == 'checker':
            background = torch.tensor(util.checkerboard(target['cloth_img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
        elif bg_type == 'black':
            background = torch.zeros(target['cloth_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
        elif bg_type == 'white':
            background = torch.ones(target['cloth_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
        elif bg_type == 'reference':
            background = target['cloth_img'][..., 0:3]
        elif bg_type == 'random':
            background = torch.rand(target['cloth_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
        
        target['all_img'] = target['cloth_img'].float().cuda()
        target['background'] = background
        target['all_img'] = torch.cat((torch.lerp(background, target['all_img'][..., 0:3], target['all_img'][..., 3:4]), target['all_img'][..., 3:4]), dim=-1)


    elif cloth_flag == False:
        assert len(target['all_img'].shape) == 4, "Image shape should be [n, h, w, c]"
        if bg_type == 'checker':
            background = torch.tensor(util.checkerboard(target['all_img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
        elif bg_type == 'black':
            background = torch.zeros(target['all_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
        elif bg_type == 'white':
            background = torch.ones(target['all_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
        elif bg_type == 'reference':
            background = target['all_img'][..., 0:3]
        elif bg_type == 'random':
            background = torch.rand(target['all_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda') 
        
        target['all_img'] = target['all_img'].float().cuda()
        target['background'] = background
        target['all_img'] = torch.cat((torch.lerp(background, target['all_img'][..., 0:3], target['all_img'][..., 3:4]), target['all_img'][..., 3:4]), dim=-1)

    else:
        assert False, "Unknown background type %s" % bg_type

    return target



@torch.no_grad()
def prepare_batch_split(target, bg_type='black'):

    assert len(target['cloth_img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['cloth_img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['cloth_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['cloth_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['cloth_img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['cloth_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    
    target['cloth_img'] = target['cloth_img'].float().cuda()
    target['cloth_background'] = background
    target['cloth_img'] = torch.cat((torch.lerp(background, target['cloth_img'][..., 0:3], target['cloth_img'][..., 3:4]), target['cloth_img'][..., 3:4]), dim=-1)


    assert len(target['all_img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['all_img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['all_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['all_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['all_img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['all_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda') 
    
    target['all_img'] = target['all_img'].float().cuda()
    target['background'] = background
    target['all_img'] = torch.cat((torch.lerp(background, target['all_img'][..., 0:3], target['all_img'][..., 3:4]), target['all_img'][..., 3:4]), dim=-1)


    assert len(target['body_img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['body_img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['body_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['body_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['body_img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['body_img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda') 
    
    target['body_img'] = target['body_img'].float().cuda()
    target['body_background'] = background
    target['body_img'] = torch.cat((torch.lerp(background, target['body_img'][..., 0:3], target['body_img'][..., 3:4]), target['body_img'][..., 3:4]), dim=-1)

    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    eval_mesh = geometry.getMesh(mat)
    try:
        eval_mesh = eval_mesh['imesh']
    except:
        pass
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks'])

    # Dilate all textures & use average color for background
    kd_avg = torch.sum(torch.sum(torch.sum(kd * mask, dim=0), dim=0), dim=0) / torch.sum(torch.sum(torch.sum(mask, dim=0), dim=0), dim=0)
    kd = util.dilate(kd, kd_avg[None, None, None, :], mask, 7)

    ks_avg = torch.sum(torch.sum(torch.sum(ks * mask, dim=0), dim=0), dim=0) / torch.sum(torch.sum(torch.sum(mask, dim=0), dim=0), dim=0)
    ks = util.dilate(ks, ks_avg[None, None, None, :], mask, 7)

    nrm_avg = torch.tensor([0, 0, 1], dtype=torch.float32, device="cuda")
    normal = nrm_avg[None, None, None, :].repeat(kd.shape[0], kd.shape[1], kd.shape[2], 1)
    
    new_mesh.material = mat.copy()
    del new_mesh.material['kd_ks']

    if FLAGS.transparency:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)
        print("kd shape", kd.shape)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    new_mesh.material.update({
        'kd'     : texture.Texture2D(kd.clone().detach().requires_grad_(True), min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks.clone().detach().requires_grad_(True), min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal.clone().detach().requires_grad_(True), min_max=[nrm_min, nrm_max]),
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max), dim=0)

        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=6, min_max=[mlp_min, mlp_max], use_float16=FLAGS.use_float16)
        mat =  {'kd_ks' : mlp_map_opt}
    else:
        raise NotImplementedError

    mat['bsdf'] = FLAGS.bsdf

    mat['no_perturbed_nrm'] = FLAGS.no_perturbed_nrm

    return mat

def initial_guess_material_knownkskd(geometry, mlp, FLAGS, init_mat=None):
    mat =  {
        'kd'     : init_mat['kd'],
        'ks'     : init_mat['ks']
    }

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat


def load_filtered_state_dict(model, checkpoint_path):
    model_state_dict = model.state_dict()
    loaded_state_dict = torch.load(checkpoint_path)
    filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
    model_state_dict.update(filtered_state_dict)
    return model_state_dict  


def load_ckp(FLAGS, save_path, geometry, mat, stage):
    if stage == "init":
        last = FLAGS.init_epoch - 1
    elif stage == "split":
        last = FLAGS.split_epoch - 1
    elif stage == "fine":
        last = FLAGS.fine_epoch - 1
    model_ckp = os.path.join(save_path, stage, "ckp", "model_{}.pt".format(last))
    mat_ckp = os.path.join(save_path, stage, "ckp", "mtl_{}.pt".format(last))
    light_ckp = os.path.join(save_path, stage, "ckp", "probe_{}.hdr".format(last))

    geometry_para = load_filtered_state_dict(geometry, model_ckp)
    mat_para = load_filtered_state_dict(mat['kd_ks'], mat_ckp)
    geometry.load_state_dict(geometry_para, strict=False)
    mat['kd_ks'].load_state_dict(mat_para, strict=False)
    lgt = light.load_env(light_ckp, scale=FLAGS.env_scale, res=[FLAGS.probe_res, FLAGS.probe_res])

    smpl_npz_path = os.path.join(save_path, stage, "ckp", "smpl_{}.pt.npz".format(last))
    smpl_npz = np.load(smpl_npz_path)
    trans_optim = smpl_npz["trans_optim"]
    rhand_pose_optim = smpl_npz["rhand_pose_optim"]
    jaw_pose_optim = smpl_npz["jaw_pose_optim"]
    expr_optim = smpl_npz["expr_optim"]
    body_pose_optim = smpl_npz["body_pose_optim"]
    root_pose_optim = smpl_npz["root_pose_optim"]
    lhand_pose_optim = smpl_npz["lhand_pose_optim"]
    leye_pose_optim = smpl_npz["leye_pose_optim"]
    reye_pose_optim = smpl_npz["reye_pose_optim"]

    FLAGS.trans_optim = torch.from_numpy(trans_optim).cuda().requires_grad_(True)
    FLAGS.rhand_pose_optim = torch.from_numpy(rhand_pose_optim).cuda().requires_grad_(True)
    FLAGS.jaw_pose_optim = torch.from_numpy(jaw_pose_optim).cuda().requires_grad_(True)
    FLAGS.expr_optim = torch.from_numpy(expr_optim).cuda().requires_grad_(True)
    FLAGS.body_pose_optim = torch.from_numpy(body_pose_optim).cuda().requires_grad_(True)
    FLAGS.root_pose_optim = torch.from_numpy(root_pose_optim).cuda().requires_grad_(True)
    FLAGS.lhand_pose_optim = torch.from_numpy(lhand_pose_optim).cuda().requires_grad_(True)
    FLAGS.leye_pose_optim = torch.from_numpy(leye_pose_optim).cuda().requires_grad_(True)
    FLAGS.reye_pose_optim = torch.from_numpy(reye_pose_optim).cuda().requires_grad_(True)
    
    return geometry, mat, lgt

###############################################################################
# Validation & testing
###############################################################################

def combine_mask(mask1, mask2):
    mask1_gray = torch.max(mask1, dim=2)[0]
    mask2_gray = torch.max(mask2, dim=2)[0]
    composite_image = torch.zeros((FLAGS.texture_res[0], FLAGS.texture_res[1], 3), dtype=torch.uint8).cuda()

    color1 = torch.tensor([255, 0, 0], dtype=torch.uint8).cuda()
    color2 = torch.tensor([0, 255, 0], dtype=torch.uint8).cuda()
    color_overlap = torch.tensor([255, 255, 0], dtype=torch.uint8).cuda()

    mask1_present = mask1_gray > 0
    mask2_present = mask2_gray > 0
    both_present = mask1_present & mask2_present
    only_mask1 = mask1_present & (~mask2_present)
    only_mask2 = (~mask1_present) & mask2_present

    composite_image[only_mask1] = color1
    composite_image[only_mask2] = color2
    composite_image[both_present] = color_overlap

    return composite_image

@torch.no_grad()
def validate_all_mesh(glctx, target, body_geometry, all_material, lgt, FLAGS, denoiser=None, t="all"):
    result_dict = {}
    render_out = body_geometry.render_seq(glctx, target, lgt, all_material, use_uv=False, denoiser=denoiser, t=t)
    all_mesh_buffers = render_out['all_mesh_buffers']
    all_mesh = render_out['all_mesh']

    tmp_all_mesh = render_out['tmp_all_mesh'] 

    result_dict['tmp_all_mesh'] = tmp_all_mesh
    result_dict['all_ref'] = util.rgb_to_srgb(target['all_img'][...,0:3])[0] # range[0,1]
    result_dict['all_opt'] = util.rgb_to_srgb(all_mesh_buffers['shaded'][...,0:3])[0] 
    result_dict['all_mask_opt'] = all_mesh_buffers['shaded'][...,3:][0].expand(-1, -1, 3)

    result_dict['all_geometric_normal'] = all_mesh_buffers['geometric_normal'][...,0:3][0] # range[0,1]
    result_dict['all_geometric_normal'][..., 1] = -result_dict['all_geometric_normal'][..., 1]
    result_dict['all_geometric_normal'][..., 2] = -result_dict['all_geometric_normal'][..., 2]
    result_dict['all_geometric_normal'] =  (result_dict['all_geometric_normal'] + 1.0) / 2.0
    result_dict['ref_all_normal']  = (target['all_normal'][...,0:3][0] + 1.0) / 2.0

    all_depth_map = all_mesh_buffers['depth'][...,:1][0].expand(-1, -1, 3) # range[1,20]
    all_min_val = torch.min(all_depth_map)
    all_max_val = torch.max(all_depth_map)
    all_normalized_depth_map = (all_depth_map - all_min_val) / (all_max_val - all_min_val)
    all_normalized_depth_map[torch.isnan(all_normalized_depth_map)] = 0
    result_dict['all_depth'] = all_normalized_depth_map

    result_dict['body_mask_ref'] = target['body_img'][...,3:][0].expand(-1, -1, 3)
    result_dict['cloth_mask_ref'] = target['cloth_img'][...,3:][0].expand(-1, -1, 3)
    result_dict['all_mask_ref'] = target['all_img'][...,3:][0].expand(-1, -1, 3)

    result_dict['all_mask']  = render_out["all_mask"][0,...,None].expand(-1, -1, 3)
    result_dict['body_mask']  = render_out["body_mask"][0,...,None].expand(-1, -1, 3)
    result_dict['cloth_mask']  = render_out["cloth_mask"][0,...,None].expand(-1, -1, 3)

    cloth_combined_mask = combine_mask(result_dict['cloth_mask_ref'], result_dict['cloth_mask'])
    body_combined_mask = combine_mask(result_dict['body_mask_ref'], result_dict['body_mask'])
    all_combined_mask = combine_mask(result_dict['all_mask_ref'], result_dict['all_mask'])

    result_image = torch.cat([
                    result_dict['all_opt'], 
                    result_dict['all_ref'], 
                    result_dict['all_geometric_normal'], 
                    result_dict['ref_all_normal'], 

                    result_dict['all_mask_ref'],
                    result_dict['all_mask'], 
                    all_combined_mask,
                    result_dict['body_mask_ref'],
                    result_dict['body_mask'],
                    body_combined_mask,
                    result_dict['cloth_mask_ref'],
                    result_dict['cloth_mask'],
                    cloth_combined_mask,
                    ], axis=1)

    return result_image, result_dict, all_mesh



@torch.no_grad()
def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, denoiser=None):
    result_dict = {}
    with torch.no_grad():
        buffers = geometry.render_init(glctx, target, lgt, opt_material, use_uv=False, denoiser=denoiser)['buffers']

        result_dict['ref'] = util.rgb_to_srgb(target['all_img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_dict['mask_opt'] = buffers['shaded'][...,3:][0].expand(-1, -1, 3)
        result_dict['mask_ref'] = target['all_img'][...,3:][0].expand(-1, -1, 3)
        result_dict['msdf_image'] = buffers['msdf_image'][...,:][0].expand(-1, -1, 3).clamp(min=0, max=1)

        result_dict['all_geometric_normal'] = buffers['geometric_normal'][...,0:3][0] # range[0,1]
        result_dict['all_geometric_normal'][..., 1] = -result_dict['all_geometric_normal'][..., 1]
        result_dict['all_geometric_normal'][..., 2] = -result_dict['all_geometric_normal'][..., 2]
        result_dict['all_geometric_normal'] = (result_dict['all_geometric_normal'] + 1.0) / 2.0
        result_dict['ref_all_normal']  = (target['all_normal'][...,0:3][0] + 1.0) / 2.0
    

        result_image = torch.cat([result_dict['opt'], result_dict['ref'], result_dict['mask_opt'], result_dict['mask_ref'], 
                                result_dict['all_geometric_normal'], result_dict['ref_all_normal'], result_dict['msdf_image']], axis=1)
              
        if FLAGS.display is not None:
            white_bg = torch.ones_like(target['background'])
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    result_dict['light_image'] = lgt.generate_image(FLAGS.display_res)
                    result_dict['light_image'] = util.rgb_to_srgb(result_dict['light_image'] / (1 + result_dict['light_image']))
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'bsdf' in layer:
                    img = render.render_mesh(FLAGS, glctx, opt_mesh, target['mvp'], target['campos'], target['light'] if lgt is None else lgt, target['resolution'],
                                                spp=target['spp'], num_layers=FLAGS.layers, background=white_bg, bsdf=layer['bsdf'], optix_ctx=geometry.optix_ctx)['shaded']
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(img[..., 0:3])[0]
                    else:
                        result_dict[layer['bsdf']] = img[0, ..., 0:3]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
                elif 'normals' in layer and not FLAGS.no_perturbed_nrm:
                    result_image = torch.cat([result_image, (buffers['perturbed_nrm'][0, ...,0:3] + 1.0) * 0.5], axis=1)
                elif 'diffuse_light' in layer:
                    result_image = torch.cat([result_image, util.rgb_to_srgb(buffers['diffuse_light'][..., 0:3])[0]], axis=1)
                elif 'specular_light' in layer:
                    result_image = torch.cat([result_image, util.rgb_to_srgb(buffers['specular_light'][..., 0:3])[0]], axis=1)

        return result_image, result_dict



@torch.no_grad()
def validate_itr_all(glctx, target, body_geometry, body_opt_material, lgt, FLAGS, denoiser=None):
    result_dict = {}

    body_buffers = body_geometry.render_split(glctx, target, lgt, body_opt_material, "body", use_uv=False, denoiser=denoiser)['buffers']
    cloth_buffers = body_geometry.render_split(glctx, target, lgt, body_opt_material, "cloth", use_uv=False, denoiser=denoiser)['buffers']

    result_dict['ref'] = util.rgb_to_srgb(target['all_img'][...,0:3])[0] # range[0,1]
    result_dict['body_ref'] = util.rgb_to_srgb(target['body_img'][...,0:3])[0] # range[0,1]
    result_dict['cloth_ref'] = util.rgb_to_srgb(target['cloth_img'][...,0:3])[0] # range[0,1]
    result_dict['body_opt'] = util.rgb_to_srgb(body_buffers['shaded'][...,0:3])[0]
    result_dict['cloth_opt'] = util.rgb_to_srgb(cloth_buffers['shaded'][...,0:3])[0]

    result_dict['body_geometric_normal'] = body_buffers['geometric_normal'][...,0:3][0] # range[0,1]
    result_dict['body_geometric_normal'][..., 1] = -result_dict['body_geometric_normal'][..., 1]
    result_dict['body_geometric_normal'][..., 2] = -result_dict['body_geometric_normal'][..., 2]
    result_dict['body_geometric_normal'] = (result_dict['body_geometric_normal'] + 1.0) / 2.0
    
    result_dict['cloth_geometric_normal'] = cloth_buffers['geometric_normal'][...,0:3][0] # range[0,1]
    result_dict['cloth_geometric_normal'][..., 1] = -result_dict['cloth_geometric_normal'][..., 1]
    result_dict['cloth_geometric_normal'][..., 2] = -result_dict['cloth_geometric_normal'][..., 2]
    result_dict['cloth_geometric_normal'] = (result_dict['cloth_geometric_normal'] + 1.0) / 2.0

    result_dict['ref_normal']  = (target['all_normal'][...,0:3][0] + 1.0) / 2.0
    result_dict['ref_body_normal']  = (target['body_normal'][...,0:3][0] + 1.0) / 2.0
    result_dict['ref_cloth_normal']  = (target['cloth_normal'][...,0:3][0] + 1.0) / 2.0

    body_depth_map = body_buffers['depth'][...,:1][0].expand(-1, -1, 3) # range[1,20]
    body_min_val = torch.min(body_depth_map)
    body_max_val = torch.max(body_depth_map)
    body_normalized_depth_map = (body_depth_map - body_min_val) / (body_max_val - body_min_val)
    body_normalized_depth_map[torch.isnan(body_normalized_depth_map)] = 0
    result_dict['body_depth'] = body_normalized_depth_map

    cloth_depth_map = cloth_buffers['depth'][...,:1][0].expand(-1, -1, 3) # range[1,20]
    cloth_min_val = torch.min(cloth_depth_map)
    cloth_max_val = torch.max(cloth_depth_map)
    cloth_normalized_depth_map = (cloth_depth_map - cloth_min_val) / (cloth_max_val - cloth_min_val)
    cloth_normalized_depth_map[torch.isnan(cloth_normalized_depth_map)] = 0
    result_dict['cloth_depth'] = cloth_normalized_depth_map

    result_dict['body_mask_opt'] = body_buffers['shaded'][...,3:][0].expand(-1, -1, 3)
    result_dict['cloth_mask_opt'] = cloth_buffers['shaded'][...,3:][0].expand(-1, -1, 3)
    result_dict['body_mask_ref'] = target['body_img'][...,3:][0].expand(-1, -1, 3)
    result_dict['cloth_mask_ref'] = target['cloth_img'][...,3:][0].expand(-1, -1, 3)
    result_dict['mask_ref'] = target['all_img'][...,3:][0].expand(-1, -1, 3)
    result_dict['body_msdf_image'] = body_buffers['msdf_image'][...,:][0].expand(-1, -1, 3).clamp(min=0, max=1)
    result_dict['cloth_msdf_image'] = cloth_buffers['msdf_image'][...,:][0].expand(-1, -1, 3).clamp(min=0, max=1)
    result_image = torch.cat([
                    result_dict['body_opt'], 
                    result_dict['body_ref'], 
                    result_dict['cloth_opt'], 
                    result_dict['cloth_ref'], 
                    result_dict['ref'], 

                    result_dict['body_geometric_normal'], 
                    result_dict['ref_body_normal'],
                    result_dict['cloth_geometric_normal'], 
                    result_dict['ref_cloth_normal'],
                    result_dict['ref_normal'], 

                    result_dict['body_mask_opt'], 
                    result_dict['body_mask_ref'], 
                    result_dict['cloth_mask_opt'], 
                    result_dict['cloth_mask_ref'], 
                    result_dict['mask_ref'], 

                    result_dict['body_msdf_image'],
                    result_dict['cloth_msdf_image']
                    ], axis=1)

    return result_image, result_dict


###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

def optimize_mesh_init(
        denoiser,
        glctx,
        geometry,
        opt_material,
        lgt,
        dataset_train,
        dataset_validate,
        FLAGS,
        warmup_iter=300,
        log_interval=10,
        pass_idx=0,
        pass_name="",
        optimize_light=True,
        optimize_geometry=True,
        optimize_camera=True,
        visualize=True,
        save_path=None
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_lgt = learning_rate[2] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate * 6.0

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter 
        return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    params = list(material.get_parameters(opt_material))

    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)

    if optimize_light:
        optimizer_light = torch.optim.Adam((lgt.parameters() if lgt is not None else []), lr=learning_rate_lgt)
        scheduler_light = torch.optim.lr_scheduler.LambdaLR(optimizer_light, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    if optimize_geometry:
        if FLAGS.use_sdf_mlp:
            lr_msdf = learning_rate_pos * 1e-2 if FLAGS.use_msdf_mlp else learning_rate_pos
            deform_params = list(v[1] for v in geometry.named_parameters() if 'deform' in v[0]) if optimize_geometry else []
            nonrigid_params = list(v[1] for v in geometry.named_parameters() if 'nonrigid' in v[0]) if optimize_geometry else []
            msdf_params = list(v[1] for v in geometry.named_parameters() if 'msdf' in v[0]) if optimize_geometry else []
            sdf_params = list(v[1] for v in geometry.named_parameters() if 'sdf' in v[0] and 'msdf' not in v[0] and 'smpl_msdf' not in v[0]) if optimize_geometry else []
            other_params = list(v[1] for v in geometry.named_parameters() if 'sdf' not in v[0] and 'msdf' not in v[0] and 'deform' not in v[0] and 'nonrigid' not in v[0] and 'smpl_msdf' not in v[0]) if optimize_geometry else []
            optimizer_mesh = torch.optim.Adam([

                    {'params': FLAGS.trans_optim, 'lr': learning_rate_pos * 1e-3},
                    {'params': FLAGS.rhand_pose, 'lr': learning_rate_pos * 1e-3},
                    {'params': FLAGS.jaw_pose, 'lr': learning_rate_pos * 1e-3},
                    {'params': FLAGS.expr, 'lr': learning_rate_pos * 1e-3},    
                    {'params': FLAGS.body_pose, 'lr': learning_rate_pos * 1e-3},
                    {'params': FLAGS.root_pose, 'lr': learning_rate_pos * 1e-3},
                    {'params': FLAGS.lhand_pose, 'lr': learning_rate_pos * 1e-3},
                    {'params': FLAGS.leye_pose, 'lr': learning_rate_pos * 1e-3},    
                    {'params': FLAGS.reye_pose, 'lr': learning_rate_pos * 1e-3},  

                    {'params': deform_params, 'lr': learning_rate_pos},
                    {'params': sdf_params, 'lr': learning_rate_pos * 1e-2},
                    {'params': other_params, 'lr': learning_rate_pos * 1e-3},
                ], eps=1e-8)
        else:
            optimizer_mesh = torch.optim.Adam(geometry.parameters(), lr=learning_rate_pos)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    optimizer = torch.optim.Adam(params, lr=learning_rate_mat)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))

    os.makedirs(os.path.join(save_path, "loss_log"), exist_ok=True)
    writer = SummaryWriter(os.path.join(save_path, "loss_log"))

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    img_loss_vec = []
    depth_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []
    normal_loss_vec = []
    msk_loss_vec = []

    if visualize:

        dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

        def cycle(iterable):
            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(iterable)

        v_it = cycle(dataloader_validate)


    for it, target in enumerate(dataloader_train):

        target = prepare_batch_init(target, 'random', cloth_flag=False)         

        if visualize and FLAGS.local_rank == 0:
            with torch.no_grad():
                display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
                save_image = FLAGS.save_interval and (it % (FLAGS.save_interval//1) == 0)

                if display_image or save_image:

                    save_mesh = FLAGS.save_interval and (it % FLAGS.save_interval) == 0

                    if save_mesh:
                        mesh_out = geometry.getMesh_init(mat, target=target, it=it)
                        imesh = mesh_out['imesh']
                        imesh_watertight = mesh_out['imesh_watertight']

                        obj.write_obj(folder=save_path, save_name="init_imesh_wt_{}.obj".format(it), mesh=imesh_watertight, save_material=False)

                    result_image, result_dict = validate_itr(glctx, prepare_batch_init(next(v_it), FLAGS.background, cloth_flag=False), geometry, opt_material, lgt, FLAGS, denoiser=denoiser)
                    np_result_image = result_image.detach().cpu().numpy()
                    if display_image:
                        util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                    if save_image:
                        util.save_image(os.path.join(save_path, ('img_%s_%06d.png' % (pass_name, img_cnt))), np_result_image)
                        img_cnt = img_cnt + 1

        iter_start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()

        if optimize_geometry:

            optimizer_mesh.zero_grad()

        if optimize_light:
            optimizer_light.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================

        xfm_lgt = None
        if optimize_light:
            lgt.update_pdf()
            
        loss_dict = geometry.tick_init(
            glctx, target, lgt, opt_material, image_loss_fn, it, 
            denoiser=denoiser)
        
        img_loss = loss_dict["img_loss"]
        depth_loss = loss_dict["depth_loss"]
        sdf_reg_loss = loss_dict["sdf_reg_loss"]
        eik_loss = loss_dict["eik_loss"]
        msk_loss = loss_dict["msk_loss"]
        delta_loss = loss_dict["delta_loss"]
        reg_loss = loss_dict["reg_loss"]
        geo_reg_loss = loss_dict["geo_reg_loss"]
        normal_loss = loss_dict["normal_loss"]

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = reg_loss + normal_loss + msk_loss

        img_loss_vec.append(img_loss.item())
        depth_loss_vec.append(depth_loss.item())
        reg_loss_vec.append(reg_loss.item())
        normal_loss_vec.append(normal_loss.item())
        msk_loss_vec.append(msk_loss.item())
   

        writer.add_scalar('Loss/img_loss', img_loss, it)
        writer.add_scalar('Loss/depth_loss', depth_loss, it)
        writer.add_scalar('Loss/reg_loss', reg_loss, it)
        writer.add_scalar('Loss/total_loss', total_loss, it)
        writer.add_scalar('Loss/geo_reg_loss', geo_reg_loss, it)
        writer.add_scalar('Loss/geo_reg_loss', msk_loss, it)

        writer.add_scalar('Loss/sdf_reg_loss', sdf_reg_loss, it)
        writer.add_scalar('Loss/eik_loss', eik_loss, it)
        writer.add_scalar('Loss/delta_loss', delta_loss, it)
        writer.add_scalar('Loss/normal_loss', normal_loss, it)
        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        
        total_loss.backward()

        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64
            
        if 'kd_ks' in opt_material:
            opt_material['kd_ks'].encoder.params.grad /= 8.0
        if 'kd_ks_back' in opt_material:
            opt_material['kd_ks_back'].encoder.params.grad /= 8.0

        # Optionally clip gradients
        if FLAGS.clip_max_norm > 0.0:
            if optimize_geometry:
                torch.nn.utils.clip_grad_norm_(geometry.parameters() + params, FLAGS.clip_max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(params, FLAGS.clip_max_norm)

        optimizer.step()
        scheduler.step()

        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()

        if optimize_light:
            optimizer_light.step()
            scheduler_light.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'kd_back' in opt_material:
                opt_material['kd_back'].clamp_()
            if 'ks_back' in opt_material:
                opt_material['ks_back'].clamp_()
            if 'normal' in opt_material and not FLAGS.normal_only:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=1e-4) # For some reason gradient dissapears if light becomes 0

            geometry.clamp_deform()
        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        if it % log_interval == 0 and FLAGS.local_rank == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            depth_loss_avg = np.mean(np.asarray(depth_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            normal_loss_avg = np.mean(np.asarray(normal_loss_vec[-log_interval:]))
            msk_loss_avg = np.mean(np.asarray(msk_loss_vec[-log_interval:]))
            
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, img_loss=%.6f, msk_loss=%.6f, depth_loss=%.6f, reg_loss=%.6f, normal_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, img_loss_avg, msk_loss_avg, depth_loss_avg, reg_loss_avg, normal_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))
            sys.stdout.flush()

        if it == FLAGS.iter:
            break

        if it % FLAGS.save_checkpoint_interval == 0 and it != 0:
            with torch.no_grad():
                os.makedirs(os.path.join(save_path, "ckp"), exist_ok=True)
                torch.save(geometry.state_dict(), os.path.join(save_path, "ckp/model_{}.pt".format(it)))
                torch.save(mat['kd_ks'].state_dict(), os.path.join(save_path, "ckp/mtl_{}.pt".format(it)))
                light.save_env_map(os.path.join(save_path, "ckp/probe_{}.hdr".format(it)), lgt)


            file_path = os.path.join(save_path, "ckp", "smpl_{}.pt".format(it))

            np.savez(file_path,
                    trans_optim=FLAGS.trans_optim.detach().cpu().numpy(),
                    rhand_pose_optim=FLAGS.rhand_pose_optim.detach().cpu().numpy(),
                    jaw_pose_optim=FLAGS.jaw_pose_optim.detach().cpu().numpy(),
                    expr_optim=FLAGS.expr_optim.detach().cpu().numpy(),
                    body_pose_optim=FLAGS.body_pose_optim.detach().cpu().numpy(),
                    root_pose_optim=FLAGS.root_pose_optim.detach().cpu().numpy(),
                    lhand_pose_optim=FLAGS.lhand_pose_optim.detach().cpu().numpy(),
                    leye_pose_optim=FLAGS.leye_pose_optim.detach().cpu().numpy(),
                    reye_pose_optim=FLAGS.reye_pose_optim.detach().cpu().numpy(),
            )
    writer.close()

    return geometry, opt_material



def optimize_mesh_split(
        denoiser,
        glctx,
        body_geometry, 
        body_opt_material, 
        lgt,
        dataset_train,
        dataset_validate,
        FLAGS,
        warmup_iter=0,
        log_interval=10,
        pass_idx=0,
        pass_name="",
        optimize_light=True,
        optimize_geometry=True,
        optimize_camera=True,
        visualize=True,
        save_path=None
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_lgt = learning_rate[2] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate * 6.0

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter 
        return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    body_params = list(material.get_parameters(body_opt_material))
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)

    if optimize_light:
        optimizer_light = torch.optim.Adam((lgt.parameters() if lgt is not None else []), lr=learning_rate_lgt)
        scheduler_light = torch.optim.lr_scheduler.LambdaLR(optimizer_light, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    if optimize_geometry:
        if FLAGS.use_sdf_mlp:
            # ----------- cloth_geometry
            lr_msdf = learning_rate_pos * 1e-2 if FLAGS.use_msdf_mlp else learning_rate_pos

            body_deform_params = list(v[1] for v in body_geometry.named_parameters() if 'deform' in v[0]) if optimize_geometry else []
            body_nonrigid_params = list(v[1] for v in body_geometry.named_parameters() if 'nonrigid' in v[0]) if optimize_geometry else []
            body_msdf_params = list(v[1] for v in body_geometry.named_parameters() if 'msdf' in v[0]) if optimize_geometry else []
            body_other_params = list(v[1] for v in body_geometry.named_parameters() if 'sdf' not in v[0] and 'msdf' not in v[0] and 'deform' not in v[0] and 'nonrigid' not in v[0] and 'smpl_msdf' not in v[0]) if optimize_geometry else []

            optimizer_mesh = torch.optim.Adam([

                    {'params': body_deform_params, 'lr': learning_rate_pos},
                    {'params': body_msdf_params, 'lr': lr_msdf},
                    {'params': body_nonrigid_params, 'lr': learning_rate_pos * 1e-3},
                    {'params': body_other_params, 'lr': learning_rate_pos * 1e-2},
                ], eps=1e-8)
        else:
            print("-----------init optimize_geometry wrong-----------")
            exit()
            # optimizer_mesh = torch.optim.Adam(cloth_geometry.parameters(), lr=learning_rate_pos)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    optimizer = torch.optim.Adam([
            {'params': body_params, 'lr': learning_rate_mat},
         ], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))

    os.makedirs(os.path.join(save_path, "loss_log"), exist_ok=True)
    writer = SummaryWriter(os.path.join(save_path, "loss_log"))

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0

    iter_dur_vec = []

    cloth_msk_loss_vec = []
    cloth_img_loss_vec = []
    cloth_depth_loss_vec = []
    cloth_reg_loss_vec = []
    cloth_normal_loss_vec = []

    body_msk_loss_vec = []
    body_img_loss_vec = []
    body_depth_loss_vec = []
    body_reg_loss_vec = []
    body_normal_loss_vec = []

    if visualize:

        dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

        def cycle(iterable):
            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(iterable)

        v_it = cycle(dataloader_validate)

    for it, target in enumerate(dataloader_train):

        target = prepare_batch_split(target, 'random')

        if it==0:
            with torch.no_grad():

                valid_target = prepare_batch_split(next(v_it), FLAGS.background)
                result_image, result_dict = validate_itr_all(glctx, valid_target, body_geometry, body_opt_material, lgt, FLAGS, denoiser=denoiser)
                np_result_image = result_image.detach().cpu().numpy()
                util.save_image(os.path.join(save_path, ('img_%s_%06d.png' % (pass_name, 0))), np_result_image)
          

        if visualize and FLAGS.local_rank == 0 and it != 0:
            with torch.no_grad():
                display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
                save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)

                if display_image or save_image:
                    save_mesh = True

                    if save_mesh:
                        os.makedirs(os.path.join(save_path, pass_name), exist_ok=True)
                        body_mesh_out = body_geometry.getMesh_split(body_opt_material, "body", target=target, it=it)
                        imesh = body_mesh_out['imesh']
                        imesh_watertight = body_mesh_out['imesh_watertight']
                        template_imesh  = body_mesh_out['template_imesh']

                        obj.write_obj(folder=save_path, save_name="split_body_imesh_{}.obj".format(it), mesh=imesh, save_material=False)
                        # obj.write_obj(folder=save_path, save_name="split_body_imesh_wt_{}.obj".format(it), mesh=imesh_watertight, save_material=False)
                        # obj.write_obj(folder=save_path, save_name="split_body_template_imesh_{}.obj".format(it), mesh=template_imesh, save_material=False)
 

                        cloth_mesh_out = body_geometry.getMesh_split(body_opt_material, "cloth", target=target, it=it)
                        imesh = cloth_mesh_out['imesh']
                        imesh_watertight = cloth_mesh_out['imesh_watertight']
                        template_imesh  = cloth_mesh_out['template_imesh']

                        obj.write_obj(folder=save_path, save_name="split_cloth_imesh_{}.obj".format(it), mesh=imesh, save_material=False)
                        # obj.write_obj(folder=save_path, save_name="split_cloth_imesh_wt_{}.obj".format(it), mesh=imesh_watertight, save_material=False)
                        # obj.write_obj(folder=save_path, save_name="split_cloth_template_imesh_{}.obj".format(it), mesh=template_imesh, save_material=False)
 

                        # cloth_smpl_mesh_out = body_geometry.getMesh_split_smpl(body_opt_material, "cloth", target=target, it=it)
                        # imesh_smpl = cloth_smpl_mesh_out['imesh']
                        # imesh_watertight_smpl = cloth_smpl_mesh_out['imesh_watertight']
                        # template_imesh_smpl  = cloth_smpl_mesh_out['template_imesh']

                        # obj.write_obj(folder=save_path, save_name="split_cloth_smpl_imesh_{}.obj".format(it), mesh=imesh_smpl, save_material=False)
                        # obj.write_obj(folder=save_path, save_name="split_cloth_smpl_imesh_wt_{}.obj".format(it), mesh=imesh_watertight_smpl, save_material=False)
                        # obj.write_obj(folder=save_path, save_name="split_cloth_smpl_template_imesh_{}.obj".format(it), mesh=template_imesh_smpl, save_material=False)
 

                        # obj.write_obj(folder=FLAGS.out_dir, save_name="wt_imesh_{}.obj".format(it), mesh=wt_imesh, save_material=False)

                    # if it < FLAGS.train_sdf_epoch:
                    # result_image, result_dict = validate_itr(glctx, prepare_batch(next(v_it), FLAGS.background, cloth_flag=False), body_geometry, body_opt_material, lgt, FLAGS, denoiser=denoiser)
                    result_image, result_dict = validate_itr_all(glctx, target, body_geometry, body_opt_material, lgt, FLAGS, denoiser=denoiser)

                    # else:
                        # result_image, result_dict = validate_itr(glctx, prepare_batch(next(v_it), FLAGS.background, cloth_flag=True), geometry, opt_material, lgt, FLAGS, denoiser=denoiser)

                    np_result_image = result_image.detach().cpu().numpy()
                    if display_image:
                        util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                    if save_image:
                        util.save_image(os.path.join(save_path, ('img_%s_%06d.png' % (pass_name, img_cnt))), np_result_image)
                        img_cnt = img_cnt + 1

        iter_start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()

        if optimize_geometry:
            optimizer_mesh.zero_grad()

        if optimize_light:
            optimizer_light.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================

        xfm_lgt = None
        if optimize_light:
            lgt.update_pdf()
            
        
        cloth_loss_dict = body_geometry.tick_split(
            glctx, target, lgt, body_opt_material, image_loss_fn, it, 
            denoiser=denoiser, type="cloth")


        body_loss_dict = body_geometry.tick_split(
            glctx, target, lgt, body_opt_material, image_loss_fn, it, 
            denoiser=denoiser, type="body")

        
        body_msk_loss = body_loss_dict["msk_loss"] * 10
        body_img_loss = body_loss_dict["img_loss"]
        body_depth_loss = body_loss_dict["depth_loss"]
        body_sdf_reg_loss = body_loss_dict["sdf_reg_loss"]
        body_eik_loss = body_loss_dict["eik_loss"]
        body_mesh_msdf_reg_loss = body_loss_dict["mesh_msdf_reg_loss"]
        body_monochrome_loss = body_loss_dict["monochrome_loss"]
        body_mtl_smooth_loss = body_loss_dict["mtl_smooth_loss"]
        body_chroma_loss = body_loss_dict["chroma_loss"]
        body_delta_loss = body_loss_dict["delta_loss"]
        body_reg_loss = body_loss_dict["reg_loss"]
        body_geo_reg_loss = body_loss_dict["geo_reg_loss"]
        body_shading_reg_loss = body_loss_dict["shading_reg_loss"]
        body_normal_loss_mse = body_loss_dict["normal_loss_mse"]
        body_normal_loss_cos = body_loss_dict["normal_loss_cos"]
        body_normal_loss = body_loss_dict["normal_loss"]

        cloth_msk_loss = cloth_loss_dict["msk_loss"] * 10
        cloth_img_loss = cloth_loss_dict["img_loss"]
        cloth_depth_loss = cloth_loss_dict["depth_loss"]
        cloth_sdf_reg_loss = cloth_loss_dict["sdf_reg_loss"]
        cloth_eik_loss = cloth_loss_dict["eik_loss"]
        cloth_mesh_msdf_reg_loss = cloth_loss_dict["mesh_msdf_reg_loss"]
        cloth_monochrome_loss = cloth_loss_dict["monochrome_loss"]
        cloth_mtl_smooth_loss = cloth_loss_dict["mtl_smooth_loss"]
        cloth_chroma_loss = cloth_loss_dict["chroma_loss"]
        cloth_delta_loss = cloth_loss_dict["delta_loss"]
        cloth_reg_loss = cloth_loss_dict["reg_loss"]
        cloth_geo_reg_loss = cloth_loss_dict["geo_reg_loss"]
        cloth_shading_reg_loss = cloth_loss_dict["shading_reg_loss"]
        cloth_normal_loss_mse = cloth_loss_dict["normal_loss_mse"]
        cloth_normal_loss_cos = cloth_loss_dict["normal_loss_cos"]
        cloth_normal_loss = cloth_loss_dict["normal_loss"]

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = cloth_img_loss  + cloth_normal_loss + cloth_reg_loss + body_img_loss + body_normal_loss + body_reg_loss + cloth_msk_loss + body_msk_loss#+ test_loss

        cloth_msk_loss_vec.append(cloth_msk_loss.item())
        cloth_img_loss_vec.append(cloth_img_loss.item())
        cloth_depth_loss_vec.append(cloth_depth_loss.item())
        cloth_reg_loss_vec.append(cloth_reg_loss.item())
        cloth_normal_loss_vec.append(cloth_normal_loss.item())

        body_msk_loss_vec.append(body_msk_loss.item())
        body_img_loss_vec.append(body_img_loss.item())
        body_depth_loss_vec.append(body_depth_loss.item())
        body_reg_loss_vec.append(body_reg_loss.item())
        body_normal_loss_vec.append(body_normal_loss.item())

        writer.add_scalar('Loss/total_loss', total_loss, it)

        writer.add_scalar('Loss/body_msk_loss', body_msk_loss, it)
        writer.add_scalar('Loss/body_img_loss', body_img_loss, it)
        writer.add_scalar('Loss/body_depth_loss', body_depth_loss, it)
        writer.add_scalar('Loss/body_reg_loss', body_reg_loss, it)
        writer.add_scalar('Loss/body_geo_reg_loss', body_geo_reg_loss, it)
        writer.add_scalar('Loss/body_shading_reg_loss', body_shading_reg_loss, it)

        writer.add_scalar('Loss/body_normal_loss_cos', body_normal_loss_cos, it)
        writer.add_scalar('Loss/body_normal_loss_mse', body_normal_loss_mse, it)
        writer.add_scalar('Loss/body_normal_loss', body_normal_loss, it)

        writer.add_scalar('Loss/body_sdf_reg_loss', body_sdf_reg_loss, it)
        writer.add_scalar('Loss/body_eik_loss', body_eik_loss, it)
        writer.add_scalar('Loss/body_mesh_msdf_reg_loss', body_mesh_msdf_reg_loss, it)
        writer.add_scalar('Loss/body_monochrome_loss', body_monochrome_loss, it)
        writer.add_scalar('Loss/body_mtl_smooth_loss', body_mtl_smooth_loss, it)
        writer.add_scalar('Loss/body_chroma_loss', body_chroma_loss, it)
        writer.add_scalar('Loss/body_delta_loss', body_delta_loss, it)

        writer.add_scalar('Loss/cloth_msk_loss', cloth_msk_loss, it)
        writer.add_scalar('Loss/cloth_img_loss', cloth_img_loss, it)
        writer.add_scalar('Loss/cloth_depth_loss', cloth_depth_loss, it)
        writer.add_scalar('Loss/cloth_reg_loss', cloth_reg_loss, it)
        writer.add_scalar('Loss/cloth_geo_reg_loss', cloth_geo_reg_loss, it)
        writer.add_scalar('Loss/cloth_shading_reg_loss', cloth_shading_reg_loss, it)

        writer.add_scalar('Loss/cloth_normal_loss_cos', cloth_normal_loss_cos, it)
        writer.add_scalar('Loss/cloth_normal_loss_mse', cloth_normal_loss_mse, it)
        writer.add_scalar('Loss/cloth_normal_loss', cloth_normal_loss, it)

        writer.add_scalar('Loss/cloth_sdf_reg_loss', cloth_sdf_reg_loss, it)
        writer.add_scalar('Loss/cloth_eik_loss', cloth_eik_loss, it)
        writer.add_scalar('Loss/cloth_mesh_msdf_reg_loss', cloth_mesh_msdf_reg_loss, it)
        writer.add_scalar('Loss/cloth_monochrome_loss', cloth_monochrome_loss, it)
        writer.add_scalar('Loss/cloth_mtl_smooth_loss', cloth_mtl_smooth_loss, it)
        writer.add_scalar('Loss/cloth_chroma_loss', cloth_chroma_loss, it)
        writer.add_scalar('Loss/cloth_delta_loss', cloth_delta_loss, it)
        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        
        total_loss.backward()

        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64

        if 'kd_ks' in body_opt_material:
            body_opt_material['kd_ks'].encoder.params.grad /= 8.0
        if 'kd_ks_back' in body_opt_material:
            body_opt_material['kd_ks_back'].encoder.params.grad /= 8.0

        # Optionally clip gradients
        if FLAGS.clip_max_norm > 0.0:

            if optimize_geometry:
                torch.nn.utils.clip_grad_norm_(body_geometry.parameters() + body_params, FLAGS.clip_max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(body_params, FLAGS.clip_max_norm)


        optimizer.step()
        scheduler.step()

        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()

        if optimize_light:
            optimizer_light.step()
            scheduler_light.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():

            if 'kd' in body_opt_material:
                body_opt_material['kd'].clamp_()
            if 'ks' in body_opt_material:
                body_opt_material['ks'].clamp_()
            if 'kd_back' in body_opt_material:
                body_opt_material['kd_back'].clamp_()
            if 'ks_back' in body_opt_material:
                body_opt_material['ks_back'].clamp_()
            if 'normal' in body_opt_material and not FLAGS.normal_only:
                body_opt_material['normal'].clamp_()
                body_opt_material['normal'].normalize_()

            if lgt is not None:
                lgt.clamp_(min=1e-4) # For some reason gradient dissapears if light becomes 0

            body_geometry.clamp_deform()
        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        if it % log_interval == 0 and FLAGS.local_rank == 0:

            cloth_msk_loss_avg = np.mean(np.asarray(cloth_msk_loss_vec[-log_interval:]))
            cloth_img_loss_avg = np.mean(np.asarray(cloth_img_loss_vec[-log_interval:]))
            cloth_reg_loss_avg = np.mean(np.asarray(cloth_reg_loss_vec[-log_interval:]))
            cloth_normal_loss_avg = np.mean(np.asarray(cloth_normal_loss_vec[-log_interval:]))

            body_msk_loss_avg = np.mean(np.asarray(body_msk_loss_vec[-log_interval:]))
            body_img_loss_avg = np.mean(np.asarray(body_img_loss_vec[-log_interval:]))
            body_reg_loss_avg = np.mean(np.asarray(body_reg_loss_vec[-log_interval:]))
            body_normal_loss_avg = np.mean(np.asarray(body_normal_loss_vec[-log_interval:]))

            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, cloth_img_loss=%.6f, cloth_msk_loss=%.6f, cloth_reg_loss=%.6f, cloth_normal_loss=%.6f, body_img_loss=%.6f, body_msk_loss=%.6f, body_reg_loss=%.6f, body_normal_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, cloth_img_loss_avg, cloth_msk_loss_avg, cloth_reg_loss_avg, cloth_normal_loss_avg, 
                 body_img_loss_avg, body_msk_loss_avg, body_reg_loss_avg, body_normal_loss_avg, 
                 optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))
            sys.stdout.flush()

        if it == FLAGS.iter:
            break

        if it % FLAGS.save_checkpoint_interval == 0 and it != 0:
            with torch.no_grad():
                os.makedirs(os.path.join(save_path, "ckp"), exist_ok=True)
                torch.save(body_geometry.state_dict(), os.path.join(save_path, "ckp/model_{}.pt".format(it)))
                torch.save(body_opt_material['kd_ks'].state_dict(), os.path.join(save_path, "ckp/mtl_{}.pt".format(it)))
                light.save_env_map(os.path.join(save_path, "ckp/probe_{}.hdr".format(it)), lgt)

            file_path = os.path.join(save_path, "ckp", "smpl_{}.pt".format(it))
            # 调用 np.savez 保存优化参数
            # np.savez(file_path,
            #         smpl_beta_optim=FLAGS.smpl_beta_optim.detach().cpu().numpy(),
            #         smpl_global_orient_optim=FLAGS.smpl_global_orient_optim.detach().cpu().numpy(),
            #         smpl_body_pose_optim=FLAGS.smpl_body_pose_optim.detach().cpu().numpy(),
            #         smpl_transl_optim=FLAGS.smpl_transl_optim.detach().cpu().numpy(),
            # )

    writer.close()

    return body_geometry, body_opt_material


def optimize_mesh_seq(
        first,
        it,
        target,
        denoiser,
        glctx,
        body_geometry, 
        body_opt_material, 
        lgt,
        dataset_train,
        dataset_validate,
        FLAGS,
        warmup_iter=0,
        log_interval=10,
        pass_idx=0,
        pass_name="",
        optimize_light=True,
        optimize_geometry=True,
        optimize_camera=True,
        visualize=True,
        save_path=None
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_lgt = learning_rate[2] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate * 6.0

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter 
        return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    body_params = list(material.get_parameters(body_opt_material))

    if optimize_light:
        optimizer_light = torch.optim.Adam((lgt.parameters() if lgt is not None else []), lr=learning_rate_lgt)
        scheduler_light = torch.optim.lr_scheduler.LambdaLR(optimizer_light, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    if optimize_geometry:
        if FLAGS.use_sdf_mlp:
            body_nonrigid_params = list(v[1] for v in body_geometry.named_parameters() if 'nonrigid' in v[0]) if optimize_geometry else []
            body_cond_params = list(v[1] for v in body_geometry.named_parameters() if 'cond' in v[0]) if optimize_geometry else []

            optimizer_mesh = torch.optim.Adam([

                    {'params': body_nonrigid_params, 'lr': learning_rate_pos * 1e-2},
                    {'params': body_cond_params, 'lr': learning_rate_pos * 1e-2},

                ], eps=1e-8)
        else:
            print("-----------init optimize_geometry wrong-----------")
            exit()
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    optimizer = torch.optim.Adam([
            {'params': body_params, 'lr': learning_rate_mat},
         ], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))

    os.makedirs(os.path.join(save_path, "loss_log"), exist_ok=True)
    writer = SummaryWriter(os.path.join(save_path, "loss_log"))

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================

    iter_dur_vec = []
    img_loss_vec = []
    reg_loss_vec = []
    normal_loss_vec = []
    msk_loss_vec = []
    body_laplacian_loss_vec = []
    cloth_laplacian_loss_vec = []

    if visualize:

        dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

        def cycle(iterable):
            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(iterable)


    target = prepare_batch_split(target, 'black')

    # ==============================================================================================
    #  Display / save outputs. Do it before training so we get initial meshes
    # ==============================================================================================

    if it==0:
        every_time = 1000
    else:
        every_time = 300

    for times in range(every_time):

        if times==(every_time-1):

            result_image, result_dict, all_mesh = validate_all_mesh(glctx, target, body_geometry, body_opt_material, lgt, FLAGS, denoiser=None, t="all")
            tmp_all_mesh = result_dict["tmp_all_mesh"]
            obj.write_ply(folder=save_path, save_name="fine_all_{}.ply".format(it), mesh=all_mesh)
            obj.write_ply(folder=save_path, save_name="tmp_all_{}.ply".format(it), mesh=tmp_all_mesh)
            np_result_image = result_image.detach().cpu().numpy()
            util.save_image(os.path.join(save_path, ('all_it%06d_times%06d.png' % (it, times))), np_result_image)

        iter_start_time = time.time()

    # ==============================================================================================
    #  Zero gradients
    # ==============================================================================================
    
        optimizer.zero_grad()


        if optimize_geometry:
            optimizer_mesh.zero_grad()

        if optimize_light:
            optimizer_light.zero_grad()

    # ==============================================================================================
    #  Training
    # ==============================================================================================

        xfm_lgt = None
        if optimize_light:
            lgt.update_pdf()

        body_loss_dict = body_geometry.tick_seq(
            glctx, target, lgt, body_opt_material, image_loss_fn, times, 
            denoiser=denoiser, t="all")
        
        visible_triangles = body_loss_dict["visible_triangles"]

        all_img_loss = body_loss_dict["all_img_loss"]
        all_msk_loss = body_loss_dict["all_msk_loss"]
        cloth_img_loss = body_loss_dict["cloth_img_loss"]
        cloth_msk_loss = body_loss_dict["cloth_msk_loss"]
        body_img_loss = body_loss_dict["body_img_loss"]
        body_msk_loss = body_loss_dict["body_msk_loss"]
        delta_loss = body_loss_dict["delta_loss"]

        colli_loss = body_loss_dict["colli_loss"]
        reg_loss = body_loss_dict["reg_loss"]
        laplacian_loss = body_loss_dict["laplacian_loss"]
        body_normal_loss = body_loss_dict["normal_loss"]
        nds_normal_loss = body_loss_dict["nds_normal_loss"]


    # ==============================================================================================
    #  Final loss
    # ==============================================================================================
        normal_loss = 250 * body_normal_loss #+ cloth_normal_loss *5
        reg_loss = 0.1 * reg_loss 
        msk_loss = body_msk_loss + cloth_msk_loss + all_msk_loss
        img_loss = body_img_loss + cloth_img_loss + all_img_loss
        laplacian_loss = 1000000*laplacian_loss 
        colli_loss = 100000*colli_loss
        nds_normal_loss = 1000*nds_normal_loss
        delta_loss = delta_loss

        total_loss = normal_loss + reg_loss + msk_loss + laplacian_loss + colli_loss + nds_normal_loss + delta_loss #+ cloth_bodylaplacian_loss 


        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())
        normal_loss_vec.append(normal_loss.item())
        msk_loss_vec.append(msk_loss.item())

        body_laplacian_loss_vec.append(laplacian_loss.item())

        writer.add_scalar('Loss/reg_loss', reg_loss, it)
        writer.add_scalar('Loss/normal_loss', normal_loss, it)
        writer.add_scalar('Loss/msk_loss', msk_loss, it)




    # ==============================================================================================
    #  Backpropagate
    # ==============================================================================================
    
        total_loss.backward()

        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64
        if FLAGS.clip_max_norm > 0.0:

            if optimize_geometry:
                torch.nn.utils.clip_grad_norm_(body_geometry.parameters() + body_params, FLAGS.clip_max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(body_params, FLAGS.clip_max_norm)

        optimizer.step()
        scheduler.step()

        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()

        if optimize_light:
            optimizer_light.step()
            scheduler_light.step()

    # ==============================================================================================
    #  Clamp trainables to reasonable range
    # ==============================================================================================
        with torch.no_grad():

            if 'kd' in body_opt_material:
                body_opt_material['kd'].clamp_()
            if 'ks' in body_opt_material:
                body_opt_material['ks'].clamp_()
            if 'kd_back' in body_opt_material:
                body_opt_material['kd_back'].clamp_()
            if 'ks_back' in body_opt_material:
                body_opt_material['ks_back'].clamp_()
            if 'normal' in body_opt_material and not FLAGS.normal_only:
                body_opt_material['normal'].clamp_()
                body_opt_material['normal'].normalize_()


            if lgt is not None:
                lgt.clamp_(min=1e-4) # For some reason gradient dissapears if light becomes 0

            body_geometry.clamp_deform()
        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)


    # ==============================================================================================
    #  Logging
    # ==============================================================================================
        if times % log_interval == 0 and FLAGS.local_rank == 0:

            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            normal_loss_avg = np.mean(np.asarray(normal_loss_vec[-log_interval:]))
            msk_loss_avg = np.mean(np.asarray(msk_loss_vec[-log_interval:]))
            body_laplacian_loss_avg = np.mean(np.asarray(body_laplacian_loss_vec[-log_interval:]))
            cloth_laplacian_loss_avg = np.mean(np.asarray(cloth_laplacian_loss_vec[-log_interval:]))

            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, times=%5d, lr=%.5f, total_loss=%.6f, img_loss=%.6f, reg_loss=%.6f, normal_loss=%.6f, msk_loss=%.6f, body_laplacian=%.6f, colli_loss=%.6f, nds_normal_loss=%.6f, delta_loss=%.6f, time=%.1f ms, rem=%s" % 
                (it, times, optimizer_mesh.param_groups[0]['lr'], total_loss,
                img_loss_avg, reg_loss_avg, normal_loss_avg, msk_loss_avg, body_laplacian_loss_avg, colli_loss, nds_normal_loss, delta_loss,
                iter_dur_avg*1000, util.time_to_text(remaining_time)))
            sys.stdout.flush()

    writer.close()

    with torch.no_grad():
        os.makedirs(os.path.join(save_path, "delta"), exist_ok=True)            
        visible_triangles_npz = visible_triangles.detach().cpu().numpy()
        delta_npz = body_loss_dict["delta"].detach().cpu().numpy()
        # print("delta_npz:", delta_npz)
        save_delta_path = os.path.join(os.path.join(save_path, "delta"), "{}.npz".format(it))
        np.savez(save_delta_path, delta=delta_npz, visible_triangles=visible_triangles_npz)

    return body_geometry, body_opt_material


#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--bsdf', type=str, default='pbr', choices=['pbr', 'diffuse', 'white'])
    parser.add_argument('--denoiser', default='bilateral', choices=['none', 'bilateral'])
    parser.add_argument('--denoiser_demodulate', type=bool, default=True)
    parser.add_argument('--index',type=int)
    parser.add_argument('--msdf_reg_open_scale', type=float, default=1e-6)
    parser.add_argument('--msdf_reg_close_scale', type=float, default=3e-6)
    parser.add_argument('--eikonal_scale', type=float)
    parser.add_argument('--sdf_regularizer', type=float, default=0.2)
    parser.add_argument('--trainset_path', type=str, default='./data')
    parser.add_argument('--testset_path', type=str, default='./data')
    parser.add_argument('--folder_name', type=str, default='')
    parser.add_argument('--fine_folder', type=str, default='fine')
    parser.add_argument('--seq_folder', type=str, default='seq')
    parser.add_argument('--split_folder', type=str, default='split')

    FLAGS = parser.parse_args()
    FLAGS.mtl_override        = None        # Override material of model
    FLAGS.gshell_grid          = 64          # Resolution of initial tet grid. We provide 64 and 128 resolution grids. 
                                            #    Other resolutions can be generated with https://github.com/crawforddoran/quartet
                                            #    We include examples in data/tets/generate_tets.py
    # FLAGS.mesh_scale          = 1.4         # Scale of tet grid box. Adjust to cover the model
    FLAGS.mesh_scale          = 1.0
    FLAGS.envlight            = None        # HDR environment probe
    FLAGS.env_scale           = 1.0         # Env map intensity multiplier
    FLAGS.probe_res           = 256         # Env map probe resolution
    FLAGS.learn_lighting      = True        # Enable optimization of env lighting
    FLAGS.display             = None        # Configure validation window/display. E.g. [{"bsdf" : "kd"}, {"bsdf" : "ks"}]
    FLAGS.transparency        = False       # Enabled transparency through depth peeling
    FLAGS.lock_light          = False       # Disable light optimization in the second pass
    FLAGS.lock_pos            = False       # Disable vertex position optimization in the second pass
    # FLAGS.sdf_regularizer     = 0.2         # Weight for sdf regularizer.
    FLAGS.laplace             = "relative"  # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 3000.0      # Weight for Laplace regularizer. Default is relative with large weight
    FLAGS.pre_load            = True        # Pre-load entire dataset into memory for faster training
    FLAGS.no_perturbed_nrm    = False       # Disable normal map
    FLAGS.decorrelated        = False       # Use decorrelated sampling in forward and backward passes
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0]
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0,  0.001, 0.0]
    FLAGS.ks_max              = [ 0.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.clip_max_norm       = 0.0
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.lambda_kd           = 0.1
    FLAGS.lambda_ks           = 0.05
    FLAGS.lambda_nrm          = 0.025
    FLAGS.lambda_nrm2         = 0.25
    FLAGS.lambda_chroma       = 0.0
    FLAGS.lambda_diffuse      = 0.15
    FLAGS.lambda_specular     = 0.0025

    FLAGS.random_lgt                  = False
    FLAGS.normal_only                 = False
    FLAGS.use_img_2nd_layer           = False
    FLAGS.use_depth                   = False
    FLAGS.use_depth_2nd_layer         = False
    FLAGS.use_tanh_deform             = False
    FLAGS.use_sdf_mlp                 = True
    FLAGS.use_msdf_mlp                = False
    FLAGS.use_eikonal                 = True
    FLAGS.sdf_mlp_pretrain_steps      = 3000

    FLAGS.sdf_mlp_pretrain_smpl_steps = 3000
    FLAGS.pretrain_smpl               = True
    FLAGS.sdf_deform_pretrain_steps   = 1000
    FLAGS.use_nonrigid_deform         = True
    FLAGS.use_mesh_msdf_reg           = True
    FLAGS.sphere_init                 = False
    FLAGS.sphere_init_norm            = 0.5
    FLAGS.pretrained_sdf_mlp_path     = f'./data/pretrained_mlp_{FLAGS.gshell_grid}_deeper.pt'
    FLAGS.n_hidden                    = 6
    FLAGS.d_hidden                    = 256
    FLAGS.n_freq                      = 6
    FLAGS.skip_in                     = [3]
    FLAGS.use_float16                 = False
    FLAGS.visualize_watertight        = True
    FLAGS.save_checkpoint_interval    = 500
    FLAGS.save_checkpoint_interval_fine  = 100
    FLAGS.train_sdf_epoch             = 2000

    FLAGS.nonrigid_begin              = 20000

    FLAGS.init_epoch                  = 2001
    FLAGS.split_epoch                 = 1001
    FLAGS.split_smpl_epoch            = 1001
    FLAGS.fine_epoch                  = 1001
    FLAGS.seq_epoch                   = 50001

    FLAGS.view_seq                    = True

    print("FLAGS.loss:", FLAGS.loss) # logl1


    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext()
    glctx_display = glctx if FLAGS.batch < 16 else dr.RasterizeGLContext() # Context for display

    mtl_default = None

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    dataset_path = FLAGS.trainset_path
    testset_path = FLAGS.testset_path

    ##################################################################
    # ----------------------- init data -----------------------------
    ##################################################################

    folder_name = FLAGS.folder_name
    FLAGS.gender = folder_name.split("/")[-1].split("-")[0]
    data_root = os.path.join(dataset_path, folder_name)

    dataset_train_init   = Dataset_split(data_root, FLAGS, examples=int(FLAGS.init_epoch))
    dataset_validate_init = Dataset_split(data_root, FLAGS)
    n_images = dataset_train_init.n_images

    FLAGS.smplx_params = dataset_train_init.smplx_params

    FLAGS.shape_param = FLAGS.smplx_params["shape_param"]
    FLAGS.face_offset = FLAGS.smplx_params["face_offset"]
    FLAGS.joint_offset = FLAGS.smplx_params["joint_offset"]
    FLAGS.locator_offset = FLAGS.smplx_params["locator_offset"]
    FLAGS.trans = FLAGS.smplx_params["trans"]
    FLAGS.rhand_pose = FLAGS.smplx_params["rhand_pose"]
    FLAGS.jaw_pose = FLAGS.smplx_params["jaw_pose"]
    FLAGS.reye_pose = FLAGS.smplx_params["reye_pose"]
    FLAGS.expr = FLAGS.smplx_params["expr"]
    FLAGS.body_pose = FLAGS.smplx_params["body_pose"]
    FLAGS.root_pose = FLAGS.smplx_params["root_pose"]
    FLAGS.lhand_pose = FLAGS.smplx_params["lhand_pose"]
    FLAGS.leye_pose = FLAGS.smplx_params["leye_pose"]

    FLAGS.shape_param_optim = FLAGS.smplx_params["shape_param"].cuda()
    FLAGS.face_offset_optim = FLAGS.smplx_params["face_offset"].cuda()
    FLAGS.joint_offset_optim = FLAGS.smplx_params["joint_offset"].cuda()
    FLAGS.locator_offset_optim = FLAGS.smplx_params["locator_offset"].cuda()
    FLAGS.trans_optim = FLAGS.smplx_params["trans"].cuda().requires_grad_(True)
    FLAGS.rhand_pose_optim = FLAGS.smplx_params["rhand_pose"].cuda().requires_grad_(True)
    FLAGS.jaw_pose_optim = FLAGS.smplx_params["jaw_pose"].cuda().requires_grad_(True)
    FLAGS.reye_pose_optim = FLAGS.smplx_params["reye_pose"].cuda().requires_grad_(True)
    FLAGS.expr_optim = FLAGS.smplx_params["expr"].cuda().requires_grad_(True)
    FLAGS.body_pose_optim = FLAGS.smplx_params["body_pose"].cuda().requires_grad_(True)
    FLAGS.root_pose_optim = FLAGS.smplx_params["root_pose"].cuda().requires_grad_(True)
    FLAGS.lhand_pose_optim = FLAGS.smplx_params["lhand_pose"].cuda().requires_grad_(True)
    FLAGS.leye_pose_optim = FLAGS.smplx_params["leye_pose"].cuda().requires_grad_(True)

    FLAGS.n_images = n_images

    if FLAGS.testset_path is not None and FLAGS.testset_path != '':
        testdata_root = os.path.join(testset_path, folder_name)
        dataset_test_init = Dataset_split(testdata_root, FLAGS)

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    
    lgt = None
    if FLAGS.learn_lighting:
        lgt = light.create_trainable_env_rnd(FLAGS.probe_res, scale=0.0, bias=0.5)
    else:
        lgt = light.load_env(FLAGS.envlight, scale=FLAGS.env_scale, res=[FLAGS.probe_res, FLAGS.probe_res])

    # ==============================================================================================
    #  Setup denoiser
    # ==============================================================================================

    denoiser = None
    if FLAGS.denoiser == 'bilateral':
        denoiser = BilateralDenoiser().cuda()
    else:
        assert FLAGS.denoiser == 'none', "Invalid denoiser %s" % FLAGS.denoiser

    # Setup geometry for optimization
    geometry = HmSDFTetsGeometry(FLAGS.gshell_grid, FLAGS.mesh_scale, FLAGS)

    if not FLAGS.normal_only:
        mat = initial_guess_material(geometry, True, FLAGS, mtl_default)
    else:
        mat = initial_guess_material_knownkskd(geometry, True, FLAGS, mtl_default)
    mat['no_perturbed_nrm'] = True

    mesh_out = geometry.getMesh_init(mat)
    init_imesh = mesh_out['imesh']
    init_wt_imesh = mesh_out['imesh_watertight']
    obj.write_obj(folder=FLAGS.out_dir, save_name="init_imesh.obj", mesh=init_imesh, save_material=False)
    obj.write_obj(folder=FLAGS.out_dir, save_name="init_wt_imesh.obj", mesh=init_wt_imesh, save_material=False)


    ##################################################################
    # ----------------------- split stage smpl -----------------------
    ##################################################################

    dataset_train_split   = Dataset_split(data_root, FLAGS, examples=int(FLAGS.split_smpl_epoch))
    dataset_validate_split = Dataset_split(data_root, FLAGS)

    geometry, mat = optimize_mesh_split(denoiser, glctx, geometry, mat, lgt, dataset_train_split, dataset_validate_split, 
                    FLAGS, pass_idx=0, pass_name="pass1", optimize_light=FLAGS.learn_lighting, save_path=os.path.join(FLAGS.out_dir, "split_smpl"))

    process_root = os.path.join(FLAGS.out_dir, "process_smpl")
    os.makedirs(process_root, exist_ok=True)
    body_path = os.path.join(FLAGS.out_dir, "split_smpl", "split_body_template_imesh_{}.obj".format(FLAGS.split_smpl_epoch-1))
    cloth_path = os.path.join(FLAGS.out_dir, "split_smpl", "split_cloth_template_imesh_{}.obj".format(FLAGS.split_smpl_epoch-1))
    smpl_path = os.path.join(FLAGS.out_dir, "smpl_template_128.obj")

    closehole_root = os.path.join(FLAGS.out_dir, "close_hole_smpl")
    os.makedirs(closehole_root, exist_ok=True)
    split_body_mesh = os.path.join(FLAGS.out_dir, "split_smpl", "split_body_imesh_{}.obj".format(FLAGS.split_smpl_epoch-1))
    split_cloth_mesh = os.path.join(FLAGS.out_dir, "split_smpl", "split_cloth_imesh_{}.obj".format(FLAGS.split_smpl_epoch-1))
    body_mesh = os.path.join(FLAGS.out_dir, "split_smpl", "split_body_template_imesh_1800.obj")
    cloth_mesh = os.path.join(FLAGS.out_dir, "split_smpl", "split_cloth_template_imesh_1800.obj")
    
    body_path, cloth_path, bbox_path = process_close_hole(FLAGS, closehole_root, split_body_mesh, split_cloth_mesh)



    ##################################################################
    # ----------------------- init stage -----------------------------
    ##################################################################

    # Run optimization

    geometry._init_msdf()
    geometry._init_sdf()
    geometry, mat = optimize_mesh_init(denoiser, glctx, geometry, mat, lgt, dataset_train_init, dataset_validate_init, 
                    FLAGS, pass_idx=0, pass_name="pass1", optimize_light=FLAGS.learn_lighting, save_path=os.path.join(FLAGS.out_dir, "init"))


    ##################################################################
    # ----------------------- split stage -----------------------------
    ##################################################################
    
    for clothtype in [0]:

        FLAGS.cloth_type = clothtype
        if FLAGS.cloth_type == 0:
            com_num=4    
            FLAGS.split_folder = "split_cloth0"
        elif FLAGS.cloth_type == 1:
            com_num=4
            FLAGS.split_folder = "split_cloth1"
        elif FLAGS.cloth_type == 2:
            com_num=3
            FLAGS.split_folder = "split_cloth2"

        geometry, mat, lgt = load_ckp(FLAGS, FLAGS.out_dir, geometry, mat, stage="init")

        geometry, mat = optimize_mesh_split(denoiser, glctx, geometry, mat, lgt, dataset_train_split, dataset_validate_split, 
                        FLAGS, pass_idx=0, pass_name="pass1", optimize_light=FLAGS.learn_lighting, save_path=os.path.join(FLAGS.out_dir, FLAGS.split_folder))

        process_root = os.path.join(FLAGS.out_dir, "process"+FLAGS.split_folder)
        os.makedirs(process_root, exist_ok=True)
        body_path = os.path.join(FLAGS.out_dir, FLAGS.split_folder, "split_body_imesh_{}.obj".format(FLAGS.split_epoch-1))
        cloth_path = os.path.join(FLAGS.out_dir, FLAGS.split_folder, "split_cloth_imesh_{}.obj".format(FLAGS.split_epoch-1))
        smpl_path = os.path.join(FLAGS.out_dir, "smpl_template_128.obj")

        closehole_root = os.path.join(FLAGS.out_dir, "close_hole_"+FLAGS.split_folder)
        os.makedirs(closehole_root, exist_ok=True)
        split_body_mesh = os.path.join(FLAGS.out_dir, FLAGS.split_folder, "split_body_imesh_{}.obj".format(FLAGS.split_epoch-1))
        split_cloth_mesh = os.path.join(FLAGS.out_dir, FLAGS.split_folder, "split_cloth_imesh_{}.obj".format(FLAGS.split_epoch-1))
        
        print("split_body_mesh:", split_body_mesh)
        print("split_cloth_mesh:", split_cloth_mesh)
        body_path, cloth_path, bbox_path = process_close_hole(FLAGS, closehole_root, split_body_mesh, split_cloth_mesh, num=5)
        bbox_path = os.path.join(closehole_root, "bbox.npz")

        cloth_path = os.path.join(FLAGS.out_dir, "close_hole_"+FLAGS.split_folder, "cloth_concat.obj")
        ori_cloth_path = cloth_path
        smpl_cloth_path = os.path.join(FLAGS.out_dir, "close_hole_smpl", "cloth_concat.obj")
        bbox_npz_path = bbox_path
        body_path = os.path.join(FLAGS.out_dir, "close_hole_"+FLAGS.split_folder, "body_concat.obj")

        print("body_path:", body_path)
        print("ori_cloth_path:", ori_cloth_path)
        print("smpl_cloth_path:", smpl_cloth_path)
        print("smpl_path:", smpl_path)
    
        if FLAGS.cloth_type == 0:
            process_body_msdf_distance_bodyedge(FLAGS, process_root, body_path, ori_cloth_path, smpl_cloth_path, smpl_path, bbox_npz_path)


    ##################################################################
    # ----------------------- seq stage -----------------------------
    ##################################################################

    v,f = om_loadmesh(os.path.join(FLAGS.out_dir, "processsplit_cloth0/merge_body_cloth.obj"))
    
    FLAGS.fix = False
    FLAGS.v = torch.from_numpy(v).cuda().float()
    FLAGS.f = torch.from_numpy(f).cuda().long()

    dataset_train_seq   = Dataset_split(data_root, FLAGS, process_path=process_root, Detail=True, examples=int(FLAGS.seq_epoch-1))
    dataset_validate_seq = Dataset_split(data_root, FLAGS, process_path=process_root, Detail=True)

    FLAGS.face_labels = dataset_train_seq.face_labels
    FLAGS.inside_body_index = dataset_train_seq.inside_body_index
    FLAGS.outside_body_index = dataset_train_seq.outside_body_index
    FLAGS.outside_index = dataset_train_seq.outside_index

    v = FLAGS.v
    f = FLAGS.f
    face_labels = FLAGS.face_labels

    v = v.float().cuda()
    f = f.to(torch.int64).cuda()
    FLAGS.body_f = FLAGS.f[face_labels==0]
    FLAGS.cloth_f = FLAGS.f[face_labels==1]

    #---------------------------
    num_vertices = v.shape[0]
    num_faces = f.shape[0]
    face_vertex_indices = f.reshape(-1) 
    face_labels_expanded = face_labels.unsqueeze(1).expand(-1, 3).reshape(-1) 
    num_labels = face_labels.max().item() + 1
    indices = face_vertex_indices * num_labels + face_labels_expanded  
    counts = torch.bincount(indices, minlength=num_vertices * num_labels)
    vertex_label_counts = counts.reshape(num_vertices, num_labels)
    vertex_labels = vertex_label_counts.argmax(dim=1)  

    connected_faces, edges = find_connected_faces(f)

    v_1 = v[vertex_labels==1] # cloth
    v_0 = v[vertex_labels==0] # body

    FLAGS.body_v = v_0
    FLAGS.cloth_v = v_1
    FLAGS.v_labels = vertex_labels.cuda()
    FLAGS.connected_faces = connected_faces.cuda()
    FLAGS.edges = edges.cuda()

    geometry._init_basedeform(v, f, FLAGS.body_v, FLAGS.cloth_v)
    geometry._init_use_body_nonrigid_deform()

    FLAGS.fix = False

    dataloader_train    = torch.utils.data.DataLoader(dataset_train_seq, batch_size=FLAGS.batch, collate_fn=dataset_train_seq.collate, shuffle=False)
    first = dataset_train_seq.begin
    last = dataset_train_seq.end
    geometry._init_basedeform(v, f, FLAGS.body_v, FLAGS.cloth_v)


    for it, target in enumerate(dataloader_train):
        if it>=20:
            break
        FLAGS.num = it
        geometry, mat = optimize_mesh_seq(first, it, target, denoiser, glctx, geometry, mat, lgt, dataset_train_seq, dataset_validate_seq, 
                FLAGS, pass_idx=0, warmup_iter=0, pass_name="pass1", optimize_light=FLAGS.learn_lighting, save_path=os.path.join(FLAGS.out_dir, FLAGS.seq_folder))
