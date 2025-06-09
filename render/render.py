# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from threading import local
import numpy as np
import torch
import nvdiffrast.torch as dr
import time

from . import util
from . import renderutils as ru
from . import optixutils as ou
from . import light

rnd_seed = 0

def write_pc(path_mesh, v, vn=None, f=None):
    # print("---------v:", v.shape)
    assert v.ndim == 2 and v.shape[1] == 3
    with open(path_mesh, 'w') as fp:
        fp.write(('v {:f} {:f} {:f}\n' * v.shape[0]).format(*v.reshape(-1)))
        if vn is not None:
            fp.write(('vn {:f} {:f} {:f}\n' * vn.shape[0]).format(*vn.reshape(-1)))
        if f is not None:
            fp.write(('f {:d} {:d} {:d}\n' * f.shape[0]).format(*f.reshape(-1) + 1))

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        FLAGS,
        idx,
        rast,
        gb_depth,
        gb_pos,
        gb_pos_original,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        lgt,
        material,
        optix_ctx,
        mesh,
        bsdf,
        denoiser,
        shadow_scale,
        use_uv=True,
        finetune_normal=True,
        xfm_lgt=None,
        shade_data=False
    ):

    offset = torch.normal(mean=0, std=0.005, size=(gb_depth.shape[0], gb_depth.shape[1], gb_depth.shape[2], 2), device="cuda")
    jitter = (util.pixel_grid(gb_depth.shape[2], gb_depth.shape[1])[None, ...] + offset).contiguous()

    mask = (rast[..., -1:] > 0).float()
    mask_tap = dr.texture(mask.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
    grad_weight = mask * mask_tap

    ################################################################################
    # Texture lookups
    ################################################################################
    perturbed_nrm = None
    if 'kd_ks' in material:
        # print("kd_ks:", material['kd_ks']) True
        # Combined texture, used for MLPs because lookups are expensive
        

        all_tex_jitter = material['kd_ks'].sample(gb_pos_original + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device="cuda"), idx)
        all_tex = material['kd_ks'].sample(gb_pos_original, idx)
        assert all_tex.shape[-1] == 6, "Combined kd_ks must be 6 channels"
        kd, ks = all_tex[..., 0:3], all_tex[..., 3:6]
        kd_grad  = torch.abs(all_tex_jitter[..., 0:3] - kd)
        ks_grad  = torch.abs(all_tex_jitter[..., 3:6] - ks) * torch.tensor([0, 1, 1], dtype=torch.float32, device='cuda')[None, None, None, :] # Omit o-component

    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1])
    kd = kd[..., 0:3]

    ################################################################################
    # Normal perturbation & normal bend
    ################################################################################
    if (not finetune_normal) or ('no_perturbed_nrm' in material and material['no_perturbed_nrm']):
        perturbed_nrm = None

    # Geometric smoothed normal regularizer
    nrm_jitter = dr.texture(gb_normal.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
    nrm_grad = torch.abs(nrm_jitter - gb_normal) * grad_weight


    if perturbed_nrm is not None:
        perturbed_nrm_jitter = dr.texture(perturbed_nrm.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
        perturbed_nrm_grad = 1.0 - util.safe_normalize(util.safe_normalize(perturbed_nrm_jitter) + util.safe_normalize(perturbed_nrm))[..., 2:3]
        perturbed_nrm_grad = perturbed_nrm_grad.repeat(1,1,1,3) * grad_weight

    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)


    ################################################################################
    # Evaluate BSDF
    ################################################################################
    assert 'bsdf' in material or bsdf is not None, "Material must specify a BSDF type"
    bsdf = material['bsdf'] if bsdf is None else bsdf
    
    bsdf = 'kd'
    if bsdf == 'pbr' or bsdf == 'diffuse' or bsdf == 'white':

        kd = torch.ones_like(kd) if bsdf == 'white' else kd

        assert isinstance(lgt, light.EnvironmentLight) and optix_ctx is not None
        ro = gb_pos + gb_normal*0.001

        global rnd_seed
        diffuse_accum, specular_accum = ou.optix_env_shade(optix_ctx, rast[..., -1], ro, gb_pos, gb_normal, view_pos, kd, ks, 
                            lgt.base, lgt._pdf, lgt.rows[:,0], lgt.cols, BSDF=bsdf, n_samples_x=FLAGS.n_samples, 
                            rnd_seed=None if FLAGS.decorrelated else rnd_seed, shadow_scale=shadow_scale)
        rnd_seed += 1

        if denoiser is not None and FLAGS.denoiser_demodulate:
            diffuse_accum  = denoiser.forward(torch.cat((diffuse_accum, gb_normal, gb_depth), dim=-1))
            specular_accum = denoiser.forward(torch.cat((specular_accum, gb_normal, gb_depth), dim=-1))

        if bsdf == 'white' or bsdf == 'diffuse':
            shaded_col = diffuse_accum * kd
        else:

            save_ks = util.rgb_to_srgb(ks)[0] # range[0,1]
            np_save_ks = save_ks.detach().cpu().numpy()
            util.save_image("chh_script/ks.png", np_save_ks)

            save_kd = util.rgb_to_srgb(kd)[0] # range[0,1]
            np_save_kd = save_kd.detach().cpu().numpy()
            util.save_image("chh_script/np_save_kd.png", np_save_kd)

            kd = kd * (1.0 - ks[..., 2:3]) # kd * (1.0 - metalness)
            shaded_col = diffuse_accum * kd + specular_accum

            save_kd = util.rgb_to_srgb(kd)[0] # range[0,1]
            np_save_kd = save_kd.detach().cpu().numpy()
            util.save_image("chh_script/np_save_kd2.png", np_save_kd)

            save_shaded_col = util.rgb_to_srgb(shaded_col)[0] # range[0,1]
            np_save_shaded_col = save_shaded_col.detach().cpu().numpy()
            util.save_image("chh_script/save_shaded_col.png", np_save_shaded_col)

        # denoise combined shaded values if possible
        if denoiser is not None and not FLAGS.denoiser_demodulate:
            shaded_col = denoiser.forward(torch.cat((shaded_col, gb_normal, gb_depth), dim=-1))

    elif bsdf == 'normal':
        shaded_col = (gb_normal + 1.0)*0.5
    elif bsdf == 'tangent':
        shaded_col = (gb_tangent + 1.0)*0.5
    elif bsdf == 'kd':
        shaded_col = kd


    elif bsdf == 'ks':
        shaded_col = ks
    else:
        assert False, "Invalid BSDF '%s'" % bsdf

    eps = 1e-8
    allone_map = torch.ones_like(alpha)
    # Return multiple buffers
    # Setting the `alphas` of depth and invdepth to 1 to avoid double blending
    # (one with background, the other in antialiasing)
    buffers = {
        'shaded'            : torch.cat((shaded_col, alpha), dim=-1),
        'z_grad'            : torch.cat((gb_depth, torch.zeros_like(alpha), alpha), dim=-1),
        'normal'            : torch.cat((gb_normal, alpha), dim=-1),
        'geometric_normal'  : torch.cat((gb_geometric_normal, alpha), dim=-1),
        'kd'                : torch.cat((kd, alpha), dim=-1),
        'ks'                : torch.cat((ks, alpha), dim=-1),
        'kd_grad'           : torch.cat((kd_grad, alpha), dim=-1),
        'ks_grad'           : torch.cat((ks_grad, alpha), dim=-1),
        'normal_grad'       : torch.cat((nrm_grad, alpha), dim=-1),
        'depth'             : torch.cat(((gb_pos - view_pos).pow(2).sum(dim=-1, keepdim=True).sqrt(), allone_map), dim=-1),
        'invdepth'          : torch.cat((1.0 / ((gb_pos - view_pos).pow(2) + eps).sum(dim=-1, keepdim=True).sqrt(), allone_map), dim=-1),
    }

    if 'diffuse_accum' in locals():
        buffers['diffuse_light'] = torch.cat((diffuse_accum, alpha), dim=-1)
    if 'specular_accum' in locals():
        buffers['specular_light'] = torch.cat((specular_accum, alpha), dim=-1)

    if perturbed_nrm is not None: 
        buffers['perturbed_nrm'] = torch.cat((perturbed_nrm, alpha), dim=-1)
        buffers['perturbed_nrm_grad'] = torch.cat((perturbed_nrm_grad, alpha), dim=-1)
    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        FLAGS,
        idx,
        v_pos_clip,
        rast,
        rast_deriv,
        mesh,
        mesh_original,
        view_pos,
        lgt,
        resolution,
        spp,
        msaa,
        optix_ctx,
        bsdf,
        denoiser,
        shadow_scale,
        use_uv=True,
        finetune_normal=True,
        extra_dict=None,
        xfm_lgt = None,
        shade_data = False
    ):

    # print("idx:", idx)
    full_res = [resolution[0]*spp, resolution[1]*spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution

    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())

    gb_pos_original, _ = interpolate(mesh_original.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())

    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)

    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())

    if use_uv:
        # Compute tangent space
        assert mesh.v_nrm is not None and mesh.v_tng is not None
        gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
        gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents

        # Texture coordinate
        assert mesh.v_tex is not None
        gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)

    else:
        # Compute tangent space

        assert mesh.v_nrm is not None
        gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
        with torch.no_grad():
            noise = torch.randn_like(gb_normal)
            noise = noise / noise.norm(dim=-1, keepdim=True)
        gb_tangent = torch.cross(noise, gb_normal) ### since we only use tangent for adding isotropic noises but not for uv maps

        # # Texture coordinate
        gb_texc, gb_texc_deriv = None, None

    # Interpolate z and z-gradient
    with torch.no_grad():
        eps = 0.00001
        clip_pos, clip_pos_deriv = interpolate(v_pos_clip, rast_out_s, mesh.t_pos_idx.int(), rast_db=rast_out_deriv_s)
        z0 = torch.clamp(clip_pos[..., 2:3], min=eps) / torch.clamp(clip_pos[..., 3:4], min=eps)
        z1 = torch.clamp(clip_pos[..., 2:3] + torch.abs(clip_pos_deriv[..., 2:3]), min=eps) / torch.clamp(clip_pos[..., 3:4] + torch.abs(clip_pos_deriv[..., 3:4]), min=eps)
        z_grad = torch.abs(z1 - z0)
        gb_depth = torch.cat((z0, z_grad), dim=-1)
    ################################################################################
    # Shade
    ################################################################################


    buffers = shade(
        FLAGS, idx, rast_out_s, gb_depth,
        gb_pos,gb_pos_original,
        gb_geometric_normal, gb_normal,
        gb_tangent, gb_texc, gb_texc_deriv,
        view_pos, lgt, mesh.material, optix_ctx, 
        mesh, bsdf,
        denoiser, shadow_scale,
        use_uv=use_uv,
        finetune_normal=finetune_normal,
        xfm_lgt=xfm_lgt,
        shade_data=shade_data
    )

    ################################################################################
    # Prepare output
    ################################################################################


    if extra_dict is not None:
        for key in extra_dict:
            if key == 'msdf' and extra_dict[key] is not None:
                assert extra_dict[key].dim() == 1 or (extra_dict[key].dim() == 2 and extra_dict[key].size(1) == 1)
                buffers['msdf_image'], _ = interpolate(extra_dict[key].squeeze()[None, :, None], rast_out_s, mesh.t_pos_idx.int())
            elif key == 'msdf_watertight' and extra_dict[key] is not None:
                assert extra_dict[key].dim() == 1 or (extra_dict[key].dim() == 2 and extra_dict[key].size(1) == 1)
                buffers['msdf_watertight_image'], _ = interpolate(extra_dict['msdf_watertight'].squeeze()[None, :, None], rast_out_s.detach(), mesh.t_pos_idx.int()) ## maybe better to stop all gradients to vpos

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')

    # Return buffers
    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        FLAGS,
        idx,
        ctx,
        mesh,
        mesh_original,
        mtx_in,
        view_pos,
        lgt,
        resolution,
        spp        = 1,
        num_layers = 1,
        msaa       = False,
        background = None,
        optix_ctx  = None,
        bsdf       = None,
        denoiser   = None,
        shadow_scale = 1.0,
        use_uv      = True,
        finetune_normal = True,
        extra_dict  = None,
        xfm_lgt     = None,
        shade_data  = False,
    ):

    def prepare_input_vector(x):
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
        return accum

    '''
        choose not to raise error since it is possible that we have msdf supervision. should clean the code later
    '''

    full_res = [resolution[0]*spp, resolution[1]*spp]

    # Convert numpy arrays to torch tensors

    view_pos    = prepare_input_vector(view_pos)
 
    # clip space transform

    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)

    # Render all layers front-to-back

    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:

        assert num_layers == 1
        rast, db = peeler.rasterize_next_layer()
        visible_triangles = rast[:,:,:,-1].long().unique()
        if visible_triangles[0] == 0:
            visible_triangles = visible_triangles[1:]
        visible_triangles = visible_triangles - 1

        layers = [
            (render_layer(
                FLAGS, idx, v_pos_clip,
                rast, db, mesh, mesh_original, view_pos, lgt, resolution, spp, msaa,
                optix_ctx, bsdf, denoiser, shadow_scale,
                use_uv=use_uv, finetune_normal=finetune_normal,
                extra_dict=extra_dict,
                xfm_lgt=xfm_lgt,
                shade_data=shade_data),
            rast)]

    # Setup background

    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    # Composite layers front-to-back
    out_buffers = {}
    out_buffers['visible_triangles'] = visible_triangles
    for key in layers[0][0].keys():

        if layers[0][0][key] is None:
            out_buffers[key] = None
            continue
        if key == 'shaded':
            accum = composite_buffer(key, layers, background, True)
        elif key == 'depth':
            # continue
            default_depth = 20.0
            accum = composite_buffer(key, layers, torch.ones_like(layers[0][0][key]) * default_depth, True)
        elif key == 'invdepth':
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), True)
        else:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), True)

        # Downscale to framebuffer resolution. Use avg pooling
        out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture):

    # clip space transform
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert all_tex.shape[-1] == 6, "Combined kd_ks must be 6 channels"
    return (rast[..., -1:] > 0).float(), all_tex[..., 0:3], all_tex[..., 3:6]