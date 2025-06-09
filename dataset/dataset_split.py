# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import json
import argparse
import imageio
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from render import util

from .dataset import Dataset_people_smplx

import cv2 as cv

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def _load_img(path):
    img = util.load_image_raw(path)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }

def get_ndc_matrix_from_ss(height, width, fx, fy, cx, cy, n=0.001, f=1000.):

    # ndc_proj_mat = cam_paras.new_zeros((batch_size, 4, 4))
    ndc_proj_mat = torch.zeros((4, 4))
    ndc_proj_mat[0, 0] = 2*fx/(width-1)
    ndc_proj_mat[0, 2] = 1-2*cx/(width-1)
    ndc_proj_mat[1, 1] = -2*fy/(height-1)
    ndc_proj_mat[1, 2] = 1-2*cy/(height-1)
    ndc_proj_mat[2, 2] = -(f+n)/(f-n)
    ndc_proj_mat[2, 3] = -(2*f*n)/(f-n)
    ndc_proj_mat[3, 2] = -1.
    return ndc_proj_mat        # )


def read_json_files(file_path):
    json_data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            json_data.append(data)
        except json.JSONDecodeError as e:
            print(f"读取文件 {file_path} 时出错: {e}")
    return json_data

def load_smplx_param(root):
    npz_path = os.path.join(root, "merged_smplx.npz")
    smplx_params = dict(np.load(str(npz_path)))

    face_offset_json = read_json_files(os.path.join(root, "smplx_optimized/face_offset.json"))
    joint_offset_json = read_json_files(os.path.join(root, "smplx_optimized/joint_offset.json"))
    locator_offset_json = read_json_files(os.path.join(root, "smplx_optimized/locator_offset.json"))
    shape_param = read_json_files(os.path.join(root, "smplx_optimized/shape_param.json"))

    smplx_dict = {
        "trans": torch.from_numpy(smplx_params["trans"].astype(np.float32)).reshape(-1, 3).cuda(),
        "rhand_pose": torch.from_numpy(smplx_params["rhand_pose"].astype(np.float32)).reshape(-1, 45).cuda(),
        "jaw_pose": torch.from_numpy(smplx_params["jaw_pose"].astype(np.float32)).reshape(-1, 3).cuda(),
        "reye_pose": torch.from_numpy(smplx_params["reye_pose"].astype(np.float32)).reshape(-1, 3).cuda(),
        "expr": torch.from_numpy(smplx_params["expr"].astype(np.float32)).reshape(-1, 50).cuda(),
        "body_pose": torch.from_numpy(smplx_params["body_pose"].astype(np.float32)).reshape(-1, 63).cuda(),
        "root_pose": torch.from_numpy(smplx_params["root_pose"].astype(np.float32)).reshape(-1, 3).cuda(),
        "lhand_pose": torch.from_numpy(smplx_params["lhand_pose"].astype(np.float32)).reshape(-1, 45).cuda(),
        "leye_pose": torch.from_numpy(smplx_params["leye_pose"].astype(np.float32)).reshape(-1, 3).cuda(),
        "face_offset": torch.from_numpy(np.array(face_offset_json).astype(np.float32)).cuda(),
        "joint_offset": torch.from_numpy(np.array(joint_offset_json).astype(np.float32)).cuda(),
        "locator_offset": torch.from_numpy(np.array(locator_offset_json).astype(np.float32)).cuda(),
        "shape_param": torch.from_numpy(np.array(shape_param).astype(np.float32)).cuda(),
    }

    return smplx_dict

class Dataset_split(Dataset_people_smplx):
    def __init__(self, base_dir, FLAGS, Detail=False, process_path=None,examples=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = base_dir
        
        self.img_root = os.path.join(self.base_dir, "images")
        self.msk_root = os.path.join(self.base_dir, "all")

        key_list_path = os.path.join(self.base_dir, "key.list")
        key_list = []
        self.key_frame = []
        with open(key_list_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                key_list.append(int(stripped_line))
        self.begin = key_list[0]
        self.end = key_list[1]
        self.key_frame = list(range(self.begin, self.end + 1))

        self.img_lists = sorted(glob.glob(f"{self.base_dir}/images/*.png"))
        self.normal_lists = sorted(glob.glob(f"{self.base_dir}/normal/*.png"))

        self.msk_lists = sorted(glob.glob(f"{self.base_dir}/all/*.png"))
        self.cloth_msk_lists = sorted(glob.glob(f"{self.base_dir}/all_cloth_mask/*.png"))
        self.body_msk_lists = sorted(glob.glob(f"{self.base_dir}/all_body_mask/*.png"))

        self.n_images = self.end - self.begin

        smplx_npz_root  = os.path.join(self.base_dir, "smplx")
        self.smplx_params = load_smplx_param(smplx_npz_root)

        self.shape_param =  self.smplx_params["shape_param"] # (1, 100)
        self.face_offset =  self.smplx_params["face_offset"] # (1, 10475, 3)
        self.joint_offset =  self.smplx_params["joint_offset"] # (1, 55, 3)
        self.locator_offset = self.smplx_params["locator_offset"] # (1, 55, 3)
   
        
        if Detail==True:
            cloth_body_path = os.path.join(process_path, "merge_body_cloth.npz")
            cloth_body_npz = np.load(cloth_body_path)
            self.v = torch.from_numpy(cloth_body_npz["v"]).cuda().float()
            self.f = torch.from_numpy(cloth_body_npz["f"]).cuda().long()
            self.face_labels = torch.from_numpy(cloth_body_npz["face_labels"]).cuda()

            self.body_inside_root = os.path.join(process_path, "inside_body_index.npz")
            body_inside_npz = np.load(self.body_inside_root)
            self.inside_body_index = torch.from_numpy(body_inside_npz["inside_body_index"]).long()
            self.outside_body_index = torch.from_numpy(body_inside_npz["outside_body_index"]).long()

            f_cloth = self.f[self.face_labels==1]
            self.cloth_index = torch.unique(f_cloth)
            self.outside_index = torch.cat((self.cloth_index.cuda(), self.outside_body_index.cuda()))


        cameras_npz_path = os.path.join(self.base_dir, "smplx/cameras.npz")
        camera = np.load(cameras_npz_path)

        K = torch.from_numpy(camera["intrinsic"]) 
        w2c = torch.from_numpy(camera["extrinsic"]).float()

        height = camera["height"] // 2
        width = camera["width"]  // 2

        fx = K[0, 0] // 2
        fy = K[1, 1] // 2
        cx = K[0, 2] // 2
        cy = K[1, 2] // 2

        self.proj_mtx = get_ndc_matrix_from_ss(height, width, fx, fy, cx, cy)


        flip_mat = torch.tensor([
            [ 1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0,  0,  1]
        ], dtype=torch.float)

        self.w2c = w2c
        self.flip_mat = flip_mat

 
        self.mv = flip_mat @ w2c
        self.campos = torch.linalg.inv(self.mv)[:3, 3]
        self.mvp = self.proj_mtx @ self.mv


    def load_img(self, img):
        img1 = torch.from_numpy(img.astype(np.float32)/255)
        img2 = srgb_to_rgb(img1)
        return img2

    def __len__(self):
        return self.n_images if self.examples is None else self.examples
    

    def __getitem__(self, itr):

        idx = self.key_frame[itr % self.n_images]
        
        iter_res = self.FLAGS.train_res
        rgb = imageio.imread(self.img_lists[idx])

        rgb = cv2.resize(rgb, (iter_res[0], iter_res[1]))
        rgb = self.load_img(rgb)


        msk = imageio.imread(self.msk_lists[idx]) 
        msk[msk > 0] = 1
        msk = cv2.resize(msk, (iter_res[0], iter_res[1]))
        msk_np = np.expand_dims(msk, axis=2).astype(float)  # Convert to HxWx1 for concatenation

        cloth_msk = imageio.imread(self.cloth_msk_lists[idx])
        cloth_msk[cloth_msk > 0] = 1
        cloth_msk = cv2.resize(cloth_msk, (iter_res[0], iter_res[1]))
        cloth_msk_np = np.expand_dims(cloth_msk, axis=2).astype(float)  # Convert to HxWx1 for concatenation

        body_msk = imageio.imread(self.body_msk_lists[idx])
        body_msk[body_msk > 0] = 1
        body_msk = cv2.resize(body_msk, (iter_res[0], iter_res[1]))
        body_msk_np = np.expand_dims(body_msk, axis=2).astype(float)  # Convert to HxWx1 for concatenation

        img = torch.from_numpy(np.concatenate((rgb, msk_np), axis=2))
        img[:,:,:3] = img[:,:,:3] * img[:,:,3:]
        img[:,:,3] = torch.sign(img[:,:,3])

        cloth_img = torch.from_numpy(np.concatenate((rgb, cloth_msk_np), axis=2))
        cloth_img[:,:,:3] = cloth_img[:,:,:3] * cloth_img[:,:,3:]
        cloth_img[:,:,3] = torch.sign(cloth_img[:,:,3])

        body_img = torch.from_numpy(np.concatenate((rgb, body_msk_np), axis=2))
        body_img[:,:,:3] = body_img[:,:,:3] * body_img[:,:,3:]
        body_img[:,:,3] = torch.sign(body_img[:,:,3])

        normal_load = cv2.imread(self.normal_lists[idx], cv2.IMREAD_COLOR)
        normal_load = cv2.cvtColor(normal_load, cv2.COLOR_BGR2RGB)
        normal = torch.from_numpy(np.array(normal_load)).float()
        normal = normal/255.
        normal = normal * 2.0 - 1.0
        
        normal = normal * msk_np
        body_normal = normal * body_msk_np
        cloth_normal = normal * cloth_msk_np


        return {
            'idx': idx,
            'mv': self.mv[None, ...].cuda(), 
            'mvp': self.mvp[None, ...].cuda(),
            'campos': self.campos[None, ...].cuda(), # camera pose

            'resolution' : iter_res,
            'spp' : self.FLAGS.spp,

            'all_img': img[None, ...].cuda(),
            'cloth_img': cloth_img[None, ...].cuda(), 
            'body_img': body_img[None, ...].cuda(),
            'all_normal': normal[None, ...].cuda().float(),
            'body_normal': body_normal[None, ...].cuda().float(),
            'cloth_normal': cloth_normal[None, ...].cuda().float(),

            'all_msk': torch.from_numpy(msk_np)[None, ...].cuda().float(),#.to(torch.bool),
            'cloth_msk': torch.from_numpy(cloth_msk_np)[None, ...].cuda().float(),#.to(torch.bool),
            'body_msk': torch.from_numpy(body_msk_np)[None, ...].cuda().float(),#.to(torch.bool),

            'trans': self.smplx_params["trans"][idx][None, ...].cuda(),
            'rhand_pose': self.smplx_params["rhand_pose"][idx][None, ...].cuda(),
            'jaw_pose': self.smplx_params["jaw_pose"][idx][None, ...].cuda(),
            'expr': self.smplx_params["expr"][idx][None, ...].cuda(),
            'body_pose': self.smplx_params["body_pose"][idx][None, ...].cuda(),
            'root_pose': self.smplx_params["root_pose"][idx][None, ...].cuda(),
            'lhand_pose': self.smplx_params["lhand_pose"][idx][None, ...].cuda(),
            'leye_pose': self.smplx_params["leye_pose"][idx][None, ...].cuda(),
        }



