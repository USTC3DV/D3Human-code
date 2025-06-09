# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

class Dataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        return {
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0) if 'img' in batch[0] else None,
            'img_second' : torch.cat(list([item['img_second'] for item in batch]), dim=0) if 'img_second' in batch[0] else None,
            'invdepth' : torch.cat(list([item['invdepth'] for item in batch]), dim=0)if 'invdepth' in batch[0] else None,
            'invdepth_second' : torch.cat(list([item['invdepth_second'] for item in batch]), dim=0) if 'invdepth_second' in batch[0] else None,
            'envlight_transform': torch.cat(list([item['envlight_transform'] for item in batch]), dim=0) if 'envlight_transform' in batch and batch[0]['envlight_transform'] is not None else None,
        }
    
class Dataset_people(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        return {
            'idx': [item['idx'] for item in batch],
            # 'idx': batch[0]['idx'],

            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'all_img' : torch.cat(list([item['all_img'] for item in batch]), dim=0) if 'all_img' in batch[0] else None,
            'img_second' : torch.cat(list([item['img_second'] for item in batch]), dim=0) if 'img_second' in batch[0] else None,
            'cloth_img' : torch.cat(list([item['cloth_img'] for item in batch]), dim=0) if 'cloth_img' in batch[0] else None,  
            'body_img' : torch.cat(list([item['body_img'] for item in batch]), dim=0) if 'body_img' in batch[0] else None,  

            'all_normal' : torch.cat(list([item['all_normal'] for item in batch]), dim=0) if 'all_normal' in batch[0] else None,  
            'body_normal' : torch.cat(list([item['body_normal'] for item in batch]), dim=0) if 'body_normal' in batch[0] else None, 
            'cloth_normal' : torch.cat(list([item['cloth_normal'] for item in batch]), dim=0) if 'cloth_normal' in batch[0] else None,  


            'all_msk' : torch.cat(list([item['all_msk'] for item in batch]), dim=0) if 'all_msk' in batch[0] else None,  
            'body_msk' : torch.cat(list([item['body_msk'] for item in batch]), dim=0) if 'body_msk' in batch[0] else None, 
            'cloth_msk' : torch.cat(list([item['cloth_msk'] for item in batch]), dim=0) if 'cloth_msk' in batch[0] else None,  

            # 'invdepth' : torch.cat(list([item['invdepth'] for item in batch]), dim=0)if 'invdepth' in batch[0] else None,
            # 'invdepth_second' : torch.cat(list([item['invdepth_second'] for item in batch]), dim=0) if 'invdepth_second' in batch[0] else None,
            # 'envlight_transform': torch.cat(list([item['envlight_transform'] for item in batch]), dim=0) if 'envlight_transform' in batch and batch[0]['envlight_transform'] is not None else None,
        
            'global_orient': torch.cat(list([item['global_orient'] for item in batch]), dim=0) if 'global_orient' in batch[0] else None,
            'body_pose': torch.cat(list([item['body_pose'] for item in batch]), dim=0) if 'body_pose' in batch[0] else None,
            'transl': torch.cat(list([item['transl'] for item in batch]), dim=0) if 'transl' in batch[0] else None,
        
            # 'body_v': [item['body_v'] for item in batch] if 'body_v' in batch[0] else None,
            # 'body_f': [item['body_f'] for item in batch] if 'body_f' in batch[0] else None,
            # 'body_n': [item['body_n'] for item in batch] if 'body_n' in batch[0] else None,
            # 'cloth_v': [item['cloth_v'] for item in batch] if 'cloth_v' in batch[0] else None,
            # 'cloth_f': [item['cloth_f'] for item in batch] if 'cloth_f' in batch[0] else None,
            # 'cloth_n': [item['cloth_n'] for item in batch] if 'cloth_n' in batch[0] else None,

            # 'v': batch['v'],
            # 'f': batch['f'],
            # 'face_labels': batch["face_labels"],
        }
    

class Dataset_people_stage1(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        return {
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0) if 'img' in batch[0] else None,
            'img_second' : torch.cat(list([item['img_second'] for item in batch]), dim=0) if 'img_second' in batch[0] else None,
            'cloth_img' : torch.cat(list([item['cloth_img'] for item in batch]), dim=0) if 'cloth_img' in batch[0] else None,  
            # 'body_img' : torch.cat(list([item['body_img'] for item in batch]), dim=0) if 'body_img' in batch[0] else None,  

            'normal' : torch.cat(list([item['normal'] for item in batch]), dim=0) if 'normal' in batch[0] else None,  
            # 'body_normal' : torch.cat(list([item['body_normal'] for item in batch]), dim=0) if 'body_normal' in batch[0] else None, 
            # 'cloth_normal' : torch.cat(list([item['cloth_normal'] for item in batch]), dim=0) if 'cloth_normal' in batch[0] else None,  

            # 'invdepth' : torch.cat(list([item['invdepth'] for item in batch]), dim=0)if 'invdepth' in batch[0] else None,
            # 'invdepth_second' : torch.cat(list([item['invdepth_second'] for item in batch]), dim=0) if 'invdepth_second' in batch[0] else None,
            # 'envlight_transform': torch.cat(list([item['envlight_transform'] for item in batch]), dim=0) if 'envlight_transform' in batch and batch[0]['envlight_transform'] is not None else None,
        
            'global_orient': torch.cat(list([item['global_orient'] for item in batch]), dim=0) if 'global_orient' in batch[0] else None,
            'body_pose': torch.cat(list([item['body_pose'] for item in batch]), dim=0) if 'body_pose' in batch[0] else None,
            'transl': torch.cat(list([item['transl'] for item in batch]), dim=0) if 'transl' in batch[0] else None,
            # 'idx': torch.cat(list([item['idx'] for item in batch]), dim=0) if 'idx' in batch[0] else None,
            # 'idx': batch[0]['idx'],
            'idx': [item['idx'] for item in batch],
            # 'body_v': [item['body_v'] for item in batch] if 'body_v' in batch[0] else None,
            # 'body_f': [item['body_f'] for item in batch] if 'body_f' in batch[0] else None,
            # 'body_n': [item['body_n'] for item in batch] if 'body_n' in batch[0] else None,
            # 'cloth_v': [item['cloth_v'] for item in batch] if 'cloth_v' in batch[0] else None,
            # 'cloth_f': [item['cloth_f'] for item in batch] if 'cloth_f' in batch[0] else None,
            # 'cloth_n': [item['cloth_n'] for item in batch] if 'cloth_n' in batch[0] else None,

            # 'v': batch['v'],
            # 'f': batch['f'],
            # 'face_labels': batch["face_labels"],
        }
    

class Dataset_people_smplx(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        return {
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0) if 'img' in batch[0] else None,
            'img_second' : torch.cat(list([item['img_second'] for item in batch]), dim=0) if 'img_second' in batch[0] else None,
            'cloth_img' : torch.cat(list([item['cloth_img'] for item in batch]), dim=0) if 'cloth_img' in batch[0] else None,  
            'body_img' : torch.cat(list([item['body_img'] for item in batch]), dim=0) if 'body_img' in batch[0] else None,  
            'all_img' : torch.cat(list([item['all_img'] for item in batch]), dim=0) if 'all_img' in batch[0] else None,

            'normal' : torch.cat(list([item['normal'] for item in batch]), dim=0) if 'normal' in batch[0] else None,  
            'all_normal' : torch.cat(list([item['all_normal'] for item in batch]), dim=0) if 'all_normal' in batch[0] else None,  
            'body_normal' : torch.cat(list([item['body_normal'] for item in batch]), dim=0) if 'body_normal' in batch[0] else None, 
            'cloth_normal' : torch.cat(list([item['cloth_normal'] for item in batch]), dim=0) if 'cloth_normal' in batch[0] else None,  

            # 'invdepth' : torch.cat(list([item['invdepth'] for item in batch]), dim=0)if 'invdepth' in batch[0] else None,
            # 'invdepth_second' : torch.cat(list([item['invdepth_second'] for item in batch]), dim=0) if 'invdepth_second' in batch[0] else None,
            # 'envlight_transform': torch.cat(list([item['envlight_transform'] for item in batch]), dim=0) if 'envlight_transform' in batch and batch[0]['envlight_transform'] is not None else None,
    
            'idx': [item['idx'] for item in batch],
    
            'trans': torch.cat(list([item['trans'] for item in batch]), dim=0) if 'trans' in batch[0] else None,
            'rhand_pose': torch.cat(list([item['rhand_pose'] for item in batch]), dim=0) if 'rhand_pose' in batch[0] else None,
            'jaw_pose': torch.cat(list([item['jaw_pose'] for item in batch]), dim=0) if 'jaw_pose' in batch[0] else None,
            'expr': torch.cat(list([item['expr'] for item in batch]), dim=0) if 'expr' in batch[0] else None,
            'body_pose': torch.cat(list([item['body_pose'] for item in batch]), dim=0) if 'body_pose' in batch[0] else None,
            'root_pose': torch.cat(list([item['root_pose'] for item in batch]), dim=0) if 'root_pose' in batch[0] else None,     
            'lhand_pose': torch.cat(list([item['lhand_pose'] for item in batch]), dim=0) if 'lhand_pose' in batch[0] else None,
            'leye_pose': torch.cat(list([item['leye_pose'] for item in batch]), dim=0) if 'leye_pose' in batch[0] else None,
        }