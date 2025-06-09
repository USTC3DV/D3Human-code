import numpy as np
import torch
import os.path as osp
import copy
from pytorch3d.io import load_obj
from .smplx_exavatar import SMPLX
from pytorch3d.ops import knn_points


def write_pc(path_mesh, v, vn=None, f=None):
    assert v.ndim == 2 and v.shape[1] == 3
    with open(path_mesh, 'w') as fp:
        fp.write(('v {:f} {:f} {:f}\n' * v.shape[0]).format(*v.reshape(-1)))
        if vn is not None:
            fp.write(('vn {:f} {:f} {:f}\n' * vn.shape[0]).format(*vn.reshape(-1)))
        if f is not None:
            fp.write(('f {:d} {:d} {:d}\n' * f.shape[0]).format(*f.reshape(-1) + 1))
    print("save mesh to:", path_mesh)


class SMPLX_Deformer(object):
    def __init__(self, model_path, gender):
        self.shape_param_dim = 100
        self.expr_param_dim = 50
        self.model_path = model_path
        self.layer_arg = {
            'create_global_orient': False,
            'create_body_pose': False,
            'create_left_hand_pose': False,
            'create_right_hand_pose': False,
            'create_jaw_pose': False,
            'create_leye_pose': False,
            'create_reye_pose': False,
            'create_betas': False,
            'create_expression': False,
            'create_transl': False
        }
        
        self.k = 1
        self.layer = SMPLX(
            model_path=model_path,
            model_type='smplx',
            gender=gender,
            num_betas=self.shape_param_dim,
            num_expression_coeffs=self.expr_param_dim,
            use_pca=False,
            use_face_contour=True,
            **self.layer_arg
        ).to('cuda')

        self.lbs_weights = self.layer.lbs_weights
        
        self.face_vertex_idx = np.load(
            osp.join(model_path, 'SMPL-X__FLAME_vertex_ids.npy')
        )
        

        self.vertex_num = 10475
        self.face = self.layer.faces.astype(np.int64)
        
        self.flip_corr = np.load(
            osp.join(model_path, 'smplx_flip_correspondences.npz')
        )
        
        self.vertex_uv, self.face_uv = self.load_uv_info()
        
        # 定义关节
        self.joint = {
            'num': 55,
            'name': (
                'Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2',
                'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck',
                'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
                'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 
                'Jaw', 'L_Eye', 'R_Eye',
                'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2',
                'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1',
                'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',
                'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2',
                'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1',
                'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3'
            )
        }
        
        self.joint['root_idx'] = self.joint['name'].index('Pelvis')
        self.joint['part_idx'] = {
            'body': range(
                self.joint['name'].index('Pelvis'),
                self.joint['name'].index('R_Wrist') + 1
            ),
            'face': range(
                self.joint['name'].index('Jaw'),
                self.joint['name'].index('R_Eye') + 1
            ),
            'lhand': range(
                self.joint['name'].index('L_Index_1'),
                self.joint['name'].index('L_Thumb_3') + 1
            ),
            'rhand': range(
                self.joint['name'].index('R_Index_1'),
                self.joint['name'].index('R_Thumb_3') + 1
            )
        }
        
        # 定义关键点
        self.kpt = {
            'num': 135,
            'name': (
                'Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle',
                'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
                'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe',
                'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',
                'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1',
                'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2',
                'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3',
                'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',
                'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1',
                'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2',
                'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3',
                'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',
                'Head', 'Jaw', *['Face_' + str(i) for i in range(1, 69)]
            ),
            'idx': (
                0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55,
                37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70,
                52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75,
                15,22,
                76,77,78,79,80,81,82,83,84,85,
                86,87,88,89,
                90,91,92,93,94,
                95,96,97,98,99,100,101,102,103,104,105,106,
                107,
                108,109,110,111,112,
                113,
                114,115,116,117,118,
                119,
                120,121,122,
                123,
                124,125,126,
                127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143
            )
        }
        
        self.kpt['root_idx'] = self.kpt['name'].index('Pelvis')
        self.kpt['part_idx'] = {
            'body': range(
                self.kpt['name'].index('Pelvis'),
                self.kpt['name'].index('Nose') + 1
            ),
            'lhand': range(
                self.kpt['name'].index('L_Thumb_1'),
                self.kpt['name'].index('L_Pinky_4') + 1
            ),
            'rhand': range(
                self.kpt['name'].index('R_Thumb_1'),
                self.kpt['name'].index('R_Pinky_4') + 1
            ),
            'face': [
                self.kpt['name'].index('Neck'),
                self.kpt['name'].index('Head'),
                self.kpt['name'].index('Jaw'),
                self.kpt['name'].index('L_Eye'),
                self.kpt['name'].index('R_Eye')
            ] + list(range(
                self.kpt['name'].index('Face_1'),
                self.kpt['name'].index('Face_68') + 1
            )) + [
                self.kpt['name'].index('L_Ear'),
                self.kpt['name'].index('R_Ear')
            ]
        }

    def initialize(self, betas, pose=None, save_path=None):

        batch_size = betas.shape[0]

        global_orient = torch.zeros(batch_size, 3)
        body_pose_t = torch.zeros(batch_size, 63)
        body_pose_t[:, 2] = torch.pi / 36
        body_pose_t[:, 5] = -torch.pi / 36

        jaw_pose = torch.zeros(batch_size, 3)
        leye_pose = torch.zeros(batch_size, 3)
        reye_pose = torch.zeros(batch_size, 3)
        left_hand_pose = torch.zeros(batch_size, 45) 
        right_hand_pose = torch.zeros(batch_size, 45) 
        expression = torch.zeros(batch_size, self.expr_param_dim)
        transl = torch.zeros(batch_size, 3) 

        if pose is not None:
            if 'global_orient' in pose:
                global_orient = pose['global_orient']
            if 'body_pose' in pose:
                body_pose_t = pose['body_pose']
            if 'jaw_pose' in pose:
                jaw_pose = pose['jaw_pose']
            if 'leye_pose' in pose:
                leye_pose = pose['leye_pose']
            if 'reye_pose' in pose:
                reye_pose = pose['reye_pose']
            if 'left_hand_pose' in pose:
                left_hand_pose = pose['left_hand_pose']
            if 'right_hand_pose' in pose:
                right_hand_pose = pose['right_hand_pose']
            if 'expression' in pose:
                expression = pose['expression']
        
        betas = betas.cuda()
        global_orient = global_orient.cuda()
        body_pose_t = body_pose_t.cuda()
        jaw_pose = jaw_pose.cuda()
        leye_pose = leye_pose.cuda()
        reye_pose = reye_pose.cuda()
        left_hand_pose = left_hand_pose.cuda()
        right_hand_pose = right_hand_pose.cuda()
        expression = expression.cuda()
        transl = transl.cuda()

        output, A = self.layer(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose_t,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=expression,
            transl=transl
        )

        self.vs_template = output.vertices
        self.f = self.layer.faces_tensor
        self.vertices = self.vs_template.float()
        self.init_A = A

        if save_path is not None:
            write_pc(save_path, self.vs_template[0], f=self.f)

    def get_expr_from_flame(self, smplx_layer):
        flame_layer = smplx.create(
            model_path=self.model_path,
            model_type='flame',
            gender='neutral',
            num_betas=self.shape_param_dim,
            num_expression_coeffs=self.expr_param_dim
        ).to('cuda')
        
        smplx_layer.expression[:, self.face_vertex_idx, :] = flame_layer.expression
        return smplx_layer

    def get_face_offset(self, face_offset):
        """
        将面部偏移量应用到模型的面部顶点上
        """
        batch_size = face_offset.shape[0]
        face_offset_pad = torch.zeros((batch_size, self.vertex_num, 3)).float().cuda()
        face_offset_pad[:, self.face_vertex_idx, :] = face_offset
        return face_offset_pad

    def get_joint_offset(self, joint_offset):
        """
        对关节偏移量进行处理，排除根关节和髋关节
        """
        weight = torch.ones((1, self.joint['num'], 1)).float().cuda()
        weight[:, self.joint['root_idx'], :] = 0
        weight[:, self.joint['name'].index('R_Hip'), :] = 0
        weight[:, self.joint['name'].index('L_Hip'), :] = 0
        joint_offset = joint_offset * weight
        return joint_offset

    def get_locator_offset(self, locator_offset):
        """
        仅对髋关节应用偏移量
        """
        weight = torch.zeros((1, self.joint['num'], 1)).float().cuda()
        weight[:, self.joint['name'].index('R_Hip'), :] = 1
        weight[:, self.joint['name'].index('L_Hip'), :] = 1
        locator_offset = locator_offset * weight
        return locator_offset

    def load_uv_info(self):
        """
        加载 UV 信息
        """
        verts, faces, aux = load_obj(
            osp.join(self.model_path,  'smplx_uv', 'smplx_uv.obj')
        )
        vertex_uv = aux.verts_uvs.numpy().astype(np.float32)  
        face_uv = faces.textures_idx.numpy().astype(np.int64)  
        return vertex_uv, face_uv

    def get_smplx_coord(
        self,
        smplx_param,
        cam_param=None,
        use_pose=True,
        use_expr=True,
        use_face_offset=True,
        use_joint_offset=True,
        use_locator_offset=True,
        root_rel=False
    ):

        self.smplx_layer = copy.deepcopy(self.layer).cuda()
        batch_size = smplx_param['root_pose'].shape[0]
        
        if use_pose:
            root_pose = smplx_param['root_pose']
            body_pose = smplx_param['body_pose']
            jaw_pose = smplx_param['jaw_pose']
            leye_pose = smplx_param['leye_pose']
            reye_pose = smplx_param['reye_pose']
            lhand_pose = smplx_param['lhand_pose']
            rhand_pose = smplx_param['rhand_pose']
        else:
            root_pose = torch.zeros_like(smplx_param['root_pose'])
            body_pose = torch.zeros_like(smplx_param['body_pose'])
            jaw_pose = torch.zeros_like(smplx_param['jaw_pose'])
            leye_pose = torch.zeros_like(smplx_param['leye_pose'])
            reye_pose = torch.zeros_like(smplx_param['reye_pose'])
            lhand_pose = torch.zeros_like(smplx_param['lhand_pose'])
            rhand_pose = torch.zeros_like(smplx_param['rhand_pose'])
        
        if use_expr:
            expr = smplx_param['expr']
        else:
            expr = torch.zeros_like(smplx_param['expr'])
        
        if use_face_offset:
            face_offset = self.get_face_offset(smplx_param['face_offset'])
        else:
            face_offset = None
        
        if use_joint_offset:
            joint_offset = self.get_joint_offset(smplx_param['joint_offset'])
        else:
            joint_offset = None
        
        if use_locator_offset:
            locator_offset = self.get_locator_offset(smplx_param['locator_offset'])
        else:
            locator_offset = None
        
        output = self.smplx_layer(
            global_orient=root_pose,
            body_pose=body_pose,
            jaw_pose=jaw_pose.detach(),
            leye_pose=leye_pose.detach(),
            reye_pose=reye_pose.detach(),
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            expression=expr.detach(),
            betas=smplx_param['shape'],
            face_offset=face_offset,
            joint_offset=joint_offset,
            locator_offset=locator_offset
        )

        return output
        

    def interpolate_weights(self, pts):

        B, P, _ = pts.shape
        dist_sq, idx, _ = knn_points(pts, self.vs_template.expand(B, -1, -1), K=self.k) # idx: [B,P,K]
        dist = torch.sqrt(dist_sq + 1e-9)
        w_distance = 1.0 / (dist + 1e-9)  # [B,P,K]
        w_distance = w_distance / w_distance.sum(dim=2, keepdim=True) # [B,P,K]
        w_distance = w_distance.unsqueeze(-1) # [B,P,K,1]

        lbs_w_expanded = self.lbs_weights.unsqueeze(0).expand(B, -1, -1) # [B,V,J]
        J = lbs_w_expanded.shape[2]

        PK = P * self.k
        idx_flat = idx.reshape(B, PK)  # [B,P*K]
        idx_flat = idx_flat.unsqueeze(-1).expand(B, PK, J) # [B,P*K,J]
        w_neighbors = torch.gather(lbs_w_expanded, 1, idx_flat) # [B,P*K,J]
        w_neighbors = w_neighbors.reshape(B, P, self.k, J) # [B,P,K,J]

        w_pts = (w_neighbors * w_distance).sum(dim=2) # [B,P,J]

        return w_pts

    def apply_lbs_inverse(self, pts, init_A, w_pts, Inverse=True):
        """
        将pts从init_A对应的姿态逆回到canonical空间（近似方法）。
        pts: [B,P,3], init_A: [B,J,4,4], w_pts: [B,P,J]
        
        原理：
        p_init = Σ_j w_{p,j} init_A_j p_canonical
        希望 p_canonical = (Σ_j w_{p,j} init_A_j)^{-1} p_init
        
        对于每个点p：
        M_p = Σ_j w_{p,j} init_A_j
        p_canonical = M_p^{-1} p_init
        需要注意的是 M_p 并不一定是刚体变换，可能需要正交化。
        """
        device = pts.device
        B, P, _ = pts.shape
        ones = torch.ones(B, P, 1, device=device)
        pts_h = torch.cat([pts, ones], dim=2) # [B,P,4]


        w_pts_mat = w_pts.unsqueeze(-1).unsqueeze(-1) # [B,P,J,1,1]
        A_expanded = init_A.unsqueeze(2) # [B,J,1,4,4]
        A_expanded = A_expanded.expand(B, -1, P, -1, -1) # [B,J,P,4,4]
        A_expanded = A_expanded.permute(0,2,1,3,4) # [B,P,J,4,4]

        M_p = (A_expanded * w_pts_mat).sum(dim=2) # [B,P,4,4]

        if Inverse==True:

            M_p_inv = torch.inverse(M_p) # [B,P,4,4]
        else:
            M_p_inv = M_p

        pts_h = pts_h.unsqueeze(-1) # [B,P,4,1]
        pts_canonical_h = torch.matmul(M_p_inv, pts_h) # [B,P,4,1]
        pts_canonical = pts_canonical_h[..., :3, 0]  
        return pts_canonical


    def lbs_forward_inverse(self, pts):
        pts = pts.cuda()
        w_pts = self.interpolate_weights(pts) # [B,P,J] 

        pts_canonical = self.apply_lbs_inverse(pts, self.init_A, w_pts)

        return pts_canonical



    def lbs_forward(self, pts, smplx_param, idx, face=None):

        B, P, _ = pts.shape
        pts = pts.cuda()

        shape = smplx_param['shape']
        face_offset = smplx_param['face_offset']
        joint_offset = smplx_param['joint_offset']
        locator_offset = smplx_param['locator_offset']

        trans = smplx_param['trans'][idx].reshape(1, 3)
        rhand_pose = smplx_param['rhand_pose'][idx].reshape(1, 45)
        jaw_pose = smplx_param['jaw_pose'][idx].reshape(1, 3)
        expr = smplx_param['expr'][idx].reshape(1, self.expr_param_dim)
        body_pose = smplx_param['body_pose'][idx].reshape(1, 63)
        root_pose = smplx_param['root_pose'][idx].reshape(1, 3)
        lhand_pose = smplx_param['lhand_pose'][idx].reshape(1, 45)
        leye_pose = smplx_param['leye_pose'][idx].reshape(1, 3)
        reye_pose = smplx_param['reye_pose'][idx].reshape(1, 3)

        w_pts = self.interpolate_weights(pts) # [B,P,J] 

        output, A = self.layer.forward(
            betas=shape,
            global_orient=root_pose,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            expression=expr,
            transl=trans,
            face_offset=face_offset,
            joint_offset=joint_offset,
            locator_offset=locator_offset,
            pose2rot=True, 
        )


        pts_canonical = self.apply_lbs_inverse(pts, self.init_A, w_pts)

        pts_new = self.apply_lbs_inverse(pts_canonical, A, w_pts, Inverse=False) + trans
        # print("pts_new:", pts_new.shape)
        # v = output["vertices"]
        # write_pc("./chh_script/pts.obj", v=pts.reshape(-1, 3), f=face)
        # write_pc("./chh_script/pts_canonical.obj", v=pts_canonical.reshape(-1, 3), f=face)
        # write_pc("./chh_script/pts_new.obj", v=pts_new.reshape(-1, 3), f=face)
        # exit()

        # print("trans:", trans, trans.shape)

        return pts_new.reshape(-1, 3)
