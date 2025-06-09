# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch

from . import obj
from . import util


def normal_consistency_loss(mesh):
    """ Compute the normal consistency term as the cosine similarity between neighboring face normals.

    Args:
        mesh (Mesh): Mesh with face normals.
    """

    # print("mesh.connected_faces[:, 0]:", mesh.connected_faces)
    # exit()

    loss = 1 - torch.cosine_similarity(mesh.face_normals[mesh.connected_faces[:, 0]], mesh.face_normals[mesh.connected_faces[:, 1]], dim=1)

    return (loss**2).mean()

def compute_laplacian_uniform(mesh):
    """
    Computes the laplacian in packed form.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph
    Returns:
        Sparse FloatTensor of shape (V, V) where V = sum(V_n)
    """

    # This code is adapted from from PyTorch3D 
    # (https://github.com/facebookresearch/pytorch3d/blob/88f5d790886b26efb9f370fb9e1ea2fa17079d19/pytorch3d/structures/meshes.py#L1128)

    
    verts_packed = mesh.v_pos # (sum(V_n), 3)
    edges_packed = mesh.edges    # (sum(E_n), 2)
    V = mesh.v_pos.shape[0]

    e0, e1 = edges_packed.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=mesh.v_pos.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=mesh.v_pos.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=mesh.v_pos.device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L


def find_edges(indices, remove_duplicates=True):
    # Extract the three edges (in terms of vertex indices) for each face 
    # edges_0 = [f0_e0, ..., fN_e0]
    # edges_1 = [f0_e1, ..., fN_e1]
    # edges_2 = [f0_e2, ..., fN_e2]
    edges_0 = torch.index_select(indices, 1, torch.tensor([0,1], device=indices.device))
    edges_1 = torch.index_select(indices, 1, torch.tensor([1,2], device=indices.device))
    edges_2 = torch.index_select(indices, 1, torch.tensor([2,0], device=indices.device))

    # Merge the into one tensor so that the three edges of one face appear sequentially
    # edges = [f0_e0, f0_e1, f0_e2, ..., fN_e0, fN_e1, fN_e2]
    edges = torch.cat([edges_0, edges_1, edges_2], dim=1).view(indices.shape[0] * 3, -1)

    if remove_duplicates:
        edges, _ = torch.sort(edges, dim=1)
        edges = torch.unique(edges, dim=0)

    return edges

# 
def find_connected_faces(indices):
    edges = find_edges(indices, remove_duplicates=False)

    # Make sure that two edges that share the same vertices have the vertex ids appear in the same order
    edges, _ = torch.sort(edges, dim=1)

    # Now find edges that share the same vertices and make sure there are only manifold edges
    _, inverse_indices, counts = torch.unique(edges, dim=0, sorted=False, return_inverse=True, return_counts=True)

    # print("counts.max():", counts.max())
    assert counts.max() == 2

    # We now create a tensor that contains corresponding faces.
    # If the faces with ids fi and fj share the same edge, the tensor contains them as
    # [..., [fi, fj], ...]
    face_ids = torch.arange(indices.shape[0])               
    face_ids = torch.repeat_interleave(face_ids, 3, dim=0) # Tensor with the face id for each edge

    face_correspondences = torch.zeros((counts.shape[0], 2), dtype=torch.int64)
    face_correspondences_indices = torch.zeros(counts.shape[0], dtype=torch.int64)

    # ei = edge index
    for ei, ei_unique in enumerate(list(inverse_indices.cpu().numpy())):
        face_correspondences[ei_unique, face_correspondences_indices[ei_unique]] = face_ids[ei] 
        face_correspondences_indices[ei_unique] += 1

    face_correspondences = face_correspondences.cuda()

    return face_correspondences[counts == 2].to(device=indices.device), edges


######################################################################################
# Base mesh class
######################################################################################
class Mesh:
    def __init__(self, v_pos=None, t_pos_idx=None, v_nrm=None, t_nrm_idx=None, v_tex=None, t_tex_idx=None, v_tng=None, t_tng_idx=None, material=None, base=None, kd=None, ks=None, uv=None, uv_idx=None, face_labels=None, v_labels=None, connected_faces=None, edges=None):
        self.v_pos = v_pos
        self.v_nrm = v_nrm
        self.v_tex = v_tex
        self.v_tng = v_tng
        self.t_pos_idx = t_pos_idx
        self.t_nrm_idx = t_nrm_idx
        self.t_tex_idx = t_tex_idx
        self.t_tng_idx = t_tng_idx
        self.kd = kd
        self.ks = ks
        self.material = material
        self.uv = uv
        self.uv_idx = uv_idx
        self.face_labels = face_labels
        self.v_labels = v_labels
        self.connected_faces = connected_faces
        self.edges = edges

        if base is not None:
            self.copy_none(base)

        self.get_edge()
        # self.face_normals = self.get_face_normals()
        # self.connected_faces, self.edges = find_connected_faces(self.t_pos_idx)
        

    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
        if self.v_tex is None:
            self.v_tex = other.v_tex
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        if self.material is None:
            self.material = other.material
        if self.kd is None:
            self.kd = other.kd
        if self.ks is None:
            self.ks = other.ks
        if self.uv is None:
            self.uv = other.uv
        if self.uv_idx is None:
            self.uv_idx = other.uv_idx
        if self.face_labels is None:
            self.face_labels = other.face_labels
        if self.v_labels is None:
            self.v_labels = other.v_labels
        if self.connected_faces is None:
            self.connected_faces = other.connected_faces
        if self.edges is None:
            self.edges = other.edges

    def clone(self):
        out = Mesh(base=self)
        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone().detach()
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone().detach()
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone().detach()
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone().detach()
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone().detach()
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone().detach()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone().detach()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone().detach()
        if out.kd is not None:
            out.kd = out.kd.clone().detach()
        if out.ks is not None:
            out.ks = out.ks.clone().detach()
        if out.uv is not None:
            out.uv = out.uv.clone().detach()
        if out.uv_idx is not None:
            out.uv_idx = out.uv_idx.clone().detach()
        if self.face_labels is not None:
            out.face_labels = out.face_labels.clone().detach()
        if self.v_labels is not None:
            out.v_labels = out.v_labels.clone().detach()
        if self.connected_faces is not None:
            out.connected_faces = out.connected_faces.clone().detach()
        if self.edges is not None:
            out.edges = out.edges.clone().detach()

        return out

    def get_edge(self):
        edges = torch.cat([
            self.t_pos_idx[:, [0, 1]],
            self.t_pos_idx[:, [1, 2]],
            self.t_pos_idx[:, [2, 0]]
        ])
        edges = torch.sort(edges, dim=1).values
        edges_unique = torch.unique(edges, dim=0)
        self.edges = edges_unique

        return self.edges



    def get_face_normals(self):
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        return face_normals
    
    @property
    def laplacian(self):
        # if self._laplacian is None:
        self._laplacian = compute_laplacian_uniform(self)
        return self._laplacian

    # @property
    def normal_consistency(self):

        # self.connected_faces, self.edges = find_connected_faces(self.t_pos_idx)
        self.face_normals = self.get_face_normals()        
        normal_loss = normal_consistency_loss(self)
        
        return normal_loss
######################################################################################
# Mesh loeading helper
######################################################################################

def load_mesh(filename, mtl_override=None, mtl_default=None, mtl_type_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override, mtl_default=mtl_default, mtl_type_override=mtl_type_override)
    assert False, "Invalid mesh file extension"

######################################################################################
# Compute AABB
######################################################################################
def aabb(mesh):
    return torch.min(mesh.v_pos, dim=0).values, torch.max(mesh.v_pos, dim=0).values

######################################################################################
# Compute AABB with only used vertices
######################################################################################
def aabb_clean(mesh):
    v_pos_clean = mesh.v_pos[mesh.t_pos_idx.unique()]
    return torch.min(v_pos_clean, dim=0).values, torch.max(v_pos_clean, dim=0).values

######################################################################################
# Compute unique edge list from attribute/vertex index list
######################################################################################
def compute_edges(attr_idx, return_inverse=False):
    with torch.no_grad():
        # Create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Eliminate duplicates and return inverse mapping
        return torch.unique(sorted_edges, dim=0, return_inverse=return_inverse)

######################################################################################
# Compute unique edge to face mapping from attribute/vertex index list
######################################################################################
def compute_edge_to_face_mapping(attr_idx, return_inverse=False):
    with torch.no_grad():
        # Get unique edges
        # Create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Elliminate duplicates and return inverse mapping
        unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        # Compute edge to face table
        mask0 = order[:,0] == 0
        mask1 = order[:,0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge

######################################################################################
# Align base mesh to reference mesh:move & rescale to match bounding boxes.
######################################################################################
def unit_size(mesh):
    with torch.no_grad():
        vmin, vmax = aabb(mesh)
        scale = 2 / torch.max(vmax - vmin).item()
        v_pos = mesh.v_pos - (vmax + vmin) / 2 # Center mesh on origin
        v_pos = v_pos * scale                  # Rescale to unit size

        return Mesh(v_pos, base=mesh)

######################################################################################
# Center & scale mesh for rendering
######################################################################################
def center_by_reference(base_mesh, ref_aabb, scale):
    center = (ref_aabb[0] + ref_aabb[1]).cuda() * 0.5
    scale = scale / torch.max(ref_aabb[1] - ref_aabb[0]).item()
    v_pos = (base_mesh.v_pos - center[None, ...]) * scale
    return Mesh(v_pos, base=base_mesh)


def center_by_reference_noscale(base_mesh, ref_aabb, scale=None):
    center = (ref_aabb[0] + ref_aabb[1]) * 0.5
    v_pos = (base_mesh.v_pos - center[None, ...])
    return Mesh(v_pos, base=base_mesh)


def center_with_global_aabb(base_mesh, ref_aabb, scale):
    # center = (base_mesh.v_pos.min(dim=0).values + base_mesh.v_pos.max(dim=0).values) * 0.5 ### used for the experiments... wrong
    center = ref_aabb.mean(dim=0).cuda()
    scale = scale / torch.max(ref_aabb[1] - ref_aabb[0]).item() * 2.0
    v_pos = (base_mesh.v_pos - center[None, ...]) * scale
    return Mesh(v_pos, base=base_mesh)


def center_with_global_aabb_perdim(base_mesh, ref_aabb, scale):
    center = (base_mesh.v_pos.min(dim=0).values + base_mesh.v_pos.max(dim=0).values) * 0.5
    scale = scale / (ref_aabb[1] - ref_aabb[0])
    v_pos = (base_mesh.v_pos - center[None, ...]) * scale.view(1, 3)
    return Mesh(v_pos, base=base_mesh)


def scale_with_global_aabb(base_mesh, ref_aabb, scale):
    scale = scale / torch.max(ref_aabb[1] - ref_aabb[0]).item() * 0.5
    v_pos = base_mesh.v_pos * scale
    return Mesh(v_pos, base=base_mesh)


def scale_with_global_aabb_perdim(base_mesh, ref_aabb, scale):
    scale = scale / (ref_aabb[1] - ref_aabb[0])
    v_pos = base_mesh.v_pos * scale.view(1, 3)
    return Mesh(v_pos, base=base_mesh)

######################################################################################
# Simple smooth vertex normal computation
######################################################################################
def auto_normals(imesh):

    i0 = imesh.t_pos_idx[:, 0]
    i1 = imesh.t_pos_idx[:, 1]
    i2 = imesh.t_pos_idx[:, 2]

    v0 = imesh.v_pos[i0, :]
    v1 = imesh.v_pos[i1, :]
    # print("----imesh:", imesh.v_pos)
    # print("----i2:", i2)
    # exit()
    v2 = imesh.v_pos[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(imesh.v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(util.dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_nrm = util.safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return Mesh(v_nrm=v_nrm, t_nrm_idx=imesh.t_pos_idx, base=imesh)

######################################################################################
# Compute tangent space from texture map coordinates
# Follows http://www.mikktspace.com/ conventions
######################################################################################
def compute_tangents(imesh, v_tng=None):
    if v_tng is not None:
        v_tng = util.safe_normalize(v_tng)
        v_tng = util.safe_normalize(v_tng - util.dot(v_tng, imesh.v_nrm) * imesh.v_nrm)
        return Mesh(v_tng=v_tng, t_tng_idx=imesh.t_nrm_idx, base=imesh)

    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    for i in range(0,3):
        pos[i] = imesh.v_pos[imesh.t_pos_idx[:, i]]
        tex[i] = imesh.v_tex[imesh.t_tex_idx[:, i]]
        vn_idx[i] = imesh.t_nrm_idx[:, i]

    tangents = torch.zeros_like(imesh.v_nrm)
    tansum   = torch.zeros_like(imesh.v_nrm)

    # Compute tangent space for each triangle
    uve1 = tex[1] - tex[0]
    uve2 = tex[2] - tex[0]
    pe1  = pos[1] - pos[0]
    pe2  = pos[2] - pos[0]
    
    nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
    denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
    
    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

    # Update all 3 vertices
    for i in range(0,3):
        idx = vn_idx[i][:, None].repeat(1,3)
        tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang
        tansum.scatter_add_(0, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
    tangents = tangents / tansum

    # Normalize and make sure tangent is perpendicular to normal
    tangents = util.safe_normalize(tangents)
    tangents = util.safe_normalize(tangents - util.dot(tangents, imesh.v_nrm) * imesh.v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return Mesh(v_tng=tangents, t_tng_idx=imesh.t_nrm_idx, base=imesh)
