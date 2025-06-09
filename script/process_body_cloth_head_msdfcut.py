import os
import numpy as np
import openmesh as om
import open3d as o3d
# from scipy.spatial import cKDTree
import torch
from collections import deque
import pymeshlab as ml
import sys
import pysdf
from scipy.spatial import KDTree

def om_loadmesh(path):
    mesh = om.read_trimesh(path)
    v = np.array(mesh.points())
    f = np.array(mesh.face_vertex_indices())

    return v, f

def om_loadmesh_normal(path):
    mesh = om.read_trimesh(path)
    v = torch.from_numpy(np.array(mesh.points()))
    f = torch.from_numpy(np.array(mesh.face_vertex_indices()))
    if not mesh.has_vertex_normals():
        mesh.request_vertex_normals()
        mesh.update_normals() 
        mesh.release_vertex_normals() 

    vn = torch.from_numpy(np.array(mesh.vertex_normals()))
    return v, f, vn

def om_loadmesh_normal_numpy(path):
    mesh = om.read_trimesh(path)
    v = np.array(mesh.points())
    f = np.array(mesh.face_vertex_indices())
    if not mesh.has_vertex_normals():
        mesh.request_vertex_normals()
        mesh.update_normals() 
        mesh.release_vertex_normals() 

    vn = np.array(mesh.vertex_normals())

    return v, f, vn

def write_pc(path_mesh, v, vn=None, f=None):
    assert v.ndim == 2 and v.shape[1] == 3
    with open(path_mesh, 'w') as fp:
        fp.write(('v {:f} {:f} {:f}\n' * v.shape[0]).format(*v.reshape(-1)))
        if vn is not None:
            fp.write(('vn {:f} {:f} {:f}\n' * vn.shape[0]).format(*vn.reshape(-1)))
        if f is not None:
            fp.write(('f {:d} {:d} {:d}\n' * f.shape[0]).format(*f.reshape(-1) + 1))

def write_pc_batch(path_mesh, v, vn=None, f=None, batch_size=100000):
    assert v.ndim == 2 and v.shape[1] == 3, "顶点数组 `v` 必须是二维且每个顶点有三个坐标。"
    if f is not None:
        assert f.ndim == 2 and f.shape[1] == 3, "面数组 `f` 必须是二维且每个面有三个顶点索引。"
        assert np.issubdtype(f.dtype, np.integer), "面数组 `f` 必须是整数类型。"
        assert (f >= 0).all() and (f < v.shape[0]).all(), "面索引超出顶点范围。"

    with open(path_mesh, 'w') as fp:
        buffer = []
        for i, vertex in enumerate(v):
            buffer.append(f"v {vertex[0]:f} {vertex[1]:f} {vertex[2]:f}\n")
            if (i + 1) % batch_size == 0:
                fp.writelines(buffer)
                buffer = []
        if buffer:
            fp.writelines(buffer)
        
        if vn is not None:
            buffer = []
            for i, normal in enumerate(vn):
                buffer.append(f"vn {normal[0]:f} {normal[1]:f} {normal[2]:f}\n")
                if (i + 1) % batch_size == 0:
                    fp.writelines(buffer)
                    buffer = []
            if buffer:
                fp.writelines(buffer)
        
        if f is not None:
            buffer = []
            for i, face in enumerate(f):
                buffer.append(f"f {face[0]+1:d} {face[1]+1:d} {face[2]+1:d}\n")
                if (i + 1) % batch_size == 0:
                    fp.writelines(buffer)
                    buffer = []
            if buffer:
                fp.writelines(buffer)


def find_open_edges(faces):
    num_faces = faces.shape[0]
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    sorted_edges = torch.sort(edges, dim=1)[0]
    unique_edges, counts = torch.unique(sorted_edges, return_counts=True, dim=0)
    
    open_edge_mask = counts == 1
    open_edges = unique_edges[open_edge_mask]
    
    open_vertices = torch.unique(open_edges)
    return open_vertices.tolist(), unique_edges

def remove_triangles_with_vertices(triangles, vertex_indices):
    vertex_indices_set = set(vertex_indices)
    mask = ~np.isin(triangles, list(vertex_indices_set)).any(axis=1)
    triangles_to_keep = triangles[mask]

    return triangles_to_keep

def merge_and_repair_meshes(mesh1, mesh2):

    vertices = np.vstack((np.asarray(mesh1.vertices), np.asarray(mesh2.vertices)))
    triangles = np.vstack((np.asarray(mesh1.triangles), np.asarray(mesh2.triangles) + len(mesh1.vertices)))
    merged_mesh = o3d.geometry.TriangleMesh()
    merged_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    merged_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    merged_mesh = merged_mesh.remove_duplicated_vertices()
    merged_mesh = merged_mesh.remove_degenerate_triangles()
    
    return merged_mesh

def build_adjacency_list(faces):
    adjacency_list = {}
    faces = faces.tolist()
    for face in faces:
        for i in range(len(face)):
            v1, v2 = face[i], face[(i + 1) % len(face)]
            if v1 not in adjacency_list:
                adjacency_list[v1] = set()
            if v2 not in adjacency_list:
                adjacency_list[v2] = set()
            adjacency_list[v1].add(v2)
            adjacency_list[v2].add(v1)
    return adjacency_list

def filter_adjacency_list(adjacency_list, vertices):
    filtered_list = {v: adjacency_list.get(v, set()) & set(vertices) for v in vertices if v in adjacency_list}
    return filtered_list

def dfs(vertex, adjacency_list, visited, component):
    stack = [vertex]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            component.append(node)
            stack.extend(adjacency_list.get(node, []) - visited)

def remove_faces_with_open_vertices(faces, open_vertices):
    open_vertices = np.array(open_vertices)
    mask = np.isin(faces, open_vertices).any(axis=1)
    filtered_faces = faces[~mask]
    
    return filtered_faces

def find_connected_components(vertices, adjacency_list):
    visited = set()
    components = []
    for vertex in vertices:
        if vertex not in visited:
            component = []
            dfs(vertex, adjacency_list, visited, component)
            components.append(component)
    return components

def find_largest_components(components, number=5):
    largest_components = sorted(components, key=len, reverse=True)[:number]
    return largest_components

def find_largest_and_other_components(components, number=5):

    sorted_components = sorted(components, key=len, reverse=True)
    largest_components = sorted_components[:number]
    other_components = sorted_components[number:]

    return largest_components, other_components


def write_ply(path_mesh, points, normals=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(path_mesh, pcd)


def fit_plane(points):
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid
    covariance_matrix = np.cov(shifted_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    D = -np.dot(normal_vector, centroid)
    return normal_vector, D

def filter_points_near_plane_and_points(mesh, plane, threshold, reference_points, ref_point_threshold):
    A, B, C, D = plane
    
    def point_plane_distance(point):
        return abs(A * point[0] + B * point[1] + C * point[2] + D)

    vertices = np.asarray(mesh.vertices)
    distances = np.apply_along_axis(point_plane_distance, 1, vertices)
    reference_points = np.array(reference_points)
    min_distances_to_ref_points = np.min(np.linalg.norm(vertices[:, np.newaxis] - reference_points, axis=2), axis=1)
    
    near_plane_indices = np.where((distances <= threshold) & (min_distances_to_ref_points <= ref_point_threshold))[0]
    near_plane_vertices = vertices[near_plane_indices]
    
    return near_plane_vertices, near_plane_indices


def remove_faces_through_plane(mesh, plane, points, distance_threshold):
    A, B, C, D = plane
    reference_points = np.array(points)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    def point_plane_distance(point):
        return A * point[0] + B * point[1] + C * point[2] + D

    def face_to_points_distance(face):
        face_points = vertices[face]
        dist_matrix = np.linalg.norm(face_points[:, np.newaxis] - reference_points, axis=2)
        return np.min(dist_matrix)

    faces_to_keep = []
    for face in triangles:
        distances = [point_plane_distance(vertices[i]) for i in face]
        if (np.any(np.array(distances) > 0) and np.any(np.array(distances) < 0)):
            if face_to_points_distance(face) <= distance_threshold:
                continue  # Skip adding this face
        faces_to_keep.append(face)

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = mesh.vertices
    new_mesh.triangles = o3d.utility.Vector3iVector(faces_to_keep)
    new_mesh.compute_vertex_normals()

    return new_mesh

def remove_distant_points(points, reference_points, distance_threshold):
    points_array = np.array(points)
    reference_points_array = np.array(reference_points)
    distances = np.linalg.norm(points_array[:, np.newaxis] - reference_points_array, axis=2)

    min_distances = np.min(distances, axis=1)
    close_points_mask = min_distances >= distance_threshold
    new_points = points_array[close_points_mask]

    return new_points


def remove_connected_edges(mesh, points_to_remove):
    triangles = np.asarray(mesh.triangles)
    points_to_remove = set(points_to_remove)
    mask = ~np.any(np.isin(triangles, list(points_to_remove)), axis=1)
    new_triangles = triangles[mask]
    
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = mesh.vertices
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    new_mesh.compute_vertex_normals()  

    return new_mesh


def find_largest_connected_component(mesh):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    graph = build_graph(triangles)
    
    visited = set()
    largest_component = []
    max_size = 0
    
    for node in graph:
        if node not in visited:
            size = 0
            component = []
            queue = deque([node])
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    size += 1
                    queue.extend(graph[current] - visited)
            if size > max_size:
                max_size = size
                largest_component = component

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices[largest_component])
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(largest_component)}
    new_triangles = []
    for tri in triangles:
        if all(v in largest_component for v in tri):
            new_triangles.append([index_map[v] for v in tri])
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    new_mesh.compute_vertex_normals()
    
    return new_mesh

def build_graph(triangles):
    graph = {}
    for tri in triangles:
        for i in range(3):
            if tri[i] not in graph:
                graph[tri[i]] = set()
            graph[tri[i]].update(tri[j] for j in range(3) if j != i)
    return graph

def run_meshlabserver_with_os_system(input_file, output_file, script_file, local=False):
    if local:
        command = f'meshlabserver -i {input_file} -o {output_file} -s {script_file}'
    else:
        command = f'xvfb-run meshlabserver -i {input_file} -o {output_file} -s {script_file}'
    exit_status = os.system(command)
    
    if exit_status == 0:
        print("MeshLabServer executed successfully!")
    else:
        print("Error in MeshLabServer execution, exit status:", exit_status)


def deform_body_collision(cloth_v, cloth_n, body_v, body_n, sdf_gt_fn):

    kdtree_B = KDTree(cloth_v)
    epsilon = 0.005
    deformed_mesh_A = np.copy(body_v)

    for j in range(100):
        sdf_values = sdf_gt_fn(np.array(deformed_mesh_A))
        for i, sdf_val in enumerate(sdf_values):
            if sdf_val < 0.002:  # If the vertex is outside

                _, index_B = kdtree_B.query(deformed_mesh_A[i])

                normal = body_n[i]
                vec_norm = np.linalg.norm(normal)
                unit_norm = normal/(vec_norm + 0.0000001)
                deformed_mesh_A[i] -= unit_norm * epsilon 

    return deformed_mesh_A

def merge_meshes_with_face_labels(mesh1, mesh2):
    vertices1 = np.asarray(mesh1.vertices)
    vertices2 = np.asarray(mesh2.vertices)
    triangles1 = np.asarray(mesh1.triangles)
    triangles2 = np.asarray(mesh2.triangles)
    
    vertices = np.vstack((vertices1, vertices2))
    
    triangles2 += len(vertices1)
    triangles = np.vstack((triangles1, triangles2))
    
    labels1 = np.zeros(len(triangles1), dtype=int) 
    labels2 = np.ones(len(triangles2), dtype=int) 
    labels = np.concatenate((labels1, labels2))
    
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    new_mesh.compute_vertex_normals()  

    return new_mesh, labels, vertices, triangles

def find_inside_point(v, body_index, sdf_gt_fn, root):
    body_v = v[body_index]

    sdf_values = sdf_gt_fn(body_v)

    inside_index = np.where(sdf_values > -0.002)
    inside_body_index = np.nonzero(np.isin(body_index, body_index[inside_index]))[0]
    inside_body_v = v[inside_body_index]
    write_pc(os.path.join(root, "inside_body_v.obj"), v=inside_body_v)

    outside_index = np.where(sdf_values <= -0.002)
    outside_body_index = np.nonzero(np.isin(body_index, body_index[outside_index]))[0]
    outside_body_v = v[outside_body_index]
    write_pc(os.path.join(root, "outside_body_v.obj"), v=outside_body_v) 

    return inside_body_index, outside_body_index


def face_in_bbox(v_indices, f):
    face_in_bbox_list = []
    face_out_bbox_list = []
    for i in f:
        (v1, v2, v3) = i
        if v1 in v_indices and v2 in v_indices and v3 in v_indices:
            face_in_bbox_list.append([v1, v2, v3])
        else:
            face_out_bbox_list.append([v1, v2, v3])

    return np.array(face_in_bbox_list), np.array(face_out_bbox_list)
        

def process_subdivide(body_path, bbox_npz_path, root, script_root, local):

    bbox_npz = np.load(bbox_npz_path)
    bbox_min = bbox_npz["bbox_min"]
    bbox_max = bbox_npz["bbox_max"]

    body_v, body_f, body_vn = om_loadmesh_normal_numpy(body_path)
    mask = np.all((body_v >= (bbox_min-0.01)) & (body_v <= (bbox_max+0.01)), axis=1)
    points_in_bbox = body_v[mask]
    indices_in_bbox = np.where(mask)[0]

    face_in_bbox_list, face_out_bbox_list = face_in_bbox(indices_in_bbox, body_f)
    save_path = os.path.join(root, "body_subdivide.obj")

    inhead_path = os.path.join(root, "p_inhead.obj")
    write_pc(inhead_path, v=body_v, f=face_in_bbox_list)
    script_file = os.path.join(script_root, "midpoint_head.mlx")
    out_subdivide_path = os.path.join(root, "p_inhead_subdivide.obj")
    run_meshlabserver_with_os_system(inhead_path, out_subdivide_path, script_file, local)

    body_midhead_v, body_midhead_f, body_midhead_vn = om_loadmesh_normal_numpy(out_subdivide_path)

    add_f = np.vstack((body_midhead_f, face_out_bbox_list))

    add_v_path = os.path.join(root, "all_v_f.obj")
    write_pc(add_v_path, v=body_midhead_v, f=add_f)

    return add_v_path

def compute_distance_map(partial_v, complete_v):

    min_distances = [] 
    kdtree_B = KDTree(partial_v)
    complete_v_len = complete_v.shape[0]

    for i in range(complete_v_len):
        v = complete_v[i]
        distance, index = kdtree_B.query(v)  
        min_distances.append(distance) 

    min_distances = np.array(min_distances) 

    return min_distances

def retain_valid_faces(vertices, faces, valid_indices):

    valid_mask = np.zeros(len(vertices), dtype=bool)
    valid_mask[valid_indices] = True

    valid_faces = []
    novalid_faces = []

    for face in faces:
        if np.all(valid_mask[list(face)]):  
            valid_faces.append(face)
        else:
            novalid_faces.append(face)

    return np.array(valid_faces), np.array(novalid_faces)



def segment_mesh_by_distance(vertices, faces, distances, threshold):
    mask = distances <= threshold
    
    keep_indices = np.where(mask)[0]
    valid_faces, novalid_faces = retain_valid_faces(vertices, faces, keep_indices)

    return valid_faces, novalid_faces

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p]) 
        return self.parent[p]
    
    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            self.parent[rootQ] = rootP 

def find_connected_components(vertices, faces):
    uf = UnionFind(len(vertices))
    for face in faces:
        uf.union(face[0], face[1])
        uf.union(face[1], face[2])
    
    component_map = {}
    for vertex in range(len(vertices)):
        root = uf.find(vertex)
        if root not in component_map:
            component_map[root] = []
        component_map[root].append(vertex)
    
    components = list(component_map.values())
    return components

def filter_and_sort_components(components):
    filtered_components = [comp for comp in components if len(comp) > 1]
    sorted_components = sorted(filtered_components, key=len, reverse=True)
    return sorted_components


def process_body_msdf_distance(FLAGS, root, body_path, ori_cloth_path, smpl_cloth_path, smpl_path, bbox_npz_path, script_root = "checkpoints/script", local=False):
    
    wt_script_path = os.path.join(script_root, "wt.mlx")
    midpoint_script_path = os.path.join(script_root, "midpoint.mlx")
    remesh_script_path = os.path.join(script_root, "remesh.mlx")

    # remesh for poisson mesh
    remesh_smpl_path = os.path.join(root, "remesh_smpl.ply")
    run_meshlabserver_with_os_system(smpl_path, remesh_smpl_path, remesh_script_path, local)
    smpl_path = remesh_smpl_path

    remesh_ori_cloth_path = os.path.join(root, "remesh_ori_cloth.ply")
    run_meshlabserver_with_os_system(ori_cloth_path, remesh_ori_cloth_path, remesh_script_path, local)
    ori_cloth_path = remesh_ori_cloth_path

    cloth_v, cloth_f = om_loadmesh(smpl_cloth_path)
    smpl_v, smpl_f = om_loadmesh(smpl_path)

    min_distances = compute_distance_map(cloth_v, smpl_v)
    valid_faces, novalid_faces = segment_mesh_by_distance(smpl_v, smpl_f, min_distances, threshold=0.02)

    valid_smpl_connected_components = find_connected_components(smpl_v, valid_faces)
    valid_smpl_sorted_components = filter_and_sort_components(valid_smpl_connected_components)

    novalid_smpl_connected_components = find_connected_components(smpl_v, novalid_faces)
    novalid_smpl_sorted_components = filter_and_sort_components(novalid_smpl_connected_components)

    valid_smpl = []

    for i, comp in enumerate(valid_smpl_sorted_components):

        if i < 1:
            valid_smpl.append(comp)    

    for i, comp in enumerate(novalid_smpl_sorted_components):

        if i >= 5:
            valid_smpl.append(comp)    

    valid_smpl_face = []
    for idx, i in enumerate(valid_smpl):
        vertices_set = set(i)
        for face_i in smpl_f.tolist():
            if set(face_i).issubset(vertices_set):
                valid_smpl_face.append(face_i)


    save_cut_smpl_path = os.path.join(root, "cut_smpl.obj")
    write_pc(save_cut_smpl_path, v=smpl_v, f=np.array(valid_smpl_face))

    save_cloth_wt_path = os.path.join(root, "cloth_wt.ply")
    run_meshlabserver_with_os_system(ori_cloth_path, save_cloth_wt_path, wt_script_path, local)

    save_cut_smpl_midpoint_path = save_cut_smpl_path
    # colli for body
    cut_smpl_v, cut_smpl_f, cut_smpl_n = om_loadmesh_normal_numpy(save_cut_smpl_midpoint_path)
    wt_cloth_v, wt_cloth_f, wt_cloth_n= om_loadmesh_normal_numpy(save_cloth_wt_path)
    sdf_gt_fn = pysdf.SDF(wt_cloth_v, wt_cloth_f)
    deformed_body_mesh_v = deform_body_collision(wt_cloth_v, wt_cloth_n, cut_smpl_v, cut_smpl_n, sdf_gt_fn)
    save_deformed_smpl_path = os.path.join(root, "deform_smpl.obj")
    write_pc(save_deformed_smpl_path, v=deformed_body_mesh_v, f=cut_smpl_f)

    # merge body and smpl
    merge_path = os.path.join(root, "merge_mesh.ply")
    mesh1 = o3d.io.read_triangle_mesh(save_deformed_smpl_path)
    mesh2 = o3d.io.read_triangle_mesh(body_path)

    merge_mesh = merge_and_repair_meshes(mesh1, mesh2)
    o3d.io.write_triangle_mesh(merge_path, merge_mesh)

    # possion for merge mesh
    merge_wt_path = os.path.join(root, "merge_wt_mesh.ply")

    run_meshlabserver_with_os_system(merge_path, merge_wt_path, wt_script_path, local)
    print("write body to:", merge_wt_path)


    #================================
    subdivide_body_path = process_subdivide(merge_wt_path, bbox_npz_path, root, script_root, local)
    #==================================

    # merge 2 mesh
    merge_npz_path = os.path.join(root, "merge_body_cloth.npz")
    mesh1 = o3d.io.read_triangle_mesh(subdivide_body_path)
    mesh2 = o3d.io.read_triangle_mesh(remesh_ori_cloth_path)
    merged_mesh, face_labels, v, f = merge_meshes_with_face_labels(mesh1, mesh2)

    merge_obj_path = os.path.join(root, "merge_body_cloth.obj")
    write_pc(merge_obj_path, v=v, f=f)
    np.savez(merge_npz_path, v=v, f=f, face_labels=face_labels)

    f_body = f[face_labels==0]
    f_cloth = f[face_labels==1]
    body_index = np.unique(f_body)
    wt_cloth_v, wt_cloth_f, wt_cloth_n= om_loadmesh_normal_numpy(save_cloth_wt_path)
    cloth_wt_fn = pysdf.SDF(wt_cloth_v, wt_cloth_f)
    inside_body_index, outside_body_index = find_inside_point(v, body_index, cloth_wt_fn, root)
    inside_body_index_path = os.path.join(root, "inside_body_index.npz")
    np.savez(inside_body_index_path, inside_body_index=inside_body_index, outside_body_index=outside_body_index)



def process_body_msdf_distance_bodyedge(FLAGS, root, body_path, ori_cloth_path, smpl_cloth_path, smpl_path, bbox_npz_path, script_root = "checkpoints/script", local=False):
    
    wt_script_path = os.path.join(script_root, "wt.mlx")
    midpoint_script_path = os.path.join(script_root, "midpoint.mlx")
    remesh_script_path = os.path.join(script_root, "remesh.mlx")

    # remesh for poisson mesh
    remesh_smpl_path = os.path.join(root, "remesh_smpl.ply")
    run_meshlabserver_with_os_system(smpl_path, remesh_smpl_path, remesh_script_path, local)
    smpl_path = remesh_smpl_path

    remesh_ori_cloth_path = os.path.join(root, "remesh_ori_cloth.obj")
    run_meshlabserver_with_os_system(ori_cloth_path, remesh_ori_cloth_path, remesh_script_path, local)
    ori_cloth_path = remesh_ori_cloth_path

    cloth_v, cloth_f = om_loadmesh(smpl_cloth_path)
    smpl_v, smpl_f = om_loadmesh(smpl_path)

    min_distances = compute_distance_map(cloth_v, smpl_v)
    valid_faces, novalid_faces = segment_mesh_by_distance(smpl_v, smpl_f, min_distances, threshold=0.02)

    valid_smpl_connected_components = find_connected_components(smpl_v, valid_faces)
    valid_smpl_sorted_components = filter_and_sort_components(valid_smpl_connected_components)

    novalid_smpl_connected_components = find_connected_components(smpl_v, novalid_faces)
    novalid_smpl_sorted_components = filter_and_sort_components(novalid_smpl_connected_components)

    valid_smpl = []

    for i, comp in enumerate(valid_smpl_sorted_components):
        if i < 1:
            valid_smpl.append(comp)    

    for i, comp in enumerate(novalid_smpl_sorted_components):
        if i >= 5:
            valid_smpl.append(comp)    

    valid_smpl_face = []
    for idx, i in enumerate(valid_smpl):
        vertices_set = set(i)
        for face_i in smpl_f.tolist():
            if set(face_i).issubset(vertices_set):
                valid_smpl_face.append(face_i)


    save_cut_smpl_path = smpl_cloth_path

    remesh_cut_smpl_path = os.path.join(root, "remesh_cut_smpl.ply")
    run_meshlabserver_with_os_system(save_cut_smpl_path, remesh_cut_smpl_path, remesh_script_path, local)

    # # 使用 MeshLabServer 执行一个脚本
    # "meshlabserver -i input_mesh.ply -o output_mesh.ply -s filter_script.mlx"
    # wt for cloth
    save_cloth_wt_path = os.path.join(root, "cloth_wt.ply")
    run_meshlabserver_with_os_system(ori_cloth_path, save_cloth_wt_path, wt_script_path, local)

    ##############################
    remesh_cut_smpl_v, remesh_cut_smpl_f, remesh_cut_smpl_n = om_loadmesh_normal(remesh_cut_smpl_path)
    for i in range(1): # 4
        body_open_vertices, _ = find_open_edges(remesh_cut_smpl_f)
        remesh_cut_smpl_f = remove_faces_with_open_vertices(remesh_cut_smpl_f, body_open_vertices)
    remesh_cut_smpl_edge_cut_path = os.path.join(root, "remesh_cut_smpl_edge.obj")
    write_pc(remesh_cut_smpl_edge_cut_path, v=remesh_cut_smpl_v, f=remesh_cut_smpl_f)
    ##############################

    # save_cut_smpl_midpoint_path = remesh_cut_smpl_path
    # colli for body
    print("remesh_cut_smpl_edge_cut_path:", remesh_cut_smpl_edge_cut_path)
    cut_smpl_v, cut_smpl_f, cut_smpl_n = om_loadmesh_normal_numpy(remesh_cut_smpl_edge_cut_path)
    wt_cloth_v, wt_cloth_f, wt_cloth_n= om_loadmesh_normal_numpy(save_cloth_wt_path)
    sdf_gt_fn = pysdf.SDF(wt_cloth_v, wt_cloth_f)
    
    deformed_body_mesh_v = deform_body_collision(wt_cloth_v, wt_cloth_n, cut_smpl_v, cut_smpl_n, sdf_gt_fn)
    save_deformed_smpl_path = os.path.join(root, "deform_smpl.obj")
    write_pc(save_deformed_smpl_path, v=deformed_body_mesh_v, f=cut_smpl_f)


    # remove cut body edge
    body_v, body_f, body_n = om_loadmesh_normal(body_path)
    for i in range(2): # 5
        body_open_vertices, _ = find_open_edges(body_f)
        body_f = remove_faces_with_open_vertices(body_f, body_open_vertices)
    body_edge_cut_path = os.path.join(root, "body_edge_cut.obj")
    write_pc(body_edge_cut_path, v=body_v, f=body_f)

    # merge body and smpl
    body_path = body_edge_cut_path
    merge_path = os.path.join(root, "merge_mesh.ply")
    mesh1 = o3d.io.read_triangle_mesh(save_deformed_smpl_path)
    mesh2 = o3d.io.read_triangle_mesh(body_path)

    
    merge_mesh = merge_and_repair_meshes(mesh1, mesh2)
    o3d.io.write_triangle_mesh(merge_path, merge_mesh)

    # possion for merge mesh
    merge_wt_path = os.path.join(root, "merge_wt_mesh.ply")

    run_meshlabserver_with_os_system(merge_path, merge_wt_path, wt_script_path, local)
    print("write body to:", merge_wt_path)

    merge_wt_remesh_path = os.path.join(root, "merge_wt_remesh.ply")
    run_meshlabserver_with_os_system(merge_wt_path, merge_wt_remesh_path, remesh_script_path, local)

    #================================
    subdivide_body_path = process_subdivide(merge_wt_remesh_path, bbox_npz_path, root, script_root, local)
    #==================================

    # merge 2 mesh
    merge_npz_path = os.path.join(root, "merge_body_cloth.npz")
    mesh1 = o3d.io.read_triangle_mesh(subdivide_body_path)
    mesh2 = o3d.io.read_triangle_mesh(remesh_ori_cloth_path)
    print("remesh_ori_cloth_path:", remesh_ori_cloth_path)

    merged_mesh, face_labels, v, f = merge_meshes_with_face_labels(mesh1, mesh2)

    print(f"f before write_pc: shape = {f.shape}")
    print(f"f before write_pc:\n{f}")

    merge_obj_path = os.path.join(root, "merge_body_cloth.obj")
    write_pc_batch(merge_obj_path, v=v, f=f)
    
    print(f"f after write_pc: shape = {f.shape}")
    print(f"f after write_pc:\n{f}")

    np.savez(merge_npz_path, v=v, f=f, face_labels=face_labels)

    f_body = f[face_labels==0]
    f_cloth = f[face_labels==1]
    body_index = np.unique(f_body)
    wt_cloth_v, wt_cloth_f, wt_cloth_n= om_loadmesh_normal_numpy(save_cloth_wt_path)
    cloth_wt_fn = pysdf.SDF(wt_cloth_v, wt_cloth_f)
    inside_body_index, outside_body_index = find_inside_point(v, body_index, cloth_wt_fn, root)
    inside_body_index_path = os.path.join(root, "inside_body_index.npz")
    np.savez(inside_body_index_path, inside_body_index=inside_body_index, outside_body_index=outside_body_index)




def process_body_msdf(FLAGS, root, body_path, ori_cloth_path, smpl_cloth_path, smpl_path, bbox_npz_path, script_root = "checkpoints/script", local=False):
    
    wt_script_path = os.path.join(script_root, "wt.mlx")
    midpoint_script_path = os.path.join(script_root, "midpoint.mlx")
    remesh_script_path = os.path.join(script_root, "remesh.mlx")

    # remesh for poisson mesh
    remesh_smpl_path = os.path.join(root, "remesh_smpl.ply")
    run_meshlabserver_with_os_system(smpl_path, remesh_smpl_path, remesh_script_path, local)
    smpl_path = remesh_smpl_path

    remesh_ori_cloth_path = os.path.join(root, "remesh_ori_cloth.ply")
    run_meshlabserver_with_os_system(ori_cloth_path, remesh_ori_cloth_path, remesh_script_path, local)
    ori_cloth_path = remesh_ori_cloth_path

    cloth_v, cloth_f, cloth_n = om_loadmesh_normal(smpl_cloth_path)
    cloth_open_vertices, cloth_unique_edges = find_open_edges(cloth_f)

    threshold = 0.01
    full_adjacency_list = build_adjacency_list(cloth_f)
    filtered_adjacency_list = filter_adjacency_list(full_adjacency_list, cloth_open_vertices)
    components = find_connected_components(cloth_open_vertices, filtered_adjacency_list)
    largest_components = find_largest_components(components, number=6)
    smpl_mesh = o3d.io.read_triangle_mesh(smpl_path)

    for i, comp in enumerate(largest_components):
        select_v = cloth_v[comp]
        normal_vector, D = fit_plane(select_v)
        plane = (*normal_vector, D)

        # 获取平面附近的顶点
        near_plane_points, near_plane_indices = filter_points_near_plane_and_points(smpl_mesh, plane, threshold, reference_points=select_v, ref_point_threshold=0.03)

        remove_faces_mesh = remove_faces_through_plane(smpl_mesh, plane, select_v, distance_threshold=0.12)
        o3d.io.write_triangle_mesh(os.path.join(root, "remove_faces_mesh_{}.ply".format(i)), remove_faces_mesh)

        # 保存新的网格
        # o3d.io.write_triangle_mesh("./close_hole/smpl_remove_face_{}.ply".format(i), new_mesh)
        # o3d.io.write_triangle_mesh(os.path.join(root, "smpl_remove_face_{}.ply".format(i)), new_mesh)
        smpl_mesh = find_largest_connected_component(remove_faces_mesh)

    save_cut_smpl_path = os.path.join(root, "cut_smpl.ply")
    o3d.io.write_triangle_mesh(save_cut_smpl_path, smpl_mesh)

    # # 使用 MeshLabServer 执行一个脚本
    # "meshlabserver -i input_mesh.ply -o output_mesh.ply -s filter_script.mlx"
    # wt for cloth
    save_cloth_wt_path = os.path.join(root, "cloth_wt.ply")
    run_meshlabserver_with_os_system(ori_cloth_path, save_cloth_wt_path, wt_script_path, local)

    save_cut_smpl_midpoint_path = os.path.join(root, "cut_smpl_midpoint.ply")
    run_meshlabserver_with_os_system(save_cut_smpl_path, save_cut_smpl_midpoint_path, midpoint_script_path, local)

    # colli for body
    cut_smpl_v, cut_smpl_f, cut_smpl_n = om_loadmesh_normal_numpy(save_cut_smpl_midpoint_path)
    wt_cloth_v, wt_cloth_f, wt_cloth_n= om_loadmesh_normal_numpy(save_cloth_wt_path)
    sdf_gt_fn = pysdf.SDF(wt_cloth_v, wt_cloth_f)
    deformed_body_mesh_v = deform_body_collision(wt_cloth_v, wt_cloth_n, cut_smpl_v, cut_smpl_n, sdf_gt_fn)
    save_deformed_smpl_path = os.path.join(root, "deform_smpl.obj")
    write_pc(save_deformed_smpl_path, v=deformed_body_mesh_v, f=cut_smpl_f)

    # merge body and smpl
    merge_path = os.path.join(root, "merge_mesh.ply")
    mesh1 = o3d.io.read_triangle_mesh(save_deformed_smpl_path)
    mesh2 = o3d.io.read_triangle_mesh(body_path)
    merge_mesh = merge_and_repair_meshes(mesh1, mesh2)
    o3d.io.write_triangle_mesh(merge_path, merge_mesh)

    # possion for merge mesh
    merge_wt_path = os.path.join(root, "merge_wt_mesh.ply")

    # return merge body
    # return merge_wt_path

    run_meshlabserver_with_os_system(merge_path, merge_wt_path, wt_script_path, local)
    print("write body to:", merge_wt_path)
    # print("write cloth to:", remesh_cloth_path)

    #================================
    subdivide_body_path = process_subdivide(merge_wt_path, bbox_npz_path, root, script_root, local)
    #==================================

    # merge 2 mesh
    merge_npz_path = os.path.join(root, "merge_body_cloth.npz")
    mesh1 = o3d.io.read_triangle_mesh(subdivide_body_path)
    mesh2 = o3d.io.read_triangle_mesh(remesh_ori_cloth_path)
    merged_mesh, face_labels, v, f = merge_meshes_with_face_labels(mesh1, mesh2)

    merge_obj_path = os.path.join(root, "merge_body_cloth.obj")
    write_pc(merge_obj_path, v=v, f=f)
    np.savez(merge_npz_path, v=v, f=f, face_labels=face_labels)

    f_body = f[face_labels==0]
    f_cloth = f[face_labels==1]
    body_index = np.unique(f_body)
    wt_cloth_v, wt_cloth_f, wt_cloth_n= om_loadmesh_normal_numpy(save_cloth_wt_path)
    cloth_wt_fn = pysdf.SDF(wt_cloth_v, wt_cloth_f)
    inside_body_index, outside_body_index = find_inside_point(v, body_index, cloth_wt_fn, root)
    inside_body_index_path = os.path.join(root, "inside_body_index.npz")
    np.savez(inside_body_index_path, inside_body_index=inside_body_index, outside_body_index=outside_body_index)


def process_msdf(FLAGS, root, body_path, cloth_path, smpl_path, bbox_npz_path, script_root = "checkpoints/script", local=False):

    wt_script_path = os.path.join(script_root, "wt.mlx")
    midpoint_script_path = os.path.join(script_root, "midpoint.mlx")
    remesh_script_path = os.path.join(script_root, "remesh.mlx")

    # remesh for poisson mesh
    remesh_cloth_path = os.path.join(root, "remesh_cloth.ply")
    run_meshlabserver_with_os_system(cloth_path, remesh_cloth_path, remesh_script_path, local)
    cloth_path = remesh_cloth_path

    body_v, body_f, body_n = om_loadmesh_normal(body_path)
    print("cloth_path:", cloth_path)
    cloth_v, cloth_f, cloth_n = om_loadmesh_normal(cloth_path)
    smpl_v, smpl_f, smpl_n = om_loadmesh_normal(smpl_path)

    # remove body edge face
    body_open_vertices, body_unique_edges = find_open_edges(body_f)
    select_body_v = body_v[body_open_vertices]
    mesh_without_certain_vertices = remove_triangles_with_vertices(body_f, body_open_vertices)
    remove_body_path = os.path.join(root, "edge_remove_body.obj")
    write_pc(remove_body_path, v=body_v, f=np.array(mesh_without_certain_vertices))

    cloth_open_vertices, cloth_unique_edges = find_open_edges(cloth_f)


    threshold = 0.01
    full_adjacency_list = build_adjacency_list(cloth_f)
    filtered_adjacency_list = filter_adjacency_list(full_adjacency_list, cloth_open_vertices)
    components = find_connected_components(cloth_open_vertices, filtered_adjacency_list)
    largest_components = find_largest_components(components, number=6)
    smpl_mesh = o3d.io.read_triangle_mesh(smpl_path)

    for i, comp in enumerate(largest_components):
        select_v = cloth_v[comp]
        normal_vector, D = fit_plane(select_v)
        plane = (*normal_vector, D)

        # 获取平面附近的顶点
        near_plane_points, near_plane_indices = filter_points_near_plane_and_points(smpl_mesh, plane, threshold, reference_points=select_v, ref_point_threshold=0.03)

        remove_faces_mesh = remove_faces_through_plane(smpl_mesh, plane, select_v, distance_threshold=0.12)
        o3d.io.write_triangle_mesh(os.path.join(root, "remove_faces_mesh_{}.ply".format(i)), remove_faces_mesh)

        smpl_mesh = find_largest_connected_component(remove_faces_mesh)

    save_cut_smpl_path = os.path.join(root, "cut_smpl.ply")
    o3d.io.write_triangle_mesh(save_cut_smpl_path, smpl_mesh)

    # # 使用 MeshLabServer 执行一个脚本
    # "meshlabserver -i input_mesh.ply -o output_mesh.ply -s filter_script.mlx"
    # wt for cloth
    save_cloth_wt_path = os.path.join(root, "cloth_wt.ply")
    run_meshlabserver_with_os_system(cloth_path, save_cloth_wt_path, wt_script_path, local)

    save_cut_smpl_midpoint_path = os.path.join(root, "cut_smpl_midpoint.ply")
    run_meshlabserver_with_os_system(save_cut_smpl_path, save_cut_smpl_midpoint_path, midpoint_script_path, local)

    # colli for body
    cut_smpl_v, cut_smpl_f, cut_smpl_n = om_loadmesh_normal_numpy(save_cut_smpl_midpoint_path)
    wt_cloth_v, wt_cloth_f, wt_cloth_n= om_loadmesh_normal_numpy(save_cloth_wt_path)
    sdf_gt_fn = pysdf.SDF(wt_cloth_v, wt_cloth_f)
    deformed_body_mesh_v = deform_body_collision(wt_cloth_v, wt_cloth_n, cut_smpl_v, cut_smpl_n, sdf_gt_fn)
    save_deformed_smpl_path = os.path.join(root, "deform_smpl.obj")
    write_pc(save_deformed_smpl_path, v=deformed_body_mesh_v, f=cut_smpl_f)

    # merge body and smpl
    merge_path = os.path.join(root, "merge_mesh.ply")
    mesh1 = o3d.io.read_triangle_mesh(save_deformed_smpl_path)
    mesh2 = o3d.io.read_triangle_mesh(remove_body_path)
    merge_mesh = merge_and_repair_meshes(mesh1, mesh2)
    o3d.io.write_triangle_mesh(merge_path, merge_mesh)

    # possion for merge mesh
    merge_wt_path = os.path.join(root, "merge_wt_mesh.ply")
    run_meshlabserver_with_os_system(merge_path, merge_wt_path, wt_script_path, local)
    print("write body to:", merge_wt_path)
    print("write cloth to:", remesh_cloth_path)

    #================================
    subdivide_body_path = process_subdivide(merge_wt_path, bbox_npz_path, root, script_root)
    #==================================

    # merge 2 mesh
    merge_npz_path = os.path.join(root, "merge_body_cloth.npz")
    mesh1 = o3d.io.read_triangle_mesh(subdivide_body_path)
    mesh2 = o3d.io.read_triangle_mesh(remesh_cloth_path)
    merged_mesh, face_labels, v, f = merge_meshes_with_face_labels(mesh1, mesh2)
    np.savez(merge_npz_path, v=v, f=f, face_labels=face_labels)

    f_body = f[face_labels==0]
    f_cloth = f[face_labels==1]
    body_index = np.unique(f_body)
    wt_cloth_v, wt_cloth_f, wt_cloth_n= om_loadmesh_normal_numpy(save_cloth_wt_path)
    cloth_wt_fn = pysdf.SDF(wt_cloth_v, wt_cloth_f)
    inside_body_index, outside_body_index = find_inside_point(v, body_index, cloth_wt_fn, root)
    inside_body_index_path = os.path.join(root, "inside_body_index.npz")
    np.savez(inside_body_index_path, inside_body_index=inside_body_index, outside_body_index=outside_body_index)
