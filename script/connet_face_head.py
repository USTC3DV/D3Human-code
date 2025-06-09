import numpy as np
import openmesh as om
import open3d as o3d
import os



def write_pc(path_mesh, v, vn=None, f=None):
    # print("---------v:", v.shape)
    assert v.ndim == 2 and v.shape[1] == 3
    with open(path_mesh, 'w') as fp:
        fp.write(('v {:f} {:f} {:f}\n' * v.shape[0]).format(*v.reshape(-1)))
        if vn is not None:
            fp.write(('vn {:f} {:f} {:f}\n' * vn.shape[0]).format(*vn.reshape(-1)))
        if f is not None:
            fp.write(('f {:d} {:d} {:d}\n' * f.shape[0]).format(*f.reshape(-1) + 1))


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

def find_connected_components_with_faces(vertices, faces):
    uf = UnionFind(len(vertices))
    face_map = {}

    for face in faces:
        root_initial = uf.find(face[0])
        for i in range(1, len(face)):
            uf.union(face[0], face[i])
        
        root_updated = uf.find(face[0])
        if root_updated not in face_map:
            face_map[root_updated] = []
        face_map[root_updated].append(face.tolist())

    component_map = {}
    for vertex in range(len(vertices)):
        root = uf.find(vertex)
        if root not in component_map:
            component_map[root] = []
        component_map[root].append(vertex)
    
    components_with_faces = [(component_map[root], face_map.get(root, [])) for root in component_map]
    return components_with_faces



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

def om_loadmesh(path):
    mesh = om.read_trimesh(path)
    v = np.array(mesh.points())
    f = np.array(mesh.face_vertex_indices())

    return v, f

def filter_and_sort_components(components):
    filtered_components = [comp for comp in components if len(comp) > 1]
    sorted_components = sorted(filtered_components, key=len, reverse=True)
    return sorted_components

def write_ply(path_mesh, points, normals=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(path_mesh, pcd)


def merge_and_repair_meshes(mesh1_v, mesh1_f, mesh2_v, mesh2_f):

    vertices = np.vstack((np.asarray(mesh1_v), np.asarray(mesh2_v)))
    print("mesh1_f:", mesh1_f.shape)
    print("mesh2_f:", mesh2_f.shape)
    print("mesh1_v:", mesh1_v.shape)
    print("mesh2_v:", mesh2_v.shape)
    triangles = np.vstack((np.asarray(mesh1_f), np.asarray(mesh2_f) + len(mesh1_v)))
    merged_mesh = o3d.geometry.TriangleMesh()
    merged_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    merged_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    merged_mesh = merged_mesh.remove_duplicated_vertices()
    merged_mesh = merged_mesh.remove_degenerate_triangles()

    return merged_mesh

def create_bbox_mesh(bbox_min, bbox_max):

    vertices = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
    ])

    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4]
    ])

    return vertices, faces

def body_head_box(body_components, body_v):
    body_components_len = len(body_components)
    mean_point_list = []
    for i in range(body_components_len):
        body_sub_components = body_components[i]
        select_v = body_v[body_sub_components]
        mean_point = np.mean(select_v, axis=0)
        mean_point_list.append(mean_point)
    y_values = [point[1] for point in mean_point_list]
    max_y_index = y_values.index(max(y_values))
    max_y_components = body_v[body_components[max_y_index]]
    bbox_min = np.min(max_y_components, axis=0) - 0.01
    bbox_max = np.max(max_y_components, axis=0) + 0.01

    return bbox_min, bbox_max


def process_close_hole(FLAGS, root, body_mesh, cloth_mesh, num=5):

    body_v, body_f = om_loadmesh(body_mesh)
    cloth_v, cloth_f = om_loadmesh(cloth_mesh)

    body_connected_components = find_connected_components(body_v, body_f)
    body_sorted_components = filter_and_sort_components(body_connected_components)

    body_components_frombody = []
    cloth_components_frombody = []

    body_components_fromcloth = []
    cloth_components_fromcloth = []

    for i, comp in enumerate(body_sorted_components):
        if i < num:
            body_components_frombody.append(comp)
        else:
            cloth_components_frombody.append(comp)


    cloth_connected_components = find_connected_components(cloth_v, cloth_f)
    cloth_sorted_components = filter_and_sort_components(cloth_connected_components)        

    for i, comp in enumerate(cloth_sorted_components):
        if i < 1:
            cloth_components_fromcloth.append(comp)
        else:
            body_components_fromcloth.append(comp)

    print("body_components_frombody:", len(body_components_frombody))
    bbox_min, bbox_max = body_head_box(body_components_frombody, body_v)
    bbox_path = os.path.join(root, "bbox.npz")
    np.savez(bbox_path, bbox_min=bbox_min, bbox_max=bbox_max)

    body_frombody_faces = []
    cloth_frombody_faces = []
    body_fromcloth_faces = []
    cloth_fromcloth_faces = []

    for idx, i in enumerate(body_components_frombody):
        vertices_set = set(i)
        for face_i in body_f:
            if set(face_i).issubset(vertices_set):
                body_frombody_faces.append(face_i)
        
    for idx, i in enumerate(cloth_components_frombody):
        vertices_set = set(i)
        for face_i in body_f:
            if set(face_i).issubset(vertices_set):
                cloth_frombody_faces.append(face_i)

    for idx, i in enumerate(body_components_fromcloth):
        vertices_set = set(i)
        for face_i in cloth_f:
            if set(face_i).issubset(vertices_set):
                body_fromcloth_faces.append(face_i)

    for idx, i in enumerate(cloth_components_fromcloth):
        vertices_set = set(i)
        for face_i in cloth_f:
            if set(face_i).issubset(vertices_set):
                cloth_fromcloth_faces.append(face_i)


    body_frombody_faces = np.array(body_frombody_faces)
    path_body_frombody_faces = os.path.join(root, "body_frombody_faces.obj")
    write_pc(path_body_frombody_faces, v=body_v, f=body_frombody_faces)


    cloth_frombody_faces = np.array(cloth_frombody_faces)
    path_cloth_frombody_faces = os.path.join(root, "cloth_frombody_faces.obj")
    write_pc(path_cloth_frombody_faces, v=body_v, f=cloth_frombody_faces)

    body_fromcloth_faces = np.array(body_fromcloth_faces)
    path_body_fromcloth_faces = os.path.join(root, "body_fromcloth_faces.obj")
    write_pc(path_body_fromcloth_faces, v=cloth_v, f=body_fromcloth_faces)

    cloth_fromcloth_faces = np.array(cloth_fromcloth_faces)
    path_cloth_fromcloth_faces = os.path.join(root, "cloth_fromcloth_faces.obj")
    write_pc(path_cloth_fromcloth_faces, v=cloth_v, f=cloth_fromcloth_faces)

    if len(cloth_frombody_faces) != 0 and len(cloth_fromcloth_faces) != 0:
        merge_cloth = merge_and_repair_meshes(body_v, cloth_frombody_faces, cloth_v, cloth_fromcloth_faces)
        cloth_concat_path = os.path.join(root, "cloth_concat.obj")
        o3d.io.write_triangle_mesh(cloth_concat_path, merge_cloth)
    else:
        cloth_concat_path = cloth_mesh

    if len(body_frombody_faces) != 0 and len(body_fromcloth_faces) != 0:
        merge_body = merge_and_repair_meshes(body_v, body_frombody_faces, cloth_v, body_fromcloth_faces)
        body_concat_path = os.path.join(root, "body_concat.obj")
        o3d.io.write_triangle_mesh(body_concat_path, merge_body)
    else:
        body_concat_path = path_body_frombody_faces


    return body_concat_path, cloth_concat_path, bbox_path


if __name__ == "__main__":

    FLAGS = None
    root = "connet_face_head_test"
    os.makedirs(root, exist_ok=True)
    body_mesh = "../res/0607_f3c/split_cloth0/split_body_imesh_800.obj"
    cloth_mesh = "../res/0607_f3c/split_cloth0/split_cloth_imesh_800.obj"
    body_concat_path, cloth_concat_path, bbox_path = process_close_hole(FLAGS, root, body_mesh, cloth_mesh)
    print("body_concat_path:", body_concat_path)
    print("cloth_concat_path:", cloth_concat_path)
