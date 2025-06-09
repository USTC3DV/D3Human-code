

import pyvista as pv
import pytetwild

import numpy as np
import tetgen

def get_tet_mesh(mesh_path, save_npz_path):

    surface_mesh = pv.read(mesh_path)
    tet = tetgen.TetGen(surface_mesh)
    tet.make_manifold(verbose=True)
    tet_grid_volume=6e-3
    vertices, indices = tet.tetrahedralize( fixedvolume=1, 
                                        maxvolume=tet_grid_volume, 
                                        regionattrib=1, 
                                        nobisect=False, steinerleft=-1, order=1, metric=1, meditview=1, nonodewritten=0, verbose=2)
        # shell = tet.grid.extract_surface()
    # tet_path = "tet_smpl2.obj"
    # vertices = vertices.to(np.float64)
    vertices = vertices.astype(np.float32)
    tet_path = save_npz_path.replace("npz", "obj")
    save_tet_mesh_as_obj(vertices, indices, tet_path)
    np.savez(save_npz_path, v=vertices, f=indices)

    return vertices, indices


def get_tet_mesh_test(mesh_path, save_npz_path):

    surface_mesh = pv.read(mesh_path)
    tetrahedral_mesh = pytetwild.tetrahedralize_pv(surface_mesh, edge_length_fac=0.1)
    tetrahedral_mesh.explode(1).plot(show_edges=True)

    v = tetrahedral_mesh.points
    f = tetrahedral_mesh.cells.reshape(-1, 5)[:, 1:]

    np.savez(save_npz_path, v=v, f=f)

    return v, f
    

def save_tet_mesh_as_obj(vertices, tetrahedra, filename):
    with open(filename, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        if tetrahedra is not None:
            for tet in tetrahedra:
                f.write(f"f {tet[0]+1} {tet[1]+1} {tet[2]+1} {tet[3]+1}\n")

