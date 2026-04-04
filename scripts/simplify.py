"""
Compare Decimation Methods
--------------------------

This example compares various decimation methods

"""

import time
import numpy as np

from numpy.matlib import rand
import pyvista as pv
from pyvista import examples

from pyamg import smoothed_aggregation_solver
from robust_laplacian import mesh_laplacian
from gravomg import MultigridSolver, csr_matrix
from gravomg import util as gravomg_util
from gravomg.util import neighbors_from_stiffness, normalize_area
import fast_simplification

import open3d as o3d
from open3d.visualization import draw
# load an example mesh
mesh = examples.download_louis_louvre()

print(f"Number of points: {mesh.n_points}")

# subdivide the mesh
mesh = mesh.subdivide(2)
print(f"Number of points after subdivision: {mesh.n_points}")
mesh = mesh.triangulate()

vertices = np.array(mesh.points)
print(f"Number of vertices: {vertices.shape}")

# mesh.plot()

faces = mesh.faces.reshape((-1, 4))[:, 1:]

print(f"Number of faces: {faces.shape}")

trimesh = o3d.geometry.TriangleMesh()
trimesh.vertices = o3d.utility.Vector3dVector(vertices)
trimesh.triangles = o3d.utility.Vector3iVector(faces)
trimesh.compute_vertex_normals()
L, M = mesh_laplacian(verts=vertices, faces=faces)

A = csr_matrix(L + 1e-5 * M)
rhs = M @ (np.ones(vertices.shape[0]) + np.random.randn(vertices.shape[0]) * 0.1)
backup_rhs = rhs.copy()

solver = smoothed_aggregation_solver(A)
begin = time.time()
x0 = np.zeros(vertices.shape[0])
solution = solver.solve(rhs, x0=x0)
end = time.time()
print(f"Time taken to solve the system: {end - begin:.4f} seconds")


# neigh = neighbors_from_stiffness(L)
# V = normalize_area(vertices, faces)
# print("creating multigrid solver")
# solver = MultigridSolver(V, neigh, M)
# lhs = csr_matrix(A)
# rhs = backup_rhs.copy()
# begin = time.time()
# print("Solving the system using Multigrid Solver")
# solution = solver.solve(lhs, rhs)
# end = time.time()
# print(f"Time taken to solve the system: {end - begin:.4f} seconds")


# # nice camera angle
# cpos = [
#     (6.264157141857314, -6.959267635766402, 11.71668951132694),
#     (1.3291685457683413, 2.267162128740896, 12.263240938610595),
#     (0.0023825740958850136, -0.05786378450796799, 0.9983216444528751),
# ]


# ###############################################################################
# # Compare decimation times
# reduction = 0.9
# print("Approach                         Time Elapsed")

# tstart = time.time()
# fas_sim = fast_simplification.simplify_mesh(mesh, target_reduction=reduction)
# fast_sim_time = time.time() - tstart
# print(f"Fast Quadratic Simplification  {fast_sim_time:8.4f} seconds")

# tstart = time.time()
# dec_std = mesh.decimate(reduction)
# dec_std_time = time.time() - tstart
# print(f"vtkQuadricDecimation           {dec_std_time:8.4f} seconds")

# tstart = time.time()
# dec_pro = mesh.decimate_pro(reduction)
# dec_pro_time = time.time() - tstart
# print(f"vtkDecimatePro                 {dec_pro_time:8.4f} seconds")


# pl = pv.Plotter(shape=(2, 2), window_size=(1000, 1000), theme=pv.themes.DocumentTheme())
# pl.add_text("Original", "upper_right", color="k")
# pl.add_mesh(mesh, show_edges=True)
# pl.camera_position = cpos

# pl.subplot(0, 1)
# pl.add_text(
#     f"Fast-Quadric-Mesh-Simplification\n{fast_sim_time:8.4f} seconds",
#     "upper_right",
#     color="k",
# )
# pl.add_mesh(fas_sim, show_edges=True)
# pl.camera_position = cpos

# pl.subplot(1, 0)
# pl.add_mesh(dec_std, show_edges=True)
# pl.add_text(f"vtkQuadricDecimation\n{dec_std_time:8.4f} seconds", "upper_right", color="k")
# pl.camera_position = cpos

# pl.subplot(1, 1)
# pl.add_mesh(dec_pro, show_edges=True)
# pl.add_text(f"vtkDecimatePro\n{dec_pro_time:8.4f} seconds", "upper_right", color="k")
# pl.camera_position = cpos

# pl.show()
