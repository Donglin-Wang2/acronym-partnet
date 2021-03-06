from math import degrees
from utils import *
from scipy.spatial.transform.rotation import Rotation as R
import open3d as o3d
SHAPENET_ID = '6cb6373befd7e1a7aa918b099faddaba'


def visualize(shapenet_id, mesh='v2', apply_rot=True):
    if apply_rot:
        rotation = R.from_rotvec([0, 180, 0], degrees=True)
        pts = rotation.apply(get_normalized_partnet_pointcloud(shapenet_id))
    else:
        pts = get_normalized_partnet_mesh(shapenet_id)
    pts = o3d.utility.Vector3dVector(pts)
    pcd = o3d.geometry.PointCloud(pts)
    if mesh == 'v2':
        mesh = o3d.io.read_triangle_mesh(
            get_shapenetv2_model_path(shapenet_id))
    else:
        mesh = o3d.io.read_triangle_mesh(
            get_shapenetv1_model_path(shapenet_id))
    o3d.visualization.draw_geometries([pcd, mesh])


def get_total_distance(shapenet_id, apply_rot=True):
    mesh = o3d.io.read_triangle_mesh(get_shapenetv2_model_path(shapenet_id))
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    if apply_rot:
        rotation = R.from_rotvec([0, 180, 0], degrees=True)
        pts = rotation.apply(get_normalized_partnet_pointcloud(shapenet_id))
    else:
        pts = get_normalized_partnet_pointcloud(shapenet_id)
    print(pts.shape)
    pts = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    scene.add_triangles(mesh)
    return scene.compute_distance(pts)


def test_dist_calc(shapenet_id):
    mesh = o3d.io.read_triangle_mesh(get_shapenetv2_model_path(shapenet_id))
    tri_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    pcd = mesh.sample_points_uniformly(number_of_points=500)
    pcd = o3d.core.Tensor(np.array(pcd.points), dtype=o3d.core.Dtype.Float32)
    scene.add_triangles(tri_mesh)
    print(scene.compute_distance(pcd).numpy().mean())


def test_pos_align(shapenet_id):
    transmat = get_shapenetv1_transmat(shapenet_id)
    translation = transmat[:3, 3]
    pts = get_normalized_partnet_pointcloud(shapenet_id)
    rotation = R.from_matrix(transmat[:3, :3])
    pts = rotation.apply(pts)
    pts += translation
    pts = o3d.utility.Vector3dVector(pts)
    pcd = o3d.geometry.PointCloud(pts)
    mesh = o3d.io.read_triangle_mesh(get_shapenetv1_model_path(shapenet_id))
    o3d.visualization.draw_geometries([pcd, mesh])


def test_align(shapenet_id):
    pts = get_normalized_partnet_pointcloud(shapenet_id)
    pts = o3d.utility.Vector3dVector(pts)
    pcd = o3d.geometry.PointCloud(pts)
    mesh = o3d.io.read_triangle_mesh(f'./temp_meshes/{shapenet_id}_v1.obj')
    o3d.visualization.draw_geometries([pcd, mesh])


def visualize_partnet_comb_and_v1(shapenet_id):
    partnet_mesh = get_normalized_partnet_mesh(shapenet_id)
    shapenetv1_mesh_path = get_shapenetv1_model_path(shapenet_id)
    shapenetv1_mesh = o3d.io.read_triangle_mesh(shapenetv1_mesh_path)
    print(get_chamf_dist(partnet_mesh, shapenetv1_mesh))
    o3d.visualization.draw_geometries([partnet_mesh, shapenetv1_mesh])


def visualize_icp(shapenet_id):
    partnet_mesh = get_partnet_icp2(shapenet_id)
    partnet_mesh.paint_uniform_color([0, 0, 1])
    shapenetv1_mesh_path = get_shapenetv2_model_path(shapenet_id)
    shapenetv1_mesh = o3d.io.read_triangle_mesh(shapenetv1_mesh_path)
    o3d.visualization.draw_geometries([partnet_mesh, shapenetv1_mesh])


def test_icp_robustness(shapenet_id):
    scene = o3d.t.geometry.RaycastingScene()
    partnet_mesh = get_partnet_icp2(shapenet_id)
    shapenetv1_mesh_path = get_shapenetv2_model_path(shapenet_id)
    shapenetv1_mesh = o3d.io.read_triangle_mesh(shapenetv1_mesh_path)
    shapenetv1_mesh = o3d.t.geometry.TriangleMesh.from_legacy(shapenetv1_mesh)
    pts = partnet_mesh.sample_points_uniformly(number_of_points=200000)
    pts = o3d.core.Tensor(np.array(pts.points), dtype=o3d.core.Dtype.Float32)
    scene.add_triangles(shapenetv1_mesh)
    print(np.abs(scene.compute_distance(pts).numpy()).max())


def visualize2(shapenet_id):
    partnet_mesh = get_normalized_partnet_mesh2(shapenet_id)
    shapenetv2_mesh = o3d.io.read_triangle_mesh(
        get_shapenetv2_model_path(shapenet_id))
    rotation = R.from_rotvec([0, -180, 0], degrees=True).as_matrix()
    rotation = so3_to_se3(rotation)
    partnet_mesh.transform(rotation)
    partnet_mesh.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([partnet_mesh, shapenetv2_mesh])

def test_shapenetsem_visual():
    mesh1 = get_normalized_shapenetsem_mesh(SHAPENET_ID)
    o3d.visualization.draw_geometries(
        [mesh1,  o3d.geometry.TriangleMesh.create_coordinate_frame()])

def test_shapenetsem_partnet_alignment_vts():
    # id2 = '6cb3d99b20e7fbb5b04cb542e2c50eb4'
    # id2 = '6bd7475753c3d1d62764cfba57a5de73'
    # id2 = '665bfb42a0362f71d577f4b88a77dd38'
    sem_mesh = get_normalized_shapenetsem_mesh(SHAPENET_ID)
    partnet_mesh = get_normalized_partnet_mesh(SHAPENET_ID).paint_uniform_color([0, 0, 1])
    partnet_to_acro_rot = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()
    transmat1 = get_shapenetsem_axis_alignment(SHAPENET_ID)
    sem_mesh.rotate(transmat1)
    partnet_mesh.rotate(partnet_to_acro_rot)
    sem_to_partnet = get_icp_between(sem_mesh, partnet_mesh)
    sem_mesh.transform(sem_to_partnet)
    calc_max_distance(sem_mesh, partnet_mesh)
    # o3d.visualization.draw_geometries(
    #     [sem_mesh, partnet_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame()])

def test_shapenetsem_partnet_alignment_rand_pnts():
    sem_mesh = get_normalized_shapenetsem_mesh(SHAPENET_ID)
    partnet_mesh = get_normalized_partnet_mesh(SHAPENET_ID).paint_uniform_color([0, 0, 1])
    partnet_to_acro_rot = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()
    transmat1 = get_shapenetsem_axis_alignment(SHAPENET_ID)
    sem_mesh.rotate(transmat1)
    partnet_mesh.rotate(partnet_to_acro_rot)
    sem_to_partnet = get_icp_between_rand_pnts(sem_mesh, partnet_mesh)
    sem_mesh.transform(sem_to_partnet)
    calc_max_distance(sem_mesh, partnet_mesh)

def calc_max_distance(source_mesh, target_mesh):
    scene = o3d.t.geometry.RaycastingScene()
    target_mesh = o3d.t.geometry.TriangleMesh.from_legacy(target_mesh)
    source_points = source_mesh.sample_points_uniformly(20000)
    source_points = o3d.core.Tensor(np.array(source_points.points), dtype=o3d.core.Dtype.Float32)
    scene.add_triangles(target_mesh)
    print(np.abs(scene.compute_distance(source_points).numpy()).max())
    

# test_shapenetsem_visual()
test_shapenetsem_partnet_alignment_vts()
test_shapenetsem_partnet_alignment_rand_pnts()
# test_icp_robustness(SHAPENET_ID)
# test_dist_calc(SHAPENET_ID)
# visualize_icp(SHAPENET_ID)
# visualize_partnet_comb_and_v1(SHAPENET_ID)
# test_align(SHAPENET_ID)
# test_pos_align(SHAPENET_ID)
# visualize2(SHAPENET_ID)
# print(sum(get_total_distance(SHAPENET_ID)))
