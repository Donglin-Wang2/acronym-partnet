import os
import csv
import json
import pickle
from cv2 import threshold

import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.spatial.transform.rotation import Rotation as R

# Root & Metadata paths
ACRONYM_MODEL_PATH = os.path.join('/', 'home', 'donglin',
                                  'Data', 'acronym', 'grasps')
PARTNET_META_PATH = os.path.join(
    '/', 'home', 'donglin', 'Github', 'partnet_dataset', 'stats', 'all_valid_anno_info.txt')
SHAPENET_SEM_META_PATH = os.path.join(
    '/', 'home', 'donglin', 'Data', 'ShapeNetSem.v0', 'metadata.csv')
SHAPENETV2_ROOT_PATH = os.path.join('/', 'home', 'donglin',
                                    'Data', 'ShapeNetCore.v2')
PARTNET_ROOT_PATH = os.path.join('/', 'home', 'donglin',
                                 'Data', 'data_v0')
SHAPENETV1_TRANSMAT_META_PATH = os.path.join(
    '/', 'home', 'donglin', 'Data', 'data_v0', 'chair_table_storagefurniture_bed_shapenetv1_to_partnet_alignment.json')


# Index file paths
PICKLE_EXT = '.pickle'
MAPPING_DIR = './mappings'
SHAPENET_IDS_PATH = os.path.join(MAPPING_DIR, 'shapenet_ids' + PICKLE_EXT)
ANNO_IDS_PATH = os.path.join(MAPPING_DIR, 'anno_ids' + PICKLE_EXT)
SHAPENET_TO_CAT_NAME_PATH = os.path.join(
    MAPPING_DIR, 'shapenet_to_cat_name' + PICKLE_EXT)
SHAPENET_TO_CAT_ID_PATH = os.path.join(
    MAPPING_DIR, 'shapenet_to_cat_id' + PICKLE_EXT)
SHAPENET_TO_ANNO_PATH = os.path.join(
    MAPPING_DIR, 'shapenet_to_anno' + PICKLE_EXT)
ANNO_TO_SHAPENET_PATH = os.path.join(
    MAPPING_DIR, 'anno_to_shapenet' + PICKLE_EXT)
SEM_TO_SHAPENET_SCALE_PATH = os.path.join(
    MAPPING_DIR, 'sem_to_shapenet_scale' + PICKLE_EXT)
SHAPENET_TO_ACRONYM_SCALE_PATH = os.path.join(
    MAPPING_DIR, 'shapenet_to_acronym_scale' + PICKLE_EXT)
SHAPENET_TO_SEM_STAT_PATH = os.path.join(
    MAPPING_DIR, 'shapenet_to_sem_stat' + PICKLE_EXT)
SHAPENET_TO_V2_STAT_PATH = os.path.join(
    MAPPING_DIR, 'shapenet_to_v2_stat' + PICKLE_EXT)
SHAPENETV1_TO_TRANSMAT_PATH = os.path.join(
    MAPPING_DIR, 'shapenetv1_to_transmat' + PICKLE_EXT)
SHAPENET_TO_ACRONYM_PATH = os.path.join(
    MAPPING_DIR, 'shapenet_to_acronym' + PICKLE_EXT)


# Path templates
SHAPENETV1_MODEL_PATH_TEMPLATE = '/home/donglin/Data/ShapeNetCore.v1/{shapenet_cat_id}/{shapenet_id}/model.obj'
SHAPENETV2_MODEL_PATH_TEMPLATE = '/home/donglin/Data/ShapeNetCore.v2/{shapenet_cat_id}/{shapenet_id}/models/model_normalized.obj'
SHAPENETV2_META_PATH_TEMPLATE = '/home/donglin/Data/ShapeNetCore.v2/{shapenet_cat_id}/{shapenet_id}/models/model_normalized.json'
PARTNET_POINTCLOUD_PATH_TEMAPLTE = '/home/donglin/Data/data_v0/{anno_id}/point_sample/sample-points-all-pts-label-10000.ply'
SHAPENETSEM_MODEL_PATH_TEMPLATE = '/home/donglin/Data/ShapeNetSem.v0/models-OBJ/models/{shapenet_id}.obj'

shapenet_ids = set()
anno_ids = set()
shapenet_to_cat_name = {}
shapenet_to_cat_id = {}
shapenet_to_anno = {}
anno_to_shapenet = {}
sem_to_shapenet_scale = {}
shapenet_to_acronym_scale = {}
joint_category_count = {}
shapenet_to_sem_stat = {}
shapenet_to_v2_stat = {}
shapenetv1_to_transmat = {}


def produce_shapenet_to_acronym_scale_map():
    for filename in os.listdir(ACRONYM_MODEL_PATH):
        tokens = filename.split("_")
        id = tokens[1]
        if len(filename) >= 28:
            shapenet_ids.add(id)
            shapenet_to_acronym_scale[id] = float(tokens[-1][:-3])
    pickle.dump(shapenet_to_acronym_scale, open(
        SHAPENET_TO_ACRONYM_SCALE_PATH, 'wb'))


def produce_partnet_map():
    with open(PARTNET_META_PATH, 'r') as file_obj:
        for line in file_obj.readlines():
            tokens = line.split(" ")
            shapenet_to_cat_name[tokens[3]] = tokens[2]
            anno_to_shapenet[tokens[0]] = tokens[3]
            shapenet_to_anno[tokens[3]] = tokens[0]
    pickle.dump(shapenet_to_cat_name, open(SHAPENET_TO_CAT_NAME_PATH, 'wb'))
    pickle.dump(anno_to_shapenet, open(ANNO_TO_SHAPENET_PATH, 'wb'))
    pickle.dump(shapenet_to_anno, open(SHAPENET_TO_ANNO_PATH, 'wb'))


def produce_id_to_cat_map():
    for cat_id in os.listdir(SHAPENETV2_ROOT_PATH):
        if not os.path.isdir(os.path.join(SHAPENETV2_ROOT_PATH, cat_id)):
            continue
        for model_id in os.listdir(os.path.join(SHAPENETV2_ROOT_PATH, cat_id)):
            shapenet_to_cat_id[model_id] = cat_id
    pickle.dump(shapenet_to_cat_id, open(SHAPENET_TO_CAT_ID_PATH, 'wb'))


def produce_shapenetsem_stat():
    with open(SHAPENET_SEM_META_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            shapenet_id = row['fullId'].split('.')[1]
            if row['up']:
                row['up'] = [float(ele) for ele in row['up'].split('\,')]
            if row['front']:
                row['front'] = [float(ele) for ele in row['front'].split('\,')]
            if row['aligned.dims']:
                row['aligned.dims'] = [
                    float(ele) for ele in row['aligned.dims'].split('\,')]
            if row['unit']:
                row['unit'] = float(row['unit'])
            shapenet_to_sem_stat[shapenet_id] = row
    pickle.dump(shapenet_to_sem_stat, open(SHAPENET_TO_SEM_STAT_PATH, 'wb'))


def produce_shapenetv2_stat():
    for cat_id in os.listdir(SHAPENETV2_ROOT_PATH):
        if not os.path.isdir(os.path.join(SHAPENETV2_ROOT_PATH, cat_id)):
            continue
        for model_id in os.listdir(os.path.join(SHAPENETV2_ROOT_PATH, cat_id)):
            json_path = SHAPENETV2_META_PATH_TEMPLATE.format(
                shapenet_cat_id=cat_id, shapenet_id=model_id)
            if not os.path.isfile(json_path):
                continue
            with open(json_path, 'r') as f:
                shapenet_to_v2_stat[model_id] = json.load(f)

    pickle.dump(shapenet_to_v2_stat, open(SHAPENET_TO_V2_STAT_PATH, 'wb'))


def produce_acronym_index():
    id_to_path = {}
    for filename in os.listdir(ACRONYM_MODEL_PATH):
        category_name, shapenet_id, scale = os.path.splitext(filename)[
            0].split("_")
        id_to_path[shapenet_id] = os.path.join(ACRONYM_MODEL_PATH, filename)
    pickle.dump(id_to_path, open(SHAPENET_TO_ACRONYM_PATH, 'wb'))


def produce_anno_id_to_transmat():
    with open(SHAPENETV1_TRANSMAT_META_PATH, 'r') as fo:
        shapenetv1_to_transmat = json.load(fo)
    pickle.dump(shapenetv1_to_transmat, open(
        SHAPENETV1_TO_TRANSMAT_PATH, 'wb'))


def get_category_count():
    hits = 0
    for id in shapenet_ids:
        if id in shapenet_to_cat_name:
            hits += 1
            joint_category_count[shapenet_to_cat_name[id]] = joint_category_count.get(
                shapenet_to_cat_name[id], 0) + 1
    return joint_category_count


def get_acronym_grasp_path(shapenet_id):
    shapenet_to_acronym = pickle.load(open(SHAPENET_TO_ACRONYM_PATH, 'rb'))
    return shapenet_to_acronym[shapenet_id]

# DONE 
def get_shapenetv1_model_path(shapenet_id):
    shapenet_to_cat_id = pickle.load(open(SHAPENET_TO_CAT_ID_PATH, 'rb'))
    shapenet_cat_id = shapenet_to_cat_id[shapenet_id]
    return SHAPENETV1_MODEL_PATH_TEMPLATE.format(shapenet_cat_id=shapenet_cat_id, shapenet_id=shapenet_id)


def get_shapenetv2_model_path(shapenet_id):
    shapenet_to_cat_id = pickle.load(open(SHAPENET_TO_CAT_ID_PATH, 'rb'))
    shapenet_cat_id = shapenet_to_cat_id[shapenet_id]
    return SHAPENETV2_MODEL_PATH_TEMPLATE.format(shapenet_cat_id=shapenet_cat_id, shapenet_id=shapenet_id)


def get_partnet_pointcloud_path(shapenet_id):
    shapenet_to_anno = pickle.load(open(SHAPENET_TO_ANNO_PATH, 'rb'))
    anno_id = shapenet_to_anno[shapenet_id]
    return PARTNET_POINTCLOUD_PATH_TEMAPLTE.format(anno_id=anno_id)


def get_shapenetv1_stat(shapenet_id):
    shapenet_to_sem_stat = pickle.load(open(SHAPENET_TO_SEM_STAT_PATH, 'rb'))
    return shapenet_to_sem_stat[shapenet_id]


def get_shapenetv2_stat(shapenet_id):
    shapenet_to_v2_stat = pickle.load(open(SHAPENET_TO_V2_STAT_PATH, 'rb'))
    return shapenet_to_v2_stat[shapenet_id]


def get_partnet_pointcloud(shapenet_id):
    pts = []
    pts_path = get_partnet_pointcloud_path(shapenet_id)
    with open(pts_path, 'r') as pts_f:
        tokens = pts_f.readlines()
        for token in tokens[10:]:
            pts.append([float(ele) for ele in token.split()[:3]])
    return np.array(pts)


def get_normalized_partnet_pointcloud(shapenet_id):
    pts = get_partnet_pointcloud(shapenet_id)
    return normalize_pointcloud(pts)


def get_normalized_partnet_pointcloud2(shapenet_id):
    # EXPERIMENTAL
    pts = get_partnet_pointcloud(shapenet_id)
    return normalize_pts(pts)


def get_anno_id(shapenet_id):
    mapping = pickle.load(open(SHAPENET_TO_ANNO_PATH, 'rb'))
    return mapping[shapenet_id]


def get_shapenetv1_transmat(shapenet_id):
    anno_id = pickle.load(open(SHAPENET_TO_ANNO_PATH, 'rb'))[shapenet_id]
    shapenetv1_to_transmat = pickle.load(
        open(SHAPENETV1_TO_TRANSMAT_PATH, 'rb'))
    print(shapenetv1_to_transmat[anno_id])
    transmat = np.reshape(shapenetv1_to_transmat[anno_id]['transmat'], (4, 4))
    return transmat


def get_min_max_center(pcd):
    minVertex = np.array([np.Infinity, np.Infinity, np.Infinity])
    maxVertex = np.array([-np.Infinity, -np.Infinity, -np.Infinity])
    aggVertices = np.zeros(3)
    numVertices = 0
    for pnt in pcd:
        aggVertices += pnt
        numVertices += 1
        minVertex = np.minimum(pnt, minVertex)
        maxVertex = np.maximum(pnt, maxVertex)
    centroid = aggVertices / numVertices
    info = {}
    info['min'] = minVertex
    info['max'] = maxVertex
    info['centroid'] = centroid
    return info


def normalize_pointcloud(pcd):
    result = []
    stats = get_min_max_center(pcd)
    diag = np.array(stats['max']) - np.array(stats['min'])
    norm = 1 / np.linalg.norm(diag)
    c = stats['centroid']
    for v in pcd:
        v_new = (v - c) * norm
        result.append(v_new)
    return np.stack(result)


def normalize_pts(pts):
    # EXPERIMENTAL
    out = np.array(pts, dtype=np.float32)
    center = np.mean(out, axis=0)
    out -= center
    scale = np.sqrt(np.max(np.sum(out**2, axis=1)))
    out /= scale
    return out


def produce_normalized_shapenetv1(shapenet_id):
    # EXPERIMENTAL
    out_path = f'./temp_meshes/{shapenet_id}_v1.obj'
    transmat = get_shapenetv1_transmat(shapenet_id)
    rotation = R.from_matrix(transmat[:3, :3])
    translation = transmat[:3, 3]
    model_path = get_shapenetv1_model_path(shapenet_id)
    with open(model_path, 'r') as f, open(out_path, 'w') as fo:
        for line in f:
            if line.startswith('v '):
                v = np.fromstring(line[2:], sep=' ')
                v = rotation.apply(v) + translation
                v_str = 'v %f %f %f\n' % (v[0], v[1], v[2])
                fo.write(v_str)
            else:
                fo.write(line)


def load_obj(fn):
    # EXPERIMENTAL CITATION
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0]
                         for item in line.split()[1:4]]))
    return np.vstack(vertices), np.vstack(faces)


def get_partnet_combined_vertices_and_faces(shapenet_id):
    # EXPERIMENTAL CITATION
    anno_id = get_anno_id(shapenet_id)
    obj_dir = os.path.join(PARTNET_ROOT_PATH, anno_id, 'objs')
    vs = []
    fs = []
    vid = 0
    for item in os.listdir(obj_dir):
        if item.endswith('.obj'):
            cur_vs, cur_fs = load_obj(os.path.join(obj_dir, item))
            vs.append(cur_vs)
            fs.append(cur_fs + vid)
            vid += cur_vs.shape[0]
    v_arr = np.concatenate(vs, axis=0)
    f_arr = np.concatenate(fs, axis=0) - 1
    tmp = np.array(v_arr[:, 0], dtype=np.float32)
    # v_arr[:, 0] = v_arr[:, 2]
    # v_arr[:, 2] = -tmp
    return v_arr, f_arr


def get_transmat_from_vertices(vertices):
    # EXPERIMENTAL CITATION
    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])
    x_center = (x_min + x_max) / 2
    x_len = x_max - x_min
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1])
    y_center = (y_min + y_max) / 2
    y_len = y_max - y_min
    z_min = np.min(vertices[:, 2])
    z_max = np.max(vertices[:, 2])
    z_center = (z_min + z_max) / 2
    z_len = z_max - z_min
    scale = np.sqrt(x_len**2 + y_len**2 + z_len**2)
    trans = np.array([[0, 0, 1.0/scale, -x_center/scale],
                      [0, 1.0/scale, 0, -y_center/scale],
                      [-1/scale, 0, 0, -z_center/scale],
                      [0, 0, 0, 1]], dtype=np.float32)
    return trans


def get_mesh_from_verices_and_faces(vertices, faces):
    # EXPERIMENTAL
    vertices, faces = o3d.utility.Vector3dVector(
        vertices), o3d.utility.Vector3iVector(faces)
    return o3d.geometry.TriangleMesh(vertices, faces)


def get_normalized_partnet_mesh(shapenet_id):
    # EXPERIMENTAL
    vertices, faces = get_partnet_combined_vertices_and_faces(shapenet_id)
    transmat = get_transmat_from_vertices(vertices)
    vertices = np.concatenate(
        [vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
    vertices = vertices @ transmat.T
    vertices = vertices[:, :3]
    return get_mesh_from_verices_and_faces(vertices, faces)


def get_normalized_partnet_mesh2(shapenet_id):
    # EXPERIMENTAL
    vertices, faces = get_partnet_combined_vertices_and_faces(shapenet_id)
    vertices = normalize_pointcloud(vertices)
    return get_mesh_from_verices_and_faces(vertices, faces)


def get_chamf_dist(mesh1, mesh2):
    # EXPERIMENTAL
    mesh1_pts = np.asarray(mesh1.sample_points_uniformly(
        number_of_points=2000).points)
    mesh2_pts = np.asarray(mesh2.sample_points_uniformly(
        number_of_points=2000).points)
    dist_mat = cdist(mesh1_pts, mesh2_pts)
    chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()
    return chamfer_dist


def get_partnet_icp1(shapenet_id):
    # EXPERIMENTAL
    partnet_mesh = get_normalized_partnet_mesh(shapenet_id)
    shapenetv1_mesh_path = get_shapenetv1_model_path(shapenet_id)
    shapenetv1_mesh = o3d.io.read_triangle_mesh(shapenetv1_mesh_path)
    partnet_vts = o3d.geometry.PointCloud(partnet_mesh.vertices)
    shapenetv1_vts = o3d.geometry.PointCloud(shapenetv1_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.005
    partnet_to_shapenetv1 = o3d.pipelines.registration.registration_icp(
        partnet_vts, shapenetv1_vts, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200000))
    print(partnet_to_shapenetv1)
    print(partnet_to_shapenetv1.transformation)
    return partnet_mesh.transform(partnet_to_shapenetv1.transformation)


def get_partnet_icp2(shapenet_id):
    # EXPERIMENTAL
    partnet_mesh = get_normalized_partnet_mesh(shapenet_id)
    partnet_mesh = partnet_mesh.merge_close_vertices(
        1e-4).remove_duplicated_triangles().remove_non_manifold_edges().remove_degenerate_triangles().remove_duplicated_triangles()
    print("Is the partnet mesh watertight?: ", partnet_mesh.is_watertight())
    shapenetv1_mesh_path = get_shapenetv2_model_path(shapenet_id)
    shapenetv1_mesh = o3d.io.read_triangle_mesh(shapenetv1_mesh_path)
    partnet_vts = partnet_mesh.sample_points_uniformly(number_of_points=200000)
    shapenetv1_vts = shapenetv1_mesh.sample_points_uniformly(
        number_of_points=200000, use_triangle_normal=True)
    trans_init = np.eye(4)
    trans_init = so3_to_se3(R.from_rotvec(
        [0, 90, 0], degrees=True).as_matrix())
    threshold = 1
    loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
    partnet_to_shapenetv1 = o3d.pipelines.registration.registration_icp(
        partnet_vts, shapenetv1_vts, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200000))
    print(partnet_to_shapenetv1)
    print(partnet_to_shapenetv1.transformation)
    return partnet_mesh.transform(partnet_to_shapenetv1.transformation)


def so3_to_se3(so3):
    se3 = np.concatenate(
        [so3, np.zeros((so3.shape[0], 1), dtype=np.float32)], axis=1)
    se3 = np.concatenate(
        [se3, np.array([0, 0, 0, 1], dtype=np.float32, ndmin=2)]
    )
    return se3


if __name__ == '__main__':
    produce_shapenet_to_acronym_scale_map()
    #DONE
    produce_partnet_map()
    #DONE
    produce_id_to_cat_map()
    produce_shapenetsem_stat()
    produce_shapenetv2_stat()
    produce_anno_id_to_transmat()
