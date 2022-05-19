### TO CONSOLIDATE PARTNET AND SEM
# PartNet: 1. Normalize 2. R.from_rotvec([90, 0, 0], degrees=True).as_matrix()
# Sem: 1. Normalize 2. sem_transmat = get_shapenetsem_axis_alignment(SHAPENET_ID) 3. 


from utils import *


import json
import math
import time
from tqdm import tqdm

def combine_meshes(obj_list):
    vs = []
    fs = []
    vid = 0
    for item in obj_list:
        if item.endswith('.obj'):
            cur_vs, cur_fs = load_obj(os.path.join(item))
            vs.append(cur_vs)
            fs.append(cur_fs + vid)
            vid += cur_vs.shape[0]
    v_arr = np.concatenate(vs, axis=0)
    f_arr = np.concatenate(fs, axis=0)
    return get_mesh_from_verices_and_faces(v_arr, f_arr)

start = time.time()

NUM_SAMPLES = 2048
ANNO_ID = '39057'
SHAPENET_ID = '6cb6373befd7e1a7aa918b099faddaba'
points = []
pnt_label_list = []
shape_label_list = []
shape_part_count = []
shape_part_label_mapping = {}
sampled_count = 0
partnet_file = open('/home/donglin/Github/partnet_dataset/stats/all_valid_anno_info.txt', 'r')
for line in tqdm(partnet_file.readlines()):
    tokens = line.split(" ")
    shapenet_id = tokens[3]
    anno_id = tokens[0]
    # shapenet_id = SHAPENET_ID
    # anno_id = ANNO_ID
    cat_name = tokens[2]
    # if tokens[2] != CAT:
    #     continue
    hier = json.load(open(f'/home/donglin/Data/data_v0/{anno_id}/result_after_merging.json', 'r'))[0]
    total_surface_area = 0
    num_sampled = 0
    part_surface_areas = []
    part_meshes = []
    part_mesh_names = []
    sampled_points = []
    point_labels = []
    normalize_transmat = get_partnet_normalize_transmat(shapenet_id)
    rotation = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()
    # icp_mat = get_icp_between()
    if 'children' not in hier:
        continue
    if len(hier['children']) == 1:
        continue
    for child in hier['children']:
        file_list = [f'/home/donglin/Data/data_v0/{anno_id}/objs/{name}.obj' for name in child['objs']]
        part_mesh = combine_meshes(file_list).transform(normalize_transmat).transform(so3_to_se3(rotation)).paint_uniform_color([0,0,1])
        part_meshes.append(part_mesh)
        part_mesh_names.append(child['name'])
        if shape_part_label_mapping.get(cat_name, None):
            shape_part_label_mapping[cat_name].add(child['name'])
        else:
            shape_part_label_mapping[cat_name] = set([child['name']])
        part_surface_area = part_mesh.get_surface_area()
        total_surface_area += part_surface_area
        part_surface_areas.append(part_surface_area)
        
    for i, part_mesh in enumerate(part_meshes):
        
        if i == len(part_meshes) - 1:
            num_sample_part = NUM_SAMPLES - num_sampled
        else:
            num_sample_part = math.floor(NUM_SAMPLES * part_surface_areas[i] / total_surface_area)
    
        num_sampled += num_sample_part
        if num_sample_part <= 0:
            continue
        point_labels += [part_mesh_names[i]] * math.floor(num_sample_part)
        sampled_points.append(np.asarray(part_mesh.sample_points_uniformly(math.floor(num_sample_part)).points))
    
    sampled_count += 1

    points.append(np.concatenate(sampled_points, axis=0))
    pnt_label_list.append(point_labels)
    shape_label_list.append(cat_name)
    shape_part_count.append(len(set(point_labels)))

    if sampled_count % 5000 == 0:
        np.save(f'./data/points_{sampled_count}', np.array(points))
        np.save(f'./data/point_labels_{sampled_count}', np.array(pnt_label_list))
        np.save(f'./data/shape_labels_{sampled_count}', np.array(shape_label_list))
        np.save(f'./data/shape_part_count_{sampled_count}', np.array(shape_part_count))
        pickle.dump(shape_part_label_mapping, open(f'./data/shape_part_label_map_{sampled_count}.pickle', 'wb'))
        points, pnt_label_list, shape_label_list, shape_part_count = [], [], [], []

# print(shape_part_label_mapping)
# print(np.array(points).shape)
# print(np.array(shape_part_count).shape, shape_part_count)
# print(np.array(pnt_label_list).shape)
# print(np.array(shape_label_list))

# np.save(f'./data/points_{sampled_count}', np.array(points))
# np.save(f'./data/point_labels_{sampled_count}', np.array(pnt_label_list))
# np.save(f'./data/shape_labels_{sampled_count}', np.array(shape_label_list))
    

end = time.time()
# points = points[0]
# print(points.shape, type(points[0]))
# mesh = get_normalized_partnet_mesh(SHAPENET_ID).transform(so3_to_se3(rotation))
# print("Here1")
# sem_to_partnet = get_sem_to_partnet_transform(SHAPENET_ID, mesh)
# print("Here 2")
# sem_transmat = get_shapenetsem_axis_alignment(SHAPENET_ID)
# print("Here 3")
# shapenet_sem_mesh = get_normalized_shapenetsem_mesh(SHAPENET_ID).transform(so3_to_se3(sem_transmat)).transform(sem_to_partnet).paint_uniform_color([0,1,0])
# print("I am here!")

# o3d.visualization.draw_geometries([mesh, shapenet_sem_mesh])

#######################################################
# points = np.array(points)
# pnt_label_list = np.array(pnt_label_list)
# shape_label_list = np.array(shape_label_list)
# print(points.shape, pnt_label_list.shape, shape_label_list.shape)
# print(end - start)

# np.save(f'./data/{CAT}_points', points)
# np.save(f'./data/{CAT}_point_labels', pnt_label_list)
# np.save(f'./data/{CAT}_shape_labels', shape_label_list)

#######################################################

# def rand_color():
#     import random
#     return [random.random(), random.random(), random.random()]

# pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[0]))
# labels = np.unique(pnt_label_list[0])
# print(labels)
# color_map = {label : rand_color() for label in labels}
# pointcloud.colors = o3d.utility.Vector3dVector([color_map[label] for label in pnt_label_list[0]])
# o3d.visualization.draw_geometries([pointcloud])

