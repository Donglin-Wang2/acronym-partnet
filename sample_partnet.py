from utils import *


import json
import math
import time

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
CAT = 'Lamp'
points = []
pnt_label_list = []
shape_label_list = []
for k, v in pickle.load(open('./mappings/cat_to_ids.pickle', 'rb')).items():
    if k != CAT:
        continue
    print(k, len(v))
    for item in v:
        anno_id = item['anno_id']
        shapenet_id = item['shapenet_id']
        
        # anno_id = '39057'
        # shapenet_id = '6cb6373befd7e1a7aa918b099faddaba'
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
            break
        if len(hier['children']) == 1:
            continue
        for child in hier['children']:
            file_list = [f'/home/donglin/Data/data_v0/{anno_id}/objs/{name}.obj' for name in child['objs']]
            part_mesh = combine_meshes(file_list).transform(normalize_transmat).transform(so3_to_se3(rotation)).paint_uniform_color([0,0,1])
            part_meshes.append(part_mesh)
            part_mesh_names.append(child['name'])
            print(child['name'])
            part_surface_area = part_mesh.get_surface_area()
            total_surface_area += part_surface_area
            part_surface_areas.append(part_surface_area)
        for i, part_mesh in enumerate(part_meshes):
            
            if i == len(part_meshes) - 1:
                num_sample_part = NUM_SAMPLES - num_sampled
            else:
                num_sample_part = math.floor(NUM_SAMPLES * part_surface_areas[i] / total_surface_area)
        
            num_sampled += num_sample_part
            point_labels += [part_mesh_names[i]] * math.floor(num_sample_part)
            sampled_points.append(np.asarray(part_mesh.sample_points_uniformly(math.floor(num_sample_part)).points))
            
        
        points.append(np.concatenate(sampled_points, axis=0))
        pnt_label_list.append(point_labels)
        shape_label_list.append(k)
        
        # print(anno_id, shapenet_id)
        break
    break
end = time.time()
# points = points[0]
# print(points.shape, type(points[0]))
# # pcs = [o3d.geometry.PointCloud(pts) for pts in points]

# # pcs.append(get_normalized_partnet_mesh(SHAPENET_ID).transform(so3_to_se3(rotation)).paint_uniform_color([0,1,0]))

# mesh = get_normalized_partnet_mesh(SHAPENET_ID).transform(so3_to_se3(rotation))
# sem_to_partnet = get_sem_to_partnet_transform(SHAPENET_ID, mesh)
# pcs = part_meshes
# pcs += [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))]
# # pcs = [part.transform(partnet_to_sem_transmat) for part in part_meshes]
# sem_transmat = get_shapenetsem_axis_alignment(shapenet_id)
# shapenet_sem_mesh = get_normalized_shapenetsem_mesh(shapenet_id).transform(so3_to_se3(sem_transmat)).transform(sem_to_partnet).paint_uniform_color([0,1,0])

# # pcs.append(mesh.transform(partnet_to_sem_transmat))
# pcs.append(mesh)
# pcs.append(shapenet_sem_mesh)
# o3d.visualization.draw_geometries(pcs)


points = np.array(points)
pnt_label_list = np.array(pnt_label_list)
shape_label_list = np.array(shape_label_list)
print(points.shape, pnt_label_list.shape, shape_label_list.shape)
print(end - start)

# np.save(f'{CAT}_points', points)
# np.save(f'{CAT}_point_labels', pnt_label_list)
# np.save(f'{CAT}_shape_labels', shape_label_list)

def rand_color():
    import random
    return [random.random(), random.random(), random.random()]

pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[0]))
labels = np.unique(pnt_label_list[0])
print(labels)
color_map = {label : rand_color() for label in labels}
pointcloud.colors = o3d.utility.Vector3dVector([color_map[label] for label in pnt_label_list[0]])
o3d.visualization.draw_geometries([pointcloud])

