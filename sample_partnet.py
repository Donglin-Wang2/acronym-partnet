from utils import *


import json
import math

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

NUM_SAMPLES = 2048
ANNO_ID = 0
SHAPENET_ID = 0
points = []
for k, v in pickle.load(open('./mappings/cat_to_ids.pickle', 'rb')).items():
    for item in v:
        # anno_id = item['anno_id']
        # shapenet_id = item['shapenet_id']
        anno_id = '39057'
        shapenet_id = '6cb6373befd7e1a7aa918b099faddaba'
        if anno_id != '39057':
            break
        hier = json.load(open(f'/home/donglin/Data/data_v0/{anno_id}/result_after_merging.json', 'r'))[0]
        total_surface_area = 0
        num_sampled = 0
        part_surface_areas = []
        part_meshes = []
        sampled_points = []
        normalize_transmat = get_partnet_normalize_transmat(shapenet_id)
        rotation = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()

        if 'children' not in hier:
            break
        for child in hier['children']:
            file_list = [f'/home/donglin/Data/data_v0/{anno_id}/objs/{name}.obj' for name in child['objs']]
            part_mesh = combine_meshes(file_list).rotate(rotation).transform(normalize_transmat).paint_uniform_color([0,0,1])
            part_meshes.append(part_mesh)
            part_surface_area = part_mesh.get_surface_area()
            total_surface_area += part_surface_area
            part_surface_areas.append(part_surface_area)
        for i, part_mesh in enumerate(part_meshes):
            
            if i == len(part_meshes) - 1:
                num_sample_part = NUM_SAMPLES - num_sampled
            else:
                num_sample_part = NUM_SAMPLES * part_surface_areas[i] / total_surface_area
        
            num_sampled += num_sample_part
            
            sampled_points.append(part_mesh.sample_points_uniformly(math.floor(num_sample_part)))
            
        points.append(sampled_points)
        ANNO_ID = anno_id
        SHAPENET_ID = shapenet_id
        print(anno_id, shapenet_id)
        break
    break

points = points[0]
# pcs = [o3d.geometry.PointCloud(pts) for pts in points]
pcs = part_meshes
print(len(points))
mesh = get_normalized_partnet_mesh(SHAPENET_ID).rotate(R.from_rotvec([90, 0, 0], degrees=True).as_matrix())
pcs.append(mesh)
o3d.visualization.draw_geometries(pcs)