import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

TRANSFORM_PATH = "./chair_table_storagefurniture_bed_shapenetv1_to_partnet_alignment.json"
SHAPENET_ID = "6cb6373befd7e1a7aa918b099faddaba"
# SEM_ORIG_PATH = f'/home/donglin/Data/ShapeNetSem.v0/models-OBJ/models/{SHAPENET_ID}.obj'
# SEM_PATH = f'/home/donglin/Desktop/{SHAPENET_ID}.obj'
SEM_PATH = f'./temp_meshes/{SHAPENET_ID}_sem_normalized.obj'
CORE_PATH = f'/home/donglin/Data/ShapeNetCore.v2/03001627/{SHAPENET_ID}/models/model_normalized.obj'
# CORE_PATH = '/home/donglin/Desktop/model_nn.obj'
# CORE_PATH = f'/home/donglin/Data/ShapeNetCore.v1/03001627/{SHAPENET_ID}/model.obj'

ANNO_ID = 39057
POINT_PATH = f'/home/donglin/Data/data_v0/{ANNO_ID}/point_sample/sample-points-all-pts-label-10000.ply'
POINT_PATH = f'/home/donglin/Data/data_v0/{ANNO_ID}/point_sample/pts-10000.txt'
SEM_SCALE = 0.0245791525405232
# SEM_SCALE = 0.060553
CENTROID = [0.3977983202067827, 0.40019718530994164, 0.5265005023087718]
ROTATION = R.from_rotvec([0,0,-90], degrees=True) * R.from_rotvec([0,-90,0], degrees=True) 
ORIENTATION = ROTATION.as_quat()
p.connect(p.GUI)


def load_sem_mesh(path, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
    shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                   fileName=path,
                                   flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                   p.GEOM_CONCAVE_INTERNAL_EDGE,
                                   rgbaColor=[0, 0, 1, 0.5],
                                   meshScale=[SEM_SCALE] * 3
                                   )
    return p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position, baseOrientation=orientation)


def load_visual_obj(path, position=[0, 0, 0], orientation=[0, 0, 0, 1],
    rgba=[1,1,1,1], scale=1):
    shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                   fileName=path,
                                   flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                   p.GEOM_CONCAVE_INTERNAL_EDGE,
                                   rgbaColor=rgba,
                                   meshScale=[scale]*3
                                   )
    return p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position, baseOrientation=orientation)


def load_point(position):
    shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                   radius=0.03,
                                   rgbaColor=[0, 0, 1, 1]
                                   )
    return p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position)

def get_points():
    pts = []
    with open(POINT_PATH, 'r') as pts_f:
        tokens = pts_f.readlines()
        for token in tokens[10:]:
            pts.append([float(ele) for ele in token.split()[:3]])
        pts = np.array(pts)
    return pts

# Adding the centroid to all the points
# pts = get_points()
# pts = np.matmul(ORIENTATION, pts.T).T
# pts = pts + [CENTROID] * pts.shape[0]



load_visual_obj(SEM_PATH, 
        orientation=ORIENTATION
        )
load_visual_obj(CORE_PATH, rgba=[0,1,0,0.5])

while 1:
    p.stepSimulation()
