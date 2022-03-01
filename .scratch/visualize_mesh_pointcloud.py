import os

import numpy as np
import pybullet as p

MESH_ID = "7223820f07fd6b55e453535335057818"
POINT_ID = "8745"
### For the ACRONYM Dataset, irrelevent for to the issue
# MESH_SCALE = 0.015551998509716471
# MESH_PATH = f'/home/donglin/Data/acronym/meshes/Mug/{MESH_ID}.obj'
POINT_PATH = f'/home/donglin/Data/data_v0/{POINT_ID}/point_sample/sample-points-all-pts-label-10000.ply'
POINT_PATH = f'/home/donglin/Data/data_v0/{POINT_ID}/point_sample/pts-10000.txt'
SHAPNET_MESH_PATH = '/home/donglin/Data/ShapeNetCore.v2/03797390/7223820f07fd6b55e453535335057818/models/model_normalized.obj'

p.connect(p.GUI)
def load_visual_obj(path, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
    shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                   fileName=path,
                                   flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                   p.GEOM_CONCAVE_INTERNAL_EDGE,
                                #    meshScale=[MESH_SCALE]*3
                                   )
    return p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position, baseOrientation=orientation)


def load_point(position):
    shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                   radius=0.03,
                                   rgbaColor=[0,0,1,1]
                                   )
    return p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position)


pts = []
with open(POINT_PATH, 'r') as pts_f:
    tokens = pts_f.readlines()
    for token in tokens:
        pts.append([float(ele) for ele in token.split()[:3]])
pts = np.array(pts)
# pts[:, [1,2]] = pts[:, [2,1]]
load_visual_obj(SHAPNET_MESH_PATH)
for pt in pts:
    load_point(pt)
while 1:
    p.stepSimulation()
