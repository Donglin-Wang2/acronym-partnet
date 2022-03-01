import pybullet as p

MESH_ID = "7223820f07fd6b55e453535335057818"
MESH_SCALE = 0.015551998509716471
AC_MESH = f'/home/donglin/Data/acronym/meshes/Mug/{MESH_ID}.obj'
SEM_MESH = f'/home/donglin/Data/ShapeNetSem.v0/models-OBJ/models/{MESH_ID}.obj'
p.connect(p.GUI)
def load_acronym_mesh(path, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
    shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                   fileName=path,
                                   flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                   p.GEOM_CONCAVE_INTERNAL_EDGE,
                                   rgbaColor=[0,0,1,0.2],
                                   meshScale=[MESH_SCALE]*3
                                   )
    return p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position, baseOrientation=orientation)

def load_visual_obj(path, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
    shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                   fileName=path,
                                   flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                   p.GEOM_CONCAVE_INTERNAL_EDGE,
                                   meshScale=[MESH_SCALE]*3
                                   )
    return p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position, baseOrientation=orientation)

load_acronym_mesh(AC_MESH)
load_visual_obj(SEM_MESH)

while 1:
    p.stepSimulation()