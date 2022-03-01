import pybullet as p


def load_visual_obj(path, position=[0, 0, 0], orientation=[0, 0, 0, 1],
    rgba=[1,1,1,1], scale=1):
    shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                   fileName=path,
                                   flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                   p.GEOM_CONCAVE_INTERNAL_EDGE,
                                   rgbaColor=rgba,
                                   meshScale=[scale]*3
                                   )
    p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position, baseOrientation=orientation)
    return shape_id

def load_collision_obj(path, position=[0, 0, 0], orientation=[0, 0, 0, 1],
    scale=1):
    shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                   fileName=path,
                                   flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                   p.GEOM_CONCAVE_INTERNAL_EDGE,
                                   meshScale=[scale]*3
                                   )
    
    return p.createMultiBody(baseCollisionShapeIndex=shape_id, basePosition=position, baseOrientation=orientation)

def load_point(position):
    shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                   radius=0.03,
                                   rgbaColor=[0, 0, 1, 1]
                                   )
    return p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position)