import pybullet as p
from pybullet_utils import load_collision_obj
from utils import *

p.connect(p.GUI)
SHAPENET_ID = '6cb6373befd7e1a7aa918b099faddaba'
id = p.loadURDF(f'./temp_meshes/{SHAPENET_ID}.urdf', basePosition=[0,0,1])
p.setGravity(0,0,-10)
while 1:
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    p.stepSimulation()
    print(p.getLinkState(id, 0, computeForwardKinematics=True))
