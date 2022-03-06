from utils import *
import pickle
joint_ids = pickle.load(open('./mappings/joint_ids.pickle', 'rb'))
print(len(joint_ids))
SHAPENET_ID = '6cb6373befd7e1a7aa918b099faddaba'
get_partnet_acronym_transform(SHAPENET_ID)
