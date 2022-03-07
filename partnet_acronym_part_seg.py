from utils import *

SHAPENET_ID = '6cb6373befd7e1a7aa918b099faddaba'
PARTNET_PATH = '/home/donglin/Data/data_v0/39057/'

print(get_partnet_pointcloud_path(SHAPENET_ID))

with open(PARTNET_PATH + 'point_sample/' + 'sample-points-all-label-10000.txt', 'r') as fo:
    labels = set(fo.readlines())
print(sorted(labels))

