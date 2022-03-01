from utils import *
from scipy.spatial.transform.rotation import Rotation as R

SHAPENET_ID = '6cb6373befd7e1a7aa918b099faddaba'
print(get_shapenetv2_model_path(SHAPENET_ID), os.path.isfile(get_shapenetv2_model_path(SHAPENET_ID)))
print(get_partnet_pointcloud_path(SHAPENET_ID),  os.path.isfile(get_partnet_pointcloud_path(SHAPENET_ID)))
print(get_shapenetv1_model_path(SHAPENET_ID), os.path.isfile(get_shapenetv1_model_path(SHAPENET_ID)))
print(get_shapenetv1_stat(SHAPENET_ID))
print(get_shapenetv2_stat(SHAPENET_ID))
print(get_partnet_pointcloud(SHAPENET_ID), get_partnet_pointcloud(SHAPENET_ID).shape)
print(get_normalized_partnet_pointcloud(SHAPENET_ID), get_normalized_partnet_pointcloud(SHAPENET_ID).shape)
print(get_shapenetv1_transmat(SHAPENET_ID))
produce_normalized_shapenetv1(SHAPENET_ID)

# load_visual_obj(get_shapenetv2_model_path(SHAPENET_ID))
# rotation = R.from_rotvec([0,180,0], degrees=True)
# pts = rotation.apply(get_partnet_pointcloud(SHAPENET_ID))
# for pt in pts:
#     load_point(pt)