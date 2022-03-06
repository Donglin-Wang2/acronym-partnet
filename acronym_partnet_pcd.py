# from utils import *

# # ids = get_shapenet_partnet_joint_ids()
# # pointclouds = []
# # for id in ids:
# #     pointclouds.append(get_partnet_pointcloud(id))
# # pointclouds = np.array(pointclouds)
# # print(pointclouds.shape)

# SHAPENET_ID = '6cb6373befd7e1a7aa918b099faddaba'
# print(get_acronym_grasp_path(SHAPENET_ID))
# print(get_shapenetv2_model_path(SHAPENET_ID))
# print(get_shapenetv1_model_path(SHAPENET_ID))
# print(get_partnet_pointcloud_path(SHAPENET_ID))

from utils import *

import glob
import h5py

#Please refer PointNet to download the part segmentation dataset
#download data at: https://github.com/charlesq34/pointnet/blob/master/part_seg/download_data.sh
ls_train=sorted(glob.glob("/Users/donglin/Desktop/Temp_Data/hdf5_data/ply_data_train*.h5"))
ls_test=sorted(glob.glob("/Users/donglin/Desktop/Temp_Data/hdf5_data/ply_data_test*.h5"))

def create_date(ls_train):
    D=[]
    L=[]
    P=[]
    for h5_filename in ls_train:
        print(h5_filename)
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        pid=f['pid'][:]
        
        D.append(data)
        L.append(label)  
        P.append(pid)

    DD=np.concatenate(D, axis=0)
    LL=np.concatenate(L, axis=0)
    PP=np.concatenate(P, axis=0)
    
    
    print(DD.shape,LL.shape,PP.shape)
    
    return DD, LL ,PP

DD, LL, PP = create_date(ls_train)
print(np.unique(PP))

cls = 4

ind=(LL==cls).reshape(-1,)
dd=DD[ind][:,:,:]
pp=PP[ind]

print(np.unique(pp))
