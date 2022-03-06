import h5py

path = "/home/donglin/Data/shapenet_part_seg_hdf5_data/hdf5_data/ply_data_test0.h5"

f = h5py.File(path, 'r')
print(f.keys())
data = f["data"][:]
label = f["label"][:]
pid = f["pid"][:]
print(data[0])
print(label[0])
print(pid[0])
print(data.shape)
print(label.shape)
print(pid.shape)