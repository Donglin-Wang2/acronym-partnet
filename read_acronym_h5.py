import h5py

import numpy as np

path = '/home/donglin/Data/acronym/grasps/AccentChair_6cb6373befd7e1a7aa918b099faddaba_0.006500588211428664.h5'

grasps = h5py.File(path, 'r')
print(grasps['/grasps/qualities/flex/object_motion_during_closing_angular'][()])