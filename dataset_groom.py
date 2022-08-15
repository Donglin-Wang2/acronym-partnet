from utils import *
import glob
from collections import Counter

def load_concat_arrays(file_list):
    data = []
    for file in file_list:
        data.append(np.load(file))
    return np.concatenate(data)

point_labels_files = ["./data/point_labels_5000.npy", "./data/point_labels_10000.npy", "./data/point_labels_11823.npy"]
points_files = ["./data/points_5000.npy", "./data/points_10000.npy", "./data/points_11823.npy"]
shape_labels_files = ["./data/shape_labels_5000.npy", "./data/shape_labels_10000.npy", "./data/shape_labels_11823.npy"]

point_labels = load_concat_arrays(point_labels_files)
points = load_concat_arrays(points_files)
shape_labels = load_concat_arrays(shape_labels_files)
print(point_labels.shape, points.shape, shape_labels.shape)


shape_labels_keys = Counter(shape_labels)
print(shape_labels_keys)
point_labels_keys = Counter(point_labels.flatten())
print(shape_labels_keys.keys())
print(point_labels_keys.keys())
shape_label_map = {key : i for i, key in enumerate(shape_labels_keys.keys())}
shape_label_to_part_labels = {}
offset = 0
for shape in shape_label_map.keys():
    idx = (shape_labels==shape).reshape(-1,)
    shapes = point_labels[idx]
    unique_part_labels = np.unique(shapes).tolist()
    shape_label_to_part_labels[shape] = {unique_label:i+offset for i, unique_label in enumerate(unique_part_labels)}
    offset += len(unique_part_labels)
print(shape_label_to_part_labels)

print(shape_label_map)
new_shape_labels = []
new_point_labels = []

for label in shape_labels:
    new_shape_labels.append(shape_label_map[label])

for i, shape in enumerate(point_labels):
    new_shape_pnt = []
    for pnt_label in shape:
        new_shape_pnt.append(shape_label_to_part_labels[shape_labels[i]][pnt_label])
    new_point_labels.append(new_shape_pnt)

new_shape_labels, new_point_labels = np.array(new_shape_labels), np.array(new_point_labels)
print(new_point_labels.shape, new_shape_labels.shape)
# np.save('./data/combined_points.npy', points)
# np.save('./data/combined_point_labels.npy', new_point_labels)
# np.save('./data/combined_shape_labels.npy', new_shape_labels)
