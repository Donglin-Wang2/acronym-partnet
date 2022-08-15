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
shape_part_count_files = ["./data/shape_part_count_5000.npy", "./data/shape_part_count_10000.npy", "./data/shape_part_count_11823.npy"]

# point_labels_files = ["./data/point_labels_5000.npy", "./data/point_labels_10000.npy"]
# points_files = ["./data/points_5000.npy", "./data/points_10000.npy"]
# shape_labels_files = ["./data/shape_labels_5000.npy", "./data/shape_labels_10000.npy"]
# shape_part_count_files = ["./data/shape_part_count_5000.npy", "./data/shape_part_count_10000.npy"]

point_labels = load_concat_arrays(point_labels_files)
points = load_concat_arrays(points_files)
shape_labels = load_concat_arrays(shape_labels_files)
shape_part_count = load_concat_arrays(shape_part_count_files)
shape_part_label_map = pickle.load(open('./data/shape_part_label_map_11823.pickle', 'rb'))

# print(shape_part_label_map)
# print(point_labels.shape, points.shape, shape_labels.shape, shape_part_count.shape)

shape_name_to_label = {shape:i for i, shape in enumerate(shape_part_label_map.keys())}
shape_name_to_part_label = {}
max_part_count_by_cat = {cat:len(parts) for cat, parts in shape_part_label_map.items()}

new_shape_labels = []
new_point_labels = []
new_shape_part_label_map = []
points_by_cat = {shape:[] for shape in shape_part_label_map.keys()}
point_labels_by_cat = {shape:[] for shape in shape_part_label_map.keys()}
temp_index = {shape:[] for shape in shape_part_label_map.keys()}



for i, (shape, parts) in enumerate(shape_part_label_map.items()):
    shape_name_to_part_label[shape] = {part:i for i, part in enumerate(parts)}


# print(max_part_count_by_cat)
# print(shape_name_to_part_label)


for label, point, point_label, count in zip(shape_labels, points, point_labels, shape_part_count):
    new_label = shape_name_to_label[label]
    new_shape_labels.append(new_label)
    new_point_label = []
    for pnt in point_label:
        new_point_label.append(shape_name_to_part_label[label][pnt])
    new_point_labels.append(new_point_label)
    points_by_cat[label].append(point)
    point_labels_by_cat[label].append(new_point_label)
    if count == max_part_count_by_cat[label]:
        index = len(points_by_cat[label]) - 1
        temp_index[label].append(index)
        

new_shape_labels = np.array(new_shape_labels)
new_point_labels = np.array(new_point_labels)
points_by_cat = {shape:np.stack(data) for shape, data in points_by_cat.items()}
point_labels_by_cat = {shape:np.stack(data) for shape, data in point_labels_by_cat.items()}
temp_index = {shape:np.stack(data) for shape, data in temp_index.items()}
# print(new_shape_labels.shape)
# print(new_point_labels.shape)
# print(np.unique(new_point_labels))
# print({shape:arr.shape for shape, arr in points_by_cat.items()})
# print({shape:arr.shape for shape, arr in point_labels_by_cat.items()})
# print({shape:arr.shape for shape, arr in temp_index.items()})
# print({shape:arr.shape for shape, arr in temp_point_labels_by_cat.items()})

# np.save('./data/combined_points.npy', points)
# np.save('./data/combined_point_labels.npy', new_point_labels)
# np.save('./data/combined_shape_labels.npy', new_shape_labels)
# np.save('./data/points_by_cat.pickle', points_by_cat)
# np.save('./data/point_labels_by_cat.pickle', point_labels_by_cat)
# np.save('./data/temp_index.pickle', temp_index)
# np.save('./data/temp_point_labels_by_cat.pickle', temp_point_labels_by_cat)
pickle.dump(points_by_cat, open('./data/points_by_cat.pkl', 'wb'))
pickle.dump(point_labels_by_cat, open('./data/point_labels_by_cat.pkl', 'wb'))
pickle.dump(temp_index, open('./data/temp_index.pkl', 'wb'))



##############################
## Testing the loading of dict

loaded_cat_pnt = pickle.load(open('./data/points_by_cat.pkl', 'rb'))
loaded_cat_pnt_label = pickle.load(open('./data/point_labels_by_cat.pkl', 'rb'))

print({shape:arr.shape for shape, arr in loaded_cat_pnt.items()})
print({shape:arr.shape for shape, arr in loaded_cat_pnt_label.items()})