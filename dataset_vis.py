import numpy as np
import open3d as o3d

# points = np.load('./data/points_5000.npy')
# point_labels = np.load('./data/point_labels_5000.npy')
# shape_labels = np.load('./data/shape_labels_5000.npy')
points = np.load('./data/combined_points.npy')
point_labels = np.load('./data/combined_point_labels.npy')
shape_labels = np.load('./data/combined_shape_labels.npy')
cls = 0
ind = (shape_labels == cls).reshape(-1,)
points = points[ind][:,:,:]
point_labels = point_labels[ind]
print(points.shape, point_labels.shape)


def rand_color():
    import random
    return [random.random(), random.random(), random.random()]

pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[1]))
labels = np.unique(point_labels[1])
print(labels)
color_map = {label : rand_color() for label in labels}
pointcloud.colors = o3d.utility.Vector3dVector([color_map[label] for label in point_labels[1]])
o3d.visualization.draw_geometries([pointcloud])