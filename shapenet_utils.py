try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

import os
import indexes
from indexing_utils import load_or_produce, IndexType, PATHS
import pickle


def get_all_shapenet_ids():
    return load_or_produce(IndexType.SHAPENET_ID_TO_CAT_ID).keys()


def get_shapenetv1_model_path(shapenet_id):
    shapenet_to_cat_id = load_or_produce(IndexType.SHAPENET_ID_TO_CAT_ID)
    shapenet_cat_id = shapenet_to_cat_id[shapenet_id]
    return os.path.join(PATHS['shapenetv1_root'], shapenet_cat_id, shapenet_id, 'model.obj')


def get_shapenetv2_model_path(shapenet_id):
    shapenet_to_cat_id = load_or_produce(IndexType.SHAPENET_ID_TO_CAT_ID)
    shapenet_cat_id = shapenet_to_cat_id[shapenet_id]
    return os.path.join(PATHS['shapenetv2_root'], shapenet_cat_id, shapenet_id, 'models', 'model_normalized.obj')
