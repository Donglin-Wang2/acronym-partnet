try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


import indexes

import os
import json
import pickle
from enum import Enum

PICKLE_EXT = '.pickle'
PATHS = json.load(pkg_resources.open_text('config.json'))


class IndexType(Enum):
    SHAPENET_ID_TO_CAT_ID = 'shapenet_id_to_cat_id' + PICKLE_EXT
    SHAPENET_ID_TO_PARTNET_ID = 'shapenet_id_to_partnet_id' + PICKLE_EXT
    PARTNET_ID_TO_SHAPENET_ID = 'partnet_id_to_shapenet_id' + PICKLE_EXT


def load_or_produce(category: IndexType):
    if pkg_resources.is_resource(indexes, category.value):
        return pickle.load(pkg_resources.open_binary(indexes, category.value))
    else:
        produce_index(category)


def produce_index(category: IndexType):
    if category == IndexType.SHAPENET_ID_TO_CAT_ID:
        produce_shapenet_id_to_cat_id()
    elif category == IndexType.SHAPENET_ID_TO_PARTNET_ID or category == IndexType.PARTNET_ID_TO_SHAPENET_ID:
        produce_partnet_map()
    

def produce_shapenet_id_to_cat_id():
    shapenet_to_cat_id = {}
    for cat_id in os.listdir(PATHS['shapenetv2_path']):
        if not os.path.isdir(os.path.join(PATHS['shapenetv2_path'], cat_id)):
            continue
        for model_id in os.listdir(os.path.join(PATHS['shapenetv2_path'], cat_id)):
            shapenet_to_cat_id[model_id] = cat_id
    pickle.dump(shapenet_to_cat_id, pkg_resources.open_binary(
        indexes, IndexType.SHAPENET_ID_TO_CAT_ID.value))


def produce_partnet_map():
    partnet_to_shapenet = {}
    shapenet_to_partnet = {}
    with open(PATHS["partnet_root"], 'r') as file_obj:
        for line in file_obj.readlines():
            tokens = line.split(" ")
            partnet_to_shapenet[tokens[0]] = tokens[3]
            shapenet_to_partnet[tokens[3]] = tokens[0]
    pickle.dump(partnet_to_shapenet, open(
        IndexType.PARTNET_ID_TO_SHAPENET_ID.value, 'wb'))
    pickle.dump(shapenet_to_partnet, open(
        IndexType.SHAPENET_ID_TO_PARTNET_ID.value, 'wb'))


def produce_partnet_acronym_joint_index():
    partnet_acronym_joint_index = {}
    with open(PATHS['partnet_metadata'], 'r') as f:
        for line in f.readlines():
            tokens = line.split(" ")
            partnet_id, shapenet_id, category_name = tokens[0], tokens[3], tokens[2]
            partnet_acronym_joint_index[shapenet_id] = {
                "partnet_id": partnet_id,
                "category_name": category_name
            }
    
