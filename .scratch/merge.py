import os
import collections

ACRONYM_PATH = os.path.join("/", "home", "donglin", "Data", "acronym", "grasps")
cnt = 0
acronym_file_list = os.listdir(ACRONYM_PATH)
acronym_ids = []
acronym_id_to_scale = collections.defaultdict(float)
for filename in acronym_file_list:
    tokens = filename.split("_")
    id = tokens[1]
    if len(filename) >= 28:
        acronym_ids.append(id)
        acronym_id_to_scale[id] = float(tokens[-1][:-3])

PARTNET_PATH = os.path.join('/', 'home', 'donglin', 'Github', 'partnet_dataset', 'stats', 'all_valid_anno_info.txt')
partnet_id_to_cat = collections.defaultdict(str)
categories_to_ids = collections.defaultdict(list)
id_to_anno = collections.defaultdict(str)
anno_to_id = collections.defaultdict(str)
category_count = collections.Counter()

with open(PARTNET_PATH, 'r') as file_obj:
    for line in file_obj.readlines():
        tokens = line.split(" ")
        partnet_id_to_cat[tokens[3]] = tokens[2]
        anno_to_id[tokens[0]] = tokens[3]
        id_to_anno[tokens[3]] = tokens[0]

hits = 0
for id in acronym_ids:
    if id in partnet_id_to_cat:
        hits += 1
        categories_to_ids[partnet_id_to_cat[id]].append(id)
        category_count[partnet_id_to_cat[id]] += 1
        
print(category_count)
print(categories_to_ids["Chair"][1])
print(id_to_anno[categories_to_ids["Chair"][1]])
print(acronym_id_to_scale[categories_to_ids["Chair"][1]])