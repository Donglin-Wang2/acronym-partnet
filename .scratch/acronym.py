import collections
import os
DATA_PATH = os.path.join("/", "home", "donglin", "Data", "acronym", "grasps")
cnt = 0
acronym_file_list = os.listdir(DATA_PATH)
acronym_id_list = [name.split("_")[1] for name in acronym_file_list]
acronym_good_ids, acronym_bad_ids = [], []
print(collections.Counter([len(id) for id in acronym_id_list]))
for id in acronym_id_list:
    if len(id) != 32:
        acronym_bad_ids.append(id)
    else:
        acronym_good_ids.append(id)
