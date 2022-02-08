import os


ACRONYM_PATH = os.path.join("/", "home", "donglin", "Data", "acronym", "grasps")
cnt = 0
acronym_file_list = os.listdir(ACRONYM_PATH)
acronym_ids = []
for filename in acronym_file_list:
    id = filename.split("_")[1]
    if len(filename) >= 28:
        acronym_ids.append(id)

print(len(acronym_ids))


PARTNET_PATH = os.path.join('/', 'home', 'donglin', 'Github', 'partnet_dataset', 'stats', 'all_valid_anno_info.txt')
partnet_ids = set()

with open(PARTNET_PATH, 'r') as file_obj:
    for line in file_obj.readlines():
        partnet_ids.add(line.split(" ")[3])

hits = 0
for id in acronym_ids:
    if id in partnet_ids:
        hits += 1
print(hits)