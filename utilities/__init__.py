import pkgutil

mapping = pkgutil.get_data(__name__, "index/anno_to_shapenet.pickle")
print(mapping)