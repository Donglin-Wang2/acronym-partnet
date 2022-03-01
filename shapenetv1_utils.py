try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from . import mappings
import pickle
if __name__ == "__main__":
    if pkg_resources.is_resource(mappings, 'haha.pickle'):
        print("Lol")
    mapping = pickle.load(pkg_resources.open_binary(mappings, 'anno_to_shapenet.pickle'))

    print(mapping)