import json
import h5py


class DemoDataParser():
    def __init__(self):
        json_path = "/content/train.json"
        with open(json_path) as annotations:
            annotations = json.load(annotations)
        N = len(annotations['categories'])
        M = len(annotations['images'])
        self.common_names = set(
            [annotations['categories'][i]['common_name'] for i in range(N)])

        cname_category_index_map = {}
        for i in range(N):
            cname_category_index_map[annotations['categories'][i]
                                     ['common_name']] = i

        self.cname_category_index_map = cname_category_index_map

        cname_image_index_map = {}
        for c in self.common_names:
            cname_image_index_map[c] = []
        for i in range(M):
            cname = annotations['categories'][annotations['annotations'][i]
                                              ['category_id']]['common_name']
            cname_image_index_map[cname].append(i)
        self.cname_image_index_map = cname_image_index_map

        cname_description_map = {}
        for i in range(N):
            cname = annotations['categories'][i]['common_name']
            cname_description_map[cname] = annotations['categories'][i][
                'description']
        self.cname_description_map = cname_description_map

        h5_file = h5py.File("/content/images.hdf5", 'r')
        self.images = h5_file['images']
        self.annotations = annotations