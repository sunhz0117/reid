
from data.common import list_pictures

from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader

import scipy.io as io
import numpy as np
import torch

from IPython import embed

class Market1501(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader
        data_path = args.datadir
        if dtype == 'train':
            data_path += '/bounding_box_train'
        elif dtype == 'test':
            data_path += '/bounding_box_test'
        else:
            data_path += '/query'

        
        self.imgs = [path for path in list_pictures(data_path) if self.id(path) != -1]
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}
        self.attribute_name, self._id2attribute = self.get_attribute_dict(dtype)

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]
        if self.id(path) not in [-1, 0]:
            person_attribute = torch.LongTensor(self._id2attribute[self.id(path)])
        else:
            person_attribute = torch.LongTensor([-1 for _ in range(12)])

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target, person_attribute

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def get_attribute_dict(mode):
        mat_path = "data/market1501_attribute/market_attribute.mat"
        data = io.loadmat(mat_path)
        info = data['market_attribute'][0, 0]

        if mode in ["train"]:
            train = info["train"][0, 0]
        else:
            train = info["test"][0, 0]

        attribute_name_dict = {
            "image_index": np.array(train['image_index'].tolist()[0]).squeeze(),

            # 2 labels
            "gender": train['gender'][0] - 1,
            "hair": train['hair'][0] - 1,
            "up": train['up'][0] - 1,
            "down": train['down'][0] - 1,
            "clothes": train['clothes'][0] - 1,
            "hat": train['hat'][0] - 1,
            "backpack": train['backpack'][0] - 1,
            "bag": train['bag'][0] - 1,
            "handbag": train['handbag'][0] - 1,

            # 4 labels
            "age": train['age'][0] - 1,

            # 8 labels: upblack, upwhite, upred, uppurple, upyellow, upgray, upblue, upgreen
            "upblack": train['upblack'][0] - 1,
            "upwhite": train['upwhite'][0] - 1,
            "upred": train['upred'][0] - 1,
            "uppurple": train['uppurple'][0] - 1,
            "upyellow": train['upyellow'][0] - 1,
            "upgray": train['upgray'][0] - 1,
            "upblue": train['upblue'][0] - 1,
            "upgreen": train['upgreen'][0] - 1,

            # 9 labels: downblack, downwhite, downpink, downpurple, downyellow, downgray, downblue, downgreen, downbrown
            "downblack": train['downblack'][0] - 1,
            "downwhite": train['downwhite'][0] - 1,
            "downpink": train['downpink'][0] - 1,
            "downpurple": train['downpurple'][0] - 1,
            "downyellow": train['downyellow'][0] - 1,
            "downgray": train['downgray'][0] - 1,
            "downblue": train['downblue'][0] - 1,
            "downgreen": train['downgreen'][0] - 1,
            "downbrown": train['downbrown'][0] - 1,
        }

        attribute_name = list(attribute_name_dict.keys())[1:11] + ["upcolor", "downcolor"]

        attribute_dict = {}
        for i in range(len(attribute_name_dict["image_index"])):
            image_index = int(attribute_name_dict["image_index"][i])
            attribute_dict[image_index] = [attrs[i] for attrs in list(attribute_name_dict.values())[1:11]]

            upcolor = -1
            if attribute_name_dict["upblack"][i]:
                upcolor = 0
            elif attribute_name_dict["upwhite"][i]:
                upcolor = 1
            elif attribute_name_dict["upred"][i]:
                upcolor = 2
            elif attribute_name_dict["uppurple"][i]:
                upcolor = 3
            elif attribute_name_dict["upyellow"][i]:
                upcolor = 4
            elif attribute_name_dict["upgray"][i]:
                upcolor = 5
            elif attribute_name_dict["upblue"][i]:
                upcolor = 6
            elif attribute_name_dict["upgreen"][i]:
                upcolor = 7

            downcolor = -1
            if attribute_name_dict["downblack"][i]:
                downcolor = 0
            elif attribute_name_dict["downwhite"][i]:
                downcolor = 1
            elif attribute_name_dict["downpink"][i]:
                downcolor = 2
            elif attribute_name_dict["downpurple"][i]:
                downcolor = 3
            elif attribute_name_dict["downyellow"][i]:
                downcolor = 4
            elif attribute_name_dict["downgray"][i]:
                downcolor = 5
            elif attribute_name_dict["downblue"][i]:
                downcolor = 6
            elif attribute_name_dict["downgreen"][i]:
                downcolor = 7
            elif attribute_name_dict["downbrown"][i]:
                downcolor = 8

            attribute_dict[image_index].append(upcolor)
            attribute_dict[image_index].append(downcolor)

        return attribute_name, attribute_dict

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]
