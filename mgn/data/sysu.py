from __future__ import print_function, absolute_import
import numpy as np
from PIL import Image
import torch.utils.data as data
import os
import random
import torch
from torchvision import transforms
from PIL import Image
from torchvision.datasets.folder import default_loader

from IPython import embed

class SYSU(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):

        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']

        # load id info
        file_path_train = os.path.join(data_dir, 'exp/train_id.txt')
        file_path_val = os.path.join(data_dir, 'exp/val_id.txt')
        with open(file_path_train, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_train = ["%04d" % x for x in ids]

        with open(file_path_val, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_val = ["%04d" % x for x in ids]

        # combine train and val split
        id_train.extend(id_val)

        files_rgb = []
        files_ir = []
        for id in sorted(id_train):
            for cam in rgb_cameras:
                img_dir = os.path.join(data_dir, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                    files_rgb.extend(new_files)

            for cam in ir_cameras:
                img_dir = os.path.join(data_dir, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                    files_ir.extend(new_files)

        # relabel
        pid_container = set()
        for img_path in files_ir:
            pid = int(img_path[-13:-9])
            pid_container.add(pid)
        self.pid2label = {pid: label for label, pid in enumerate(pid_container)}

        self.train_color_label, self.train_thermal_label = [], []
        for img_path in files_rgb:
            # label
            pid = int(img_path[-13:-9])
            pid = self.pid2label[pid]
            self.train_color_label.append(pid)

        for img_path in files_ir:
            # label
            pid = int(img_path[-13:-9])
            pid = self.pid2label[pid]
            self.train_thermal_label.append(pid)

        # rgb attribute
        self.attr_dict = {}
        with open("data/sysu_attribute_raw.txt", "r")as file:
            for i in file.readlines():
                info = i.strip("\n").split()
                path = info[0]
                id = int(info[1])
                attrs = list(map(int, info[2:]))
                self.attr_dict[path] = attrs

        self.train_color_image = files_rgb
        self.train_thermal_image = files_ir
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.loader = default_loader


    def __getitem__(self, index):
        img1, target1 = self.loader(self.train_color_image[self.cIndex[index]]), self.train_color_label[self.cIndex[index]]
        img2, target2 = self.loader(self.train_thermal_image[self.tIndex[index]]), self.train_thermal_label[self.tIndex[index]]

        assert self.pid2label[int(self.train_color_image[self.cIndex[index]][-13:-9])] == target1
        assert self.pid2label[int(self.train_thermal_image[self.tIndex[index]][-13:-9])] == target2

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        attribute1 = torch.LongTensor(self.attr_dict[self.train_color_image[self.cIndex[index]]])
        attribute2 = torch.LongTensor([-1 for _ in range(12)])

        return img1, img2, target1, target2, attribute1, attribute2

    def __len__(self):
        return len(self.train_color_label)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None):

        self.test_image = test_img_file
        self.test_label = test_label
        self.transform = transform
        self.loader = default_loader
        self.imgs = test_img_file
        self.ids = test_label
        self.cameras = [int(path[-15]) for path in test_img_file]

        # rgb attribute
        self.attr_dict = {}
        with open("data/sysu_attribute_raw.txt", "r")as file:
            for i in file.readlines():
                info = i.strip("\n").split()
                path = info[0]
                id = int(info[1])
                attrs = list(map(int, info[2:]))
                self.attr_dict[path] = attrs

    def __getitem__(self, index):
        img = self.loader(self.test_image[index])
        img = self.transform(img)
        target = self.test_label[index]
        if self.test_image[index] in self.attr_dict.keys():
            attribute = torch.LongTensor(self.attr_dict.keys[self.test_image[index]])
        else:
            attribute = torch.LongTensor([-1 for _ in range(12)])
        return img, target, attribute

    def __len__(self):
        return len(self.test_image)

# for get sysu attribute
class SYSU_INFERENCE(data.Dataset):
    def __init__(self, data_dir, transform=None):

        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']

        # load id info
        file_path_train = os.path.join(data_dir, 'exp/train_id.txt')
        file_path_test = os.path.join(data_dir, 'exp/test_id.txt')
        file_path_val = os.path.join(data_dir, 'exp/val_id.txt')
        with open(file_path_train, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_train = ["%04d" % x for x in ids]

        with open(file_path_test, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_test = ["%04d" % x for x in ids]

        with open(file_path_val, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_val = ["%04d" % x for x in ids]

        # combine train and test and val split
        id_train.extend(id_test)
        id_train.extend(id_val)

        files_rgb = []
        files_ir = []
        for id in sorted(id_train):
            for cam in rgb_cameras:
                img_dir = os.path.join(data_dir, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                    files_rgb.extend(new_files)

        # save ids
        self.label = []

        for img_path in files_rgb:
            # label
            pid = int(img_path[-13:-9])
            self.label.append(pid)

        self.color_image = files_rgb
        self.transform = transform
        self.loader = default_loader
        self.imgs = files_rgb

    def __getitem__(self, index):
        img, target = self.loader(self.color_image[index]), self.label[index]
        img = self.transform(img)
        attribute = torch.Tensor([-1 for _ in range(12)])

        return img, target, attribute

    def __len__(self):
        return len(self.label)

def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos

def process_query_sysu(data_path, mode='all', img_mode='ir', relabel=False):

    if mode == 'all' and img_mode == "rgb":
        cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor' and img_mode == "rgb":
        cameras = ['cam1', 'cam2']
    elif mode == 'all' and img_mode == "ir":
        cameras = ['cam3', 'cam6']
    elif mode == 'indoor' and img_mode == "ir":
        cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    os.makedirs("show", exist_ok=True)
    with open("show/query.txt", "w") as txt:
        for id in sorted(ids):
            for cam in cameras:
                img_dir = os.path.join(data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                    for i in new_files:
                        txt.writelines(i + "\n")
                    files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, mode='all', img_mode='rgb', trial=0, relabel=False):
    random.seed(trial)

    if mode == 'all' and img_mode == "rgb":
        cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor' and img_mode == "rgb":
        cameras = ['cam1', 'cam2']
    elif mode == 'all' and img_mode == "ir":
        cameras = ['cam3', 'cam6']
    elif mode == 'indoor' and img_mode == "ir":
        cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    os.makedirs("show", exist_ok=True)
    with open("show/gallery.txt", "w") as txt:
        for id in sorted(ids):
            for cam in cameras:
                img_dir = os.path.join(data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                    file_path = random.choice(new_files)
                    txt.writelines(file_path + "\n")
                    files_rgb.append(file_path)
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
