from __future__ import print_function, absolute_import
import os
import numpy as np
import random

def process_query_sysu(data_path, mode = 'all', img_mode='ir', relabel=False):

    if mode == 'all' and img_mode == "rgb":
        cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor' and img_mode == "rgb":
        cameras = ['cam1', 'cam2']
    elif mode == 'all' and img_mode == "ir":
        cameras = ['cam3', 'cam6']
    elif mode == 'indoor' and img_mode == "ir":
        cameras = ['cam3', 'cam6']

    
    file_path = os.path.join(data_path,'exp/test_id.txt')
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
                img_dir = os.path.join(data_path,cam,id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
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

def process_gallery_sysu(data_path, mode = 'all', img_mode = "rgb", trial = 0, relabel=False):
    
    random.seed(trial)

    if mode == 'all' and img_mode == "rgb":
        cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor' and img_mode == "rgb":
        cameras = ['cam1', 'cam2']
    elif mode == 'all' and img_mode == "ir":
        cameras = ['cam3', 'cam6']
    elif mode == 'indoor' and img_mode == "ir":
        cameras = ['cam3', 'cam6']

        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    os.makedirs("show", exist_ok=True)
    with open("show/gallery.txt", "w") as txt:
        for id in sorted(ids):
            for cam in cameras:
                img_dir = os.path.join(data_path,cam,id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
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
    
def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    if modal=='visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal=='thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)
