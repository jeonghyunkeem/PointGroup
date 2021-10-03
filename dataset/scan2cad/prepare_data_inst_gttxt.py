'''
Generate instance groundtruth .txt files (for evaluation)
'''

import numpy as np
import glob
import torch
import os

from s2c_map import CARED_CLASS_MASK

semantic_label_idxs = [sem_idx+1 for sem_idx in CARED_CLASS_MASK]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_DATA = os.path.join(BASE_DIR, 'meta_data')
DATA_ROOT = os.path.join(BASE_DIR, 'data3')
FILENAME = '_inst.pth'

def load_data_split(split):
    all_scan_names = list(set(os.path.basename(scan)[:12] \
            for scan in os.listdir(DATA_ROOT) if scan.startswith('scene')))
    split_file = os.path.join(META_DATA, 'scan2cad_{}.txt'.format(split))
    with open(split_file, 'r') as f:
        split_names = []
        split_names = f.read().splitlines()
    error_meta = os.path.join(DATA_ROOT, 'error_scan.txt')
    with open(error_meta, 'r') as f:
        error_list = []
        error_list = f.read().splitlines()
    split_data_names = []
    for i, scan_name in enumerate(split_names):
        if scan_name not in all_scan_names or scan_name in error_list: continue
        split_data_names.append(os.path.join(DATA_ROOT, scan_name + FILENAME))
    return split_data_names

if __name__ == '__main__':
    split = 'val'
    data_names = load_data_split(split)
    files = sorted(data_names)
    rooms = [torch.load(i) for i in files if os.path.exists(i)]

    SPLIT_DIR = os.path.join(BASE_DIR, '{}'.format(split)) + '_gt'
    if not os.path.exists(SPLIT_DIR):
        os.mkdir(SPLIT_DIR)

    for i in range(len(rooms)):
        xyz, rgb, label, instance_label = rooms[i]   # label 0~19 (-1);  instance_label 0~instance_num-1 (-1)
        scene_name = files[i].split('/')[-1][:12]
        print(f'{(i+1):4d}/{len(rooms):4d}: {scene_name:12s}')

        instance_label_new = np.zeros(instance_label.shape, dtype=np.int32)  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            sem_id = int(label[instance_mask[0]])
            if(sem_id == -1): sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

        np.savetxt(os.path.join(SPLIT_DIR, scene_name + '.txt'), instance_label_new, fmt='%d')





