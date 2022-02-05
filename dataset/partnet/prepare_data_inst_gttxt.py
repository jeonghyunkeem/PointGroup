'''
Generate instance groundtruth .txt files (for evaluation)
'''

import numpy as np
import glob
import torch
import os

# semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
# semantic_label_names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

DATA_ROOT = '/root/PointGroup/dataset'
DATASET = 'partnet'
PARTNET_PATH = 'partnet_pts'
CATEGORY = 'chair'
DATA_PATH = os.path.join(DATA_ROOT, DATASET, PARTNET_PATH)
CATEGORY_PATH = os.path.join(DATA_PATH, CATEGORY)

def load_split_files(split):
    with open(os.path.join(CATEGORY_PATH, f'{split}.txt'), 'r') as f:
        file_names = [item.rstrip() for item in f.readlines()]

    return file_names

if __name__ == '__main__':
    split = 'val'
    files = load_split_files(split) #sorted(glob.glob('{}/scene*_inst_nostuff.pth'.format(split)))
    rooms = [torch.load(os.path.join(CATEGORY_PATH, i+'.pth')) for i in files]

    out_path = os.path.join(CATEGORY_PATH, split + '_gt')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for i in range(len(rooms)):
        # xyz, rgb, label, instance_label = rooms[i]   # label 0~19 -100;  instance_label 0~instance_num-1 -100
        xyz_origin, label, gt_mask, gt_other_mask, gt_valid = rooms[i]
        instance_label = np.argmax(gt_mask.transpose(), axis=1) # (0, 0~num_ins-1)
        
        partnet_id = files[i].split('/')[-1][:12]
        print('{}/{} {}'.format(i + 1, len(rooms), partnet_id))

        instance_label_new = np.zeros(instance_label.shape, dtype=np.int32)  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            sem_id = int(label[instance_mask[0]])
            if(sem_id == -100): sem_id = 0
            semantic_label = sem_id
            instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

        np.savetxt(os.path.join(out_path, partnet_id + '.txt'), instance_label_new, fmt='%d')





