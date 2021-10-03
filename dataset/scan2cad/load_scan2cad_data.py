"""
Description:
    Load exported scan2cad data from /root/Dataset to put them in split folders according to scan2cad_{split}.txt
"""
import os,sys

DATA_PATH = '/root/Dataset/Scan2CAD/pointgroup/data'
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
META_DIR = os.path.join(BASE_PATH, 'meta_data')
SPLITS = ['train', 'val']

class Spliter():
    def __init__(self, splits):
        self.data_path = DATA_PATH
        for split in splits:
            if split == 'train':
                train_list = self.load_meta_file(split)
            elif split == 'val':
                val_list = self.load_meta_file(split)
            else:
                exit(-1)
    
    def load_meta_file(self, split):
        data_list = []
        split_file = os.path.join(META_DIR, 'scan2cad_{}.txt'.format(split))
        with open(split_file, 'r') as f:
            data_list = f.read.splitlines()

        return data_list, len(data_list)

    def split_data(self):
        pass

