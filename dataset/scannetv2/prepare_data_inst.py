'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, argparse

import scannet_util
from scannet_util import DIR_PATH, DATASET_DIR

# ->
SCANNETV2_DATA_PATH = '/root/Dataset/ScanNet'
import os
# <-

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i
print(remapper)
parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))

def link_dataset(split):
    with open(DIR_PATH+DATASET_DIR+'/scannetv2_{}.txt'.format(split), 'r') as f:
        scan_names = f.read().splitlines()    
    
    split_scan_ply = []
    split_scan_labels = []
    split_scan_segs = []
    split_scan_agg = []
    
    if split != 'test':
        scan_dir_path = SCANNETV2_DATA_PATH+'/scans'            
        for scan_name in scan_names:
            split_scan_ply.append(os.path.join(scan_dir_path, scan_name, scan_name+'_vh_clean_2.ply'))
            split_scan_labels.append(os.path.join(scan_dir_path, scan_name, scan_name+'_vh_clean_2.labels.ply'))
            split_scan_segs.append(os.path.join(scan_dir_path, scan_name, scan_name+'_vh_clean_2.0.010000.segs.json'))
            split_scan_agg.append(os.path.join(scan_dir_path, scan_name, scan_name+'.aggregation.json'))
    else:
        scan_dir_path = SCANNETV2_DATA_PATH+'/scans_test'
        split_scan_ply = []
        for scan_name in scan_names:
            split_scan_ply.append(os.path.join(scan_dir_path, scan_name, scan_name+'_vh_clean_2.ply'))
        
    return split_scan_ply, split_scan_labels, split_scan_segs, split_scan_agg

files1, files2, files3, files4 = link_dataset(split)
files = sorted(files1)
if opt.data_split != 'test':
    assert len(files) == len(files2)
    assert len(files) == len(files3)
    assert len(files) == len(files4), "{} {}".format(len(files), len(files4))

# files = sorted(glob.glob(split + '/*_vh_clean_2.ply'))
# if opt.data_split != 'test':
#     files2 = sorted(glob.glob(split + '/*_vh_clean_2.labels.ply'))
#     files3 = sorted(glob.glob(split + '/*_vh_clean_2.0.010000.segs.json'))
#     files4 = sorted(glob.glob(split + '/*[0-9].aggregation.json'))
#     assert len(files) == len(files2)
#     assert len(files) == len(files3)
#     assert len(files) == len(files4), "{} {}".format(len(files), len(files4))

def f_test(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    torch.save((coords, colors), fn[:-15] + '_inst_nostuff.pth')
    print('Saving to ' + fn[:-15] + '_inst_nostuff.pth')


def f(fn):
    fn2 = fn[:-3] + 'labels.ply'
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'
    print(fn)

    id_scan = fn[-27:-15]
    dump_path = os.path.join(DIR_PATH+DATASET_DIR, opt.data_split)
    dump_filename = os.path.join(dump_path, id_scan+'_inst_notstuff.pth')

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    f2 = plyfile.PlyData().read(fn2)
    sem_labels = remapper[np.array(f2.elements[0]['label'])]

    with open(fn3) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    instance_segids = []
    labels = []
    with open(fn4) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            if scannet_util.g_raw2scannetv2[x['label']] != 'wall' and scannet_util.g_raw2scannetv2[x['label']] != 'floor':
                instance_segids.append(x['segments'])
                labels.append(x['label'])
                assert(x['label'] in scannet_util.g_raw2scannetv2.keys())
    if(fn == 'val/scene0217_00_vh_clean_2.ply' and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]):
        instance_segids = instance_segids[: int(len(instance_segids) / 2)]
    check = []
    for i in range(len(instance_segids)): check += instance_segids[i]
    assert len(np.unique(check)) == len(check)

    instance_labels = np.ones(sem_labels.shape[0]) * -100
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        assert(len(np.unique(sem_labels[pointids])) == 1)

    if os.path.exists(dump_filename):
        print("------> Skipping: {} is already done".format(fn))
        return None

    torch.save((coords, colors, sem_labels, instance_labels), dump_filename)
    print('Saving to ' + dump_filename)

# for fn in files:
#     f(fn)

p = mp.Pool(processes=1)#mp.cpu_count())
if opt.data_split == 'test':
    p.map(f_test, files)
else:
    p.map(f, files)
p.close()
p.join()
