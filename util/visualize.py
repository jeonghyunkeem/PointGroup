'''
Visualization
Written by Li Jiang
'''

import numpy as np
# import mayavi.mlab as mlab
import os, glob, argparse
import torch
from operator import itemgetter

# ->
from plyfile import PlyData, PlyElement
# <-


COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

COLOR40 = np.array(
        [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168], [75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]])

SEMANTIC_IDXS = np.array([-1,0,1,2,3,4,5,6,7,8,9,10,11,17,20,23,24,26,31,32,34])
SEMANTIC_IDXS = [sem_idx+1 for sem_idx in SEMANTIC_IDXS]
SEMANTIC_NAMES = np.array(['bathtub', 'bookshelf', 'cabinet', 'chair', 'display', 'sofa', 'table', 'trash bin', 'bag', 'basket', 
                        'bed', 'bench', 'keyboard', 'file cabinet', 'lamp', 'laptop', 'microwave', 'printer', 'stove', 'washer'])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'bathtub': [143, 223, 142],
    'bookshelf': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'chair': [255, 188, 126],
    'display': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'trash bin': [222, 40, 47],
    'bag': [197, 176, 212],
    'basket': [150, 103, 185],
    'bed': [200, 156, 149],
    'bench': [0, 190, 206],
    'keyboard': [252, 183, 210],
    'file cabinet': [219, 219, 146],
    'lamp': [255, 127, 43],
    'laptop': [234, 119, 192],
    'microwave': [150, 218, 228],
    'printer': [0, 160, 55],
    'stove': [110, 128, 143],
    'washer': [80, 83, 160]
}
SEMANTIC_IDX2NAME = {sem_idx:SEMANTIC_NAMES[i] for i, sem_idx in enumerate(SEMANTIC_IDXS)}
SEMANTIC_IDX2NAME = {1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7: 'table', 8: 'door', 9: 'window', 10: 'bookshelf', 11: 'picture',
                12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refridgerator', 28: 'shower curtain', 33: 'toilet',  34: 'sink', 36: 'bathtub', 39: 'otherfurniture'}

def get_coords_color(opt):
    input_file = os.path.join(opt.data_root, opt.room_split, opt.room_name + '_inst.pth')
    assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
    if opt.room_split == 'test':
        xyz, rgb = torch.load(input_file)
    else:
        xyz, rgb, label, inst_label = torch.load(input_file)
    rgb = (rgb + 1) * 127.5

    if (opt.task == 'semantic_gt'):
        assert opt.room_split != 'test'
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (opt.task == 'instance_gt'):
        assert opt.room_split != 'test'
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

    elif (opt.task == 'semantic_pred'):
        assert opt.room_split != 'train'
        semantic_file = os.path.join(opt.result_root, opt.room_split, 'semantic', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif (opt.task == 'instance_pred'):
        assert opt.room_split != 'train'
        instance_file = os.path.join(opt.result_root, opt.room_split, opt.room_name + '.txt')
        assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #
        for i in range(len(masks) - 1, -1, -1):
            mask_path = os.path.join(opt.result_root, opt.room_split, masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.09):
                continue
            mask = np.loadtxt(mask_path).astype(np.int)
            print('{} {}: {} pointnum: {}'.format(i, masks[i], SEMANTIC_IDX2NAME[int(masks[i][1])], mask.sum()))
            inst_label_pred_rgb[mask == 1] = COLOR20[i % len(COLOR20)]
        rgb = inst_label_pred_rgb

    if opt.room_split != 'test':
        sem_valid = (label != -1)
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb

def write_ply_color(points, colors, filename):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    N = points.shape[0]
    vertex = []
    for i in range(N):
        c = colors[i]
        vertex.append((points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='path to the input dataset files', default='../dataset/scan2cad')
    parser.add_argument('--result_root', help='path to the predicted results', default='exp/scan2cad/pointgroup/pointgroup_run3_scan2cad/result/epoch500_nmst0.3_scoret0.09_npointt100')
    parser.add_argument('--dump_root', help='', default='exp/scan2cad/pointgroup/pointgroup_run3_scan2cad/dump')
    parser.add_argument('--room_name', help='room_name', default='scene0000_00')
    parser.add_argument('--room_split', help='train / val / test', default='val')
    parser.add_argument('--task', help='input / semantic_gt / semantic_pred / instance_gt / instance_pred', default='instance_gt')
    opt = parser.parse_args()

    print(opt.room_name)

    xyz, rgb = get_coords_color(opt)
    if not os.path.exists(opt.dump_root):
        os.mkdir(opt.dump_root)
    write_ply_color(xyz, rgb, os.path.join(opt.dump_root, opt.room_name))

    # fig = mlab.figure(figure=None, bgcolor=(1.0, 1.0, 1.0), size=((800, 800)))
    # visualize_pts_rgb(fig, xyz, rgb)
    # mlab.show()

