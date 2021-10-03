# Jeonghyun Kim, UVR KAIST @jeonghyunct.kaist.ac.kr

import os, sys
import json
import h5py
import numpy as np
import quaternion
import torch
from torch.utils.data import Dataset

BASE_DIR_1 = os.path.dirname(os.path.abspath(__file__)) # scan2cad
BASE_DIR = os.path.dirname(BASE_DIR_1)  # dataset
ROOT_DIR = os.path.dirname(BASE_DIR)    # PointGroup
DATA_DIR = os.path.dirname(ROOT_DIR)    # /root/
DATA_DIR = os.path.join(DATA_DIR, 'Dataset')    # /root/Dataset
DUMP_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from s2c_map import CLASS_MAPPING, ID2NAME, CARED_CLASS_MASK
from s2c_config import Scan2CADDatasetConfig
import s2c_utils

sys.path.append(os.path.join(ROOT_DIR, 'models/retrieval/'))

DC = Scan2CADDatasetConfig()
MAX_NUM_POINT = 50000
MAX_NUM_OBJ = 64
INS_NUM_POINT = 2048
FEATURE_DIMENSION = 512

MAX_DATA_SIZE = 15000
CHUNK_SIZE = 1000

INF = 9999
NOT_CARED_ID = np.array([INF])    # wall, floor

# Thresholds
PADDING = 0.05
SCALE_THRASHOLD = 0.05
SEG_THRESHOLD = 1

REMAPPER = np.ones(35, dtype=np.int64) * (-1)
for i, x in enumerate(CARED_CLASS_MASK):
    REMAPPER[x] = i
    print(f'REMAPPER[{x:2d}] => {i:2d}')
SYM2CLASS = {"__SYM_NONE": 0, "__SYM_ROTATE_UP_2": 1, "__SYM_ROTATE_UP_4": 2, "__SYM_ROTATE_UP_INF": 3}

# functions ==============================================================================================
def from_q_to_6d(q):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    mat = quaternion.as_rotation_matrix(q)  # 3x3
    rep6d = mat[:, 0:2].transpose().reshape(-1, 6)   # 6
    return rep6d

def nn_search(p, ps):
    target = torch.from_numpy(ps.copy()) 
    p = torch.from_numpy(p.copy())
    p_diff = target - p
    p_dist = torch.sum(p_diff**2, dim=-1)
    dist, idx = torch.min(p_dist, dim=-1)
    return dist.item(), idx.item()

def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M 

def compose_mat4(t, q, s, center=None):
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M 

def decompose_mat4(M):
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:,0] /= sx
    R[:,1] /= sy
    R[:,2] /= sz

    q = quaternion.from_rotation_matrix(R[0:3, 0:3])

    t = M[0:3, 3]
    return t, q, s

# ========================================================================================================
LOG_N = 100
def print_log(log):
    print('-'*LOG_N+'\n'+log+' \n'+'-'*LOG_N)
        
class Scan2CADCollect(Dataset):
    def __init__(self, split_set='train', distr_check=False):
        self.data_path = os.path.join(DATA_DIR, 'Scan2CAD/export')
        self.out_path = os.path.join(BASE_DIR_1, 'data4')
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
            print("Create export directory: {}".format(self.out_path))

        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))

        self.scan_names = []
        if split_set in ['all', 'train', 'val', 'test']:
            split_filenames = os.path.join(BASE_DIR_1, 'meta_data',
                'scan2cad_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_list = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_list)
            self.scan_list = [sname for sname in self.scan_list \
                if sname in all_scan_names]
            print_log('Dataset for {}: kept {} scans out of {}'.format(split_set, len(self.scan_list), num_scans))
            num_scans = len(self.scan_list)
        else:
            print('illegal split name')
            return

        filename_json = BASE_DIR_1 + "/full_annotations.json"
        assert filename_json
        self.dataset = {}
        cat_summary = dict.fromkeys(DC.ClassToName, 0)
        cat_ids = []
        with open(filename_json, 'r') as f:
            data = json.load(f)
            d = {}
            i = -1
            for idx, r in enumerate(data):
                i_scan = r["id_scan"]
                if i_scan not in self.scan_list: 
                    continue
                self.scan_names.append(i_scan)
                i += 1
                d[i] = {}
                d[i]['id_scan'] = i_scan
                d[i]['trs'] = r["trs"]
                
                n_model = r["n_aligned_models"]
                d[i]['n_total'] = n_model
                d[i]['models'] = {}
                for j in range(n_model):
                    d[i]['models'][j] = {}
                    d[i]['models'][j]['trs'] = r["aligned_models"][j]['trs']
                    d[i]['models'][j]['center'] = r["aligned_models"][j]['center']
                    d[i]['models'][j]['bbox'] = r["aligned_models"][j]['bbox']
                    d[i]['models'][j]['sym'] = SYM2CLASS[r["aligned_models"][j]['sym']]
                    d[i]['models'][j]['fname'] = r["aligned_models"][j]['id_cad']
                    cat_id = r["aligned_models"][j]['catid_cad']
                    cat_ids.append(cat_id)
                    d[i]['models'][j]['cat_id'] = cat_id
                    cat_class = DC.ShapenetIDtoClass(cat_id) 
                    d[i]['models'][j]['sem_cls'] = cat_class
                    # category summary
                    cat_summary[cat_class]+=1  

        self.dataset = d
        self.cat_ids = np.unique(cat_ids)

        if distr_check:
            for k, v in sorted(cat_summary.items(), key=lambda item:item[1], reverse=True):
                print(f'{k:2d}: {DC.ClassToName[k]:12s} => {v:4d}')

    def __len__(self):
        return len(self.dataset)

    def size_check(self, scale, id_scan, sem_cls):
        check = False
        if scale[0] < SCALE_THRASHOLD:
            scale[0] = SCALE_THRASHOLD
            check = True
        if scale[1] < SCALE_THRASHOLD:
            scale[1] = SCALE_THRASHOLD
            check = True
        if scale[2] < SCALE_THRASHOLD:
            scale[2] = SCALE_THRASHOLD
            check = True

        return scale

    def collect(self, N, dump=False):
        """ Return dictionary of {verts(x,y,z): cad filename} 
            
            Note:
                NK = a total number of instances in dataset 
                V = a number of vertices
            
            args:
                N: int
                    a size of dataset
            
            return:
                dict: (NK, 1, V, 3)
                    a dictionary for verts-cad_file pairs
        """
        # ======= GLOBAL LABEL VARIABLES =======
        error_scan = {} # Text
        # Anchor collection (for detection)
        print_log(" LOADING SCENES")
        collect_path = os.path.join(BASE_DIR, 'collect')
        for index in range(N):
            data = self.dataset[index]
            id_scan = data['id_scan']

            K = data['n_total']
            assert(K <= MAX_NUM_OBJ)
            
            # Point Cloud
            mesh_vertices   = np.load(os.path.join(self.data_path, id_scan) + '_vert.npy')      # (N, 3)
            semantic_labels = np.load(os.path.join(self.data_path, id_scan) + '_sem_label.npy') # (N, sem_cls(0, 1~35, 36~MAX, INF))

            point_cloud = mesh_vertices[:,0:3]
            colors = mesh_vertices[:,3:6] / 127.5 - 1
            instance_vertices = np.ones((point_cloud.shape[0]), dtype=np.int64) * (-1)
            semantic_vertices = np.ones((point_cloud.shape[0]), dtype=np.int64) * (-1)

            # Sorting points cropping order to avoid overlapping 
            sort_by_scale = {}
            for model in range(K):
                obj_scale = np.array(data['models'][model]['trs']['scale'])
                sort_by_scale[model] = np.sum(obj_scale)
            model_scale_order = {model: scale for model, scale in sorted(sort_by_scale.items(), key=(lambda item:item[1]), reverse=True)}
            K = len(model_scale_order.keys())
            
            # Iterate on scale_order
            checked = False
            k = -1
            for i, model in enumerate(model_scale_order.keys()):
                k += 1
                # semantics ()
                sem_cls = data['models'][model]['sem_cls']    # (0~num_classes-1)

                # Transform
                obj_center = np.array(data['models'][model]['center'])
                obj_translation = np.array(data['models'][model]['trs']['translation'])
                obj_rotation = np.array(data['models'][model]['trs']['rotation'])
                obj_scale = np.array(data['models'][model]['trs']['scale'])
                obj_scale = self.size_check(obj_scale, id_scan, sem_cls)
                Mobj = compose_mat4(obj_translation, obj_rotation, obj_scale, obj_center)

                # Instance vertices
                # - (1) Region Crop & Axis-aligned Bounding Box
                vert_choices = np.array([])
                ins_bbox = np.array(data['models'][model]['bbox'])
                obj_corners = s2c_utils.get_3d_box_rotated(ins_bbox, Mobj, padding=PADDING)
                ex_points, obj_vert_ind = s2c_utils.extract_pc_in_box3d(point_cloud, obj_corners)
                nx = ex_points.shape[0]
                # - (2) Instance Segments Crop
                seg_points, vert_choices = \
                    s2c_utils.filter_dominant_cls(point_cloud, obj_vert_ind, semantic_labels, sem_cls+1, NOT_CARED_ID)
                seg_nx = seg_points.shape[0]

                # ======= Semantic/Instance vertices =======
                if seg_nx < SEG_THRESHOLD: 
                    k -= 1
                    checked = True
                    continue

                sem_cls = REMAPPER[sem_cls]
                # if sem_cls < 0: continue    # ignore non-valid class object (only preserve CARED classes)
                instance_vertices[vert_choices] = k # (0~K-1) NOTE:unannotated=-1                
                semantic_vertices[vert_choices] = sem_cls # (0~num_classes-1) NOTE:unannotated=-1

            # error check
            ins_list = np.unique(instance_vertices)
            if (np.max(instance_vertices)+1) != (len(ins_list)-1):
                print_log(f"[{index}/{N} Error] Please check this scene --> {id_scan}")
                error_scan[id_scan] = 0
                continue

            # DUMP COLLECT RESULTS
            if dump:
                scene_path = os.path.join(collect_path, f'{id_scan}')
                if not os.path.exists(scene_path):
                    os.mkdir(scene_path)
                    print("Created scene directory: {}".format(scene_path))
                s2c_utils.write_scene_results(points=point_cloud, ins_points=instance_vertices, num_instances=K, bboxes=None, file_path=scene_path)
            
            point_cloud = np.ascontiguousarray(point_cloud[:, :3] - point_cloud[:, :3].mean(0))
            pcoord = point_cloud.astype(np.float64)
            colors = colors.astype(np.float32)
            sem_labels = semantic_vertices.astype(np.float64)
            ins_labels = instance_vertices.astype(np.float64)

            # ============ DUMP ============
            # scene data
            file_path = os.path.join(self.out_path, id_scan+'_inst.pth')
            torch.save((pcoord, colors, sem_labels, ins_labels), file_path)
            print(f"[{index}/{N} Saved] {id_scan} >>> {file_path}")

        # error scan
        with open(self.out_path+'/error_scan.txt', 'w') as f:
            print_log("ERROR SCAN")
            for i, sname in enumerate(error_scan.keys()):
                print('{:2d}: {}'.format(i, sname))
                f.write(sname)
                f.write('\n')


if __name__ == "__main__":
    Dataset = Scan2CADCollect(split_set='all', distr_check=True)
    N = len(Dataset)    
    Dataset.collect(N, dump=False)