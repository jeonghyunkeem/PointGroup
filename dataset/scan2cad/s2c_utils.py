import math
from numpy.random import choice
import quaternion
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh
from matplotlib import pyplot

# ======================================================
# FILE I/O
# ======================================================
def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_instances(points, labels, filename, num_instances=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_instances is None:
        num_instances = np.max(labels)+1
    else:
        assert(num_instances>np.max(labels))
    
    vertex = []
    colors = [colormap(i/float(num_instances)) for i in range(num_instances)]    
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x*255) for x in c]
        vertex.append((points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]) )
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)

def write_bbox(scene_bbox, out_filename):
    """Export scene bbox to meshes
    Args:
        scene_bbox: (N x 6 numpy array): xyz pos of center and 3 lengths
        out_filename: (string) filename

    Note:
        To visualize the boxes in MeshLab.
        1. Select the objects (the boxes)
        2. Filters -> Polygon and Quad Mesh -> Turn into Quad-Dominant Mesh
        3. Select Wireframe view.
    """
    def convert_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    mesh_list.export(out_filename, file_type='ply')

def write_scene_results(points, ins_points, num_instances, bboxes, file_path):
    """
    1. scene point cloud
    2. instance-colored point cloud
    3. axis-aligned bounding boxes
    """
    N = points.shape[0]
    K = bboxes.shape[0]
    instance_labels = ins_points.astype(int)
    
    # 1. scene points
    scene_ply_fname = file_path+'/scene.ply'
    write_ply(points, scene_ply_fname)

    # 2. scene points with colored instances
    instances_ply_fname = file_path+'/instances.ply'
    scene_verts = []
    colormap = pyplot.cm.jet
    colors = [colormap(i/float(num_instances+1)) for i in range(num_instances+1)]   # (0~K) NOTE: 0 is unannotated
    colors[0] = tuple([0.0, 0.0, 0.0, 1.0])
    for p in range(N):
        c = colors[instance_labels[p]]
        c = [int(x*255) for x in c]
        scene_verts.append((points[p,0],points[p,1],points[p,2],c[0],c[1],c[2]))
    scene_verts = np.array(scene_verts, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    scene_elements = PlyElement.describe(scene_verts, 'vertex', comments=['vertices'])
    PlyData([scene_elements], text=True).write(instances_ply_fname)

    # 3. bounding boxes
    bboxes_fname = file_path+'/bbox.ply'
    def convert_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt
    bbox_scene = trimesh.scene.Scene()
    for k in range(K):
        bbox_scene.add_geometry(convert_box_to_trimesh_fmt(bboxes[k]))
    bbox_list = trimesh.util.concatenate(bbox_scene.dump())
    bbox_list.export(bboxes_fname, file_type='ply')


def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
     
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
     
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
     
    return roll_x, pitch_y, yaw_z # in radians


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


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2


def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) 

        Order of indices:
           (3)-----(2)
          / |     / |
        (4)-+---(1) |
         | (7)---+-(6)
         | /     | / 
        (8)-----(5)

        -: l (x)
        |: h (y)
        /: w (z)
    '''

    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])


def rotate_aligned_boxes(centers, lengths, rot_mat):    
    # centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
           
    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]

    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)    
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)
                  
    return new_centers, new_lengths

def filter_dominant_cls(points, obj_inds, sem_cls, target_cls, not_cared_id, threshold=20):
    '''
        args:
            points: (N, 3) NOTE: scene-level
            obj_inds: (M,)
            sem_cls: (N, 1)
            target_cls: 1
            not_cared_id: array
        
        returns:
            dom_points: (M, 3)
    '''
    # Find **target** cls points
    # NOTE: sem_cls ~ (1,35)
    cls_list = np.unique(sem_cls[obj_inds]).tolist()
    choices = []
    if target_cls in cls_list:
        sem_cls_indices = np.where(sem_cls==target_cls)[0]
        for i in sem_cls_indices:
            if obj_inds[i] == True:
                choices.append(i)
        if len(choices) > threshold:
            return points[choices], choices
    cls_book = {sem_cls: 0 for sem_cls in cls_list}
    ind_book = {sem_cls: [] for sem_cls in cls_list}
    for i_cls in cls_list:
        if i_cls in not_cared_id:
            continue
        sem_cls_indices = np.where(sem_cls==i_cls)[0]
        for i in sem_cls_indices:
            if obj_inds[i] == True:
                cls_book[i_cls] += 1
                ind_book[i_cls].append(i)
    
    sort_list = [sem_cls for sem_cls, value in sorted(cls_book.items(), key=lambda item: item[1], reverse=True)]
    dom_cls = sort_list[0]
    choices = ind_book[dom_cls]
    return points[choices], choices
    
def get_3d_box_rotated(box_size, rot_mat, padding=None):
    ''' @J. Kim, KAIST

        box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d

        Order of indices:
           (3)-----(2)
          / |     / |
        (4)-+---(1) | 
         | (7)---+-(6)
         | /     | / 
        (8)-----(5) 

        -: l (x)
        |: h (y)
        /: w (z)

        args:
            box_size: float (3)
            rot_mat: float (4, 4)
        
        returns:
            corners_3d: float (8, 3)
    '''
    R = rot_mat # (4, 4)
    l,h,w = box_size * 2
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    hcoord    = np.ones(8, dtype=np.float32)
    corners_3d = np.vstack([x_corners,y_corners,z_corners, hcoord])
    
    if padding:
        p = padding
        corners_3d[0:1, :] += [p, p, -p, -p, p, p, -p, -p]
        corners_3d[1:2, :] += [p, p, p, p, -p, -p, -p, -p]
        corners_3d[2:3, :] += [p, -p, -p, p, p, -p, -p, p]

    corners_3d = np.dot(R, corners_3d)
    corners_3d = np.transpose(corners_3d)
    corners_3d = corners_3d[:, :3]
    assert(corners_3d.shape[0] * corners_3d.shape[1] == 24)
    
    return corners_3d


def batch_get_3d_box_rotated(box_size, rot_mat):
    ''' box_size: [x1,x2,...,xn,3]          => (B, 3)
        heading_angle: [x1,x2,...,xn,4,4]   => (B, 4, 4)
        center: [x1,x2,...,xn,3]            => (B, 3)
    Return:
        [x1,x3,...,xn,8,3]                  => (B, 8, 3)
    '''
    input_shape = box_size.shape[0] # B
    R = rot_mat # (B, 4, 4)
    l = np.expand_dims(box_size[...,0], -1) # [x1,...,xn,1]
    w = np.expand_dims(box_size[...,1], -1)
    h = np.expand_dims(box_size[...,2], -1)
    corners_3d = np.zeros(tuple(list(input_shape)+[8,3]))
    corners_3d[...,:,0] = np.concatenate((l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2, 1), -1)
    corners_3d[...,:,1] = np.concatenate((h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2, 1), -1)
    corners_3d[...,:,2] = np.concatenate((w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2, 1), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape)+1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    assert(corners_3d.shape[1] * corners_3d.shape[2] == 24)
    return corners_3d
    
    
def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    
    return corners_3d

def nms_3d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    area = (x2-x1)*(y2-y1)*(z2-z1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])

        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type:
            o = (l*w*h)/area[I[:last-1]]
        else:
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0])))

    return pick

def nms_3d_faster_samecls(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    cls = boxes[:,7]
    area = (x2-x1)*(y2-y1)*(z2-z1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])
        cls1 = cls[i]
        cls2 = cls[I[:last-1]]

        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type:
            o = (l*w*h)/area[I[:last-1]]
        else:
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)
        o = o * (cls1==cls2)

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0])))

    return pick
