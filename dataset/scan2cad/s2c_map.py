# Jeonghyun Kim, KAIST 

""" 
    Mapping dictionary of ShapeNetCore category id to Scan2CAD category name

    ALL_CATEGORY: a dictonary of {cat_id : cat_name} used for Scan2CAD annotation
    NAME2CLASS: a dictionary of {cat_name: cat_class}
            0 ~ 7   : cared cat_ids
            8 ~ 35  : others

    * CARED_LIST: a list of cared cat_ids in Scan2CAD benchmark 
"""

ID2NAME = {
    # 35 categories / Mapping from ID to NAME
    '02747177': 'trash bin',
    '02773838': 'bag',
    '02801938': 'basket',
    '02808440': 'bathtub', 
    '02818832': 'bed',
    '02828884': 'bench',
    '02871439': 'bookshelf',
    '02876657': 'bottle', 
    '02880940': 'bowl',
    '02933112': 'cabinet',
    '02946921': 'can', 
    '02954340': 'cap',
    '03001627': 'chair',
    '03046257': 'clock',
    '03085013': 'keyboard',
    '03207941': 'dishwasher',
    '03211117': 'display',
    '03325088': 'faucet',
    '03337140': 'file cabinet', 
    '03467517': 'guitar', 
    '03593526': 'jar', 
    '03636649': 'lamp',
    '03642806': 'laptop',
    '03691459': 'speaker',
    '03761084': 'microwave',
    '03790512': 'motorcycle', 
    '03928116': 'piano', 
    '03938244': 'pillow', 
    '03991062': 'pot', 
    '04004475': 'printer',
    '04256520': 'sofa',
    '04330267': 'stove',
    '04379243': 'table',
    '04401088': 'telephone', 
    '04554684': 'washer'
} 

NAME2CLASS = {
    # ===== CARED =====
    # Top 8 categories of Scan2CAD
    'bathtub': 0,
    'bookshelf': 1,
    'cabinet': 2, 
    'chair': 3, 
    'display': 4,
    'sofa': 5,
    'table': 6, 
    'trash bin': 7, 
    # ===== OTHERS =====    
    'bag': 8,
    'basket': 9,
    'bed': 10,
    'bench': 11,
    'bottle': 12,#0 
    'bowl': 13, #0
    'can': 14, #0
    'cap': 15, #0
    'clock': 16,
    'keyboard': 17,
    'dishwasher': 18,
    'faucet': 19, #0
    'file cabinet': 20, 
    'guitar': 21, 
    'jar': 22, #0
    'lamp': 23,
    'laptop': 24,
    'speaker': 25, #0
    'microwave': 26,
    'motorcycle': 27, #0
    'piano': 28, 
    'pillow': 29, #0
    'pot': 30, 
    'printer': 31,
    'stove': 32,
    'telephone': 33, #0 
    'washer': 34
}

NONE_CLASS_MASK = [12,13,14,15,19,22,25,27,29,33]
CARED_CLASS_MASK = [0,1,2,3,4,5,6,7,8,9,10,11,17,20,23,24,26,31,32,34]
assert(len(CARED_CLASS_MASK)==20)

CLASS_MAPPING = {
    0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 
    10:10, 11:11, 16:12, 17:13, 18:14, 20:15, 21:16, 23:17, 24:18, 26:19, 28:20, 
    30:21, 31:22, 32:23, 34:24,
    12:None,13:None,14:None,15:None,19:None,22:None,25:None,27:None,29:None,33:None
}

