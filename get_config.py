import random
import os
import torch

VERSION = '02'

def get_config():
    config = {
        'split_seed_list':[0],
        # 'FOLD_LIST':[0,1,2,3],
        'FOLD_LIST': [0],
        'num_folds': 4,
        'VERSION':VERSION,
        'save_indices_path': os.path.join(os.path.dirname(__file__), 'data_preparation', 'dataset', 'indexes_lists'),
        'INPUT_PATH': os.path.join(os.path.dirname(__file__), 'data_preparation', 'dataset', 'hubmap-organ-segmentation'),
        'train_data_path_list':[
            os.path.join(os.path.dirname(__file__), 'data_preparation', 'dataset', 'train_images') 
        ],
        
        'pretrain_path_list':None,
        'trn_idxs_list_path':None, 
        'val_idxs_list_path':None,
        
        'input_resolution':320,
        'resolution': 1024,
        'pad_size' : 256,
        'dice_threshold':0.5,
        'small_mask_threshold':0,
        'multiplier_bin':20,
        'binned_max':7,
        'tta':1,
        'trn_batch_size':4,
        'test_batch_size':4,        
        'Adam':{
            'lr':1e-4,
            # 'betas':(0.9, 0.999),
            # 'weight_decay':1e-5,
        },
        'SGD':{
            'lr':0.01,
            'momentum':0.9,
        },

        'lr_scheduler_name':'CosineAnnealingLR', #'OneCycleLR', #'ReduceLROnPlateau', #'StepLR',#'WarmUpLinearDecay', 

        'lr_scheduler':{
            'ReduceLROnPlateau':{
                'factor':0.8,
                'patience':5,
                'min_lr':1e-5,
                'verbose':True,
            },
            'OneCycleLR':{
                'pct_start':0.1,
                'div_factor':1e3, 
                'max_lr':1e-2,
                'epochs':25,
            },
            'CosineAnnealingLR':{
                'step_size_min':1e-6,
                't0':19,
                'tmult':1,
                'curr_epoch':-1,
                'last_epoch':-1,
            },
            'WarmUpLinearDecay':{
                'train_steps':40,
                'warm_up_step':3,
            },
            'StepLR':{
                'milestones':[1,2,3,20,40],
                'multipliers':[0.5,0.3,0.1,0.03,0.003],
            },
        },
        
        'num_epochs': 50,
        'early_stopping':True,
        'patience':5,
        'num_workers':0,
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    return config