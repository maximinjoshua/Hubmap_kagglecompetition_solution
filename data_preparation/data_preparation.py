import sys
sys.path.insert(0, '../../')
import warnings
warnings.simplefilter('ignore')
from utils import fix_seed
from utils_data_generation import HuBMAPDataset, my_collate_fn, generate_data

import random
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import os
from os.path import join as opj
import gc

def get_config():
    config = {
        # 'OUTPUT_PATH':f'D:/new_hubmap_with_toms_solution/save_dataset_folder/shifted_512/',
        # 'INPUT_PATH':'D:/new_hubmap_with_toms_solution/inputs_folder/',
        'OUTPUT_PATH': os.path.join(os.path.dirname(__file__), 'dataset', 'train_images'),
        'INPUT_PATH':os.path.join(os.path.dirname(__file__), 'dataset', 'hubmap-organ-segmentation'),
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'reduce': 1,
        'tile_size':1024,
        'batch_size':5,
        'shift_h': 512,
        'shift_w': 512,
    }
    return config
    
if __name__=='__main__':
    # config
    fix_seed(2021)
    config = get_config()
    INPUT_PATH = config['INPUT_PATH']
    OUTPUT_PATH = config['OUTPUT_PATH']
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = config['device']
    print(device)

    # import data
    train_df = pd.read_csv(opj(INPUT_PATH, 'train.csv'))
    print('train_df.shape = ', train_df.shape)
    
    shift_list = [0, 512]
    # generate training data
    data_list = []

    for i in shift_list:
        config['shift_h'] = i
        config['shift_w'] = i
        for idx,filename in enumerate(train_df['id'].values):
            print('idx = {}, {}'.format(idx,filename))
            ds = HuBMAPDataset(train_df, str(filename), config)
            #rasterio cannot be used with multiple workers
            dl = DataLoader(ds,batch_size=config['batch_size'],
                            num_workers=0,shuffle=False,pin_memory=True,
                            collate_fn=my_collate_fn)
            img_patches  = []
            mask_patches = []
            for data in tqdm(dl):
                img_patch = data['img']
                mask_patch = data['mask']
                img_patches.append(img_patch)
                mask_patches.append(mask_patch)
            img_patches  = np.vstack(img_patches)
            mask_patches = np.vstack(mask_patches)

            # sort by number of masked pixels
            bs,sz,sz,c = img_patches.shape
            idxs = np.argsort(mask_patches.reshape(bs,-1).sum(axis=1))[::-1]
            img_patches  = img_patches[idxs].reshape(-1,sz,sz,c)
            mask_patches = mask_patches[idxs].reshape(-1,sz,sz,1)
            # data = Parallel(n_jobs=-1)(delayed(generate_data)(filename, i, x, y, config) for i,(x,y) in enumerate(tqdm(zip(img_patches, mask_patches)))) 
            data = [generate_data(str(filename), i, x, y, config) for i,(x,y) in enumerate(tqdm(zip(img_patches, mask_patches)))]
            # if data != -1:
            data_list.append(data)
    
    # save
    data_df = pd.concat([pd.DataFrame(data_list[i]) for i in range(len(data_list))], axis=0).reset_index(drop=True)
    data_df.columns = ['filename_img', 'filename_rle', 'num_masked_pixels', 'ratio_masked_area', 'std_img']
    print(data_df.shape)
    # print(data_df)
    print(OUTPUT_PATH)
    data_df.to_csv(opj(OUTPUT_PATH, 'data.csv'), index=False)
    
    