import sys
sys.path.insert(0, '../')
import warnings
warnings.simplefilter('ignore')
from utils import fix_seed
from get_config import get_config
from get_fold_idxs_list import get_fold_idxs_list
from run_only import run
import pickle

import numpy as np
import pandas as pd
import os
from os.path import join as opj


if __name__=='__main__':
    # config
    fix_seed(2021)
    config = get_config()
    FOLD_LIST = config['FOLD_LIST']
    VERSION = config['VERSION']
    INPUT_PATH = config['INPUT_PATH']
    OUTPUT_PATH = config['OUTPUT_PATH']
    # os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = config['device']
    print(device)
    
    # import data 
    train_df = pd.read_csv(opj(INPUT_PATH, 'train.csv'))
    # info_df  = pd.read_csv(opj(INPUT_PATH,'HuBMAP-20-dataset_information.csv'))
    # sub_df = pd.read_csv(opj(INPUT_PATH, 'sample_submission.csv'))
    # print('train_df.shape = ', train_df.shape)
    # print('info_df.shape  = ', info_df.shape)
    # print('sub_df.shape = ', sub_df.shape)
    # print(train_df['organ'].value_counts())
    
    # dataset
    data_df = []
    for data_path in config['train_data_path_list']:
        _data_df = pd.read_csv(opj(data_path,'data.csv'))
        _data_df['data_path'] = data_path
        data_df.append(_data_df)
        # print(data_path)
    data_df = pd.concat(data_df, axis=0).reset_index(drop=True)

    # print('data_df.shape = ', data_df.shape)
    data_df = data_df[data_df['std_img']>10].reset_index(drop=True)
    # print('data_df.shape = ', data_df.shape)
    data_df['binned'] = np.round(data_df['ratio_masked_area'] * config['multiplier_bin']).astype(int)
    data_df['is_masked'] = data_df['binned']>0
    # print(data_df['binned'].value_counts())
    # data_df.to_csv('D:/coat/data_df_visualize_binned.csv')

    trn_df = data_df.copy()
    trn_df['binned'] = trn_df['binned'].apply(lambda x:config['binned_max'] if x>=config['binned_max'] else x)
    trn_df_1 = trn_df[trn_df['is_masked']==True]
    # print(trn_df['is_masked'].value_counts())
    # print(trn_df_1['binned'].value_counts())
    # print('mean = ', int(trn_df_1['binned'].value_counts().mean()))

    # print(data_df.info())
    data_df['image_name'] = data_df['filename_img'].apply(lambda x: int(x.split('_')[0]))
    # data_df.to_csv('D:/new_hubmap_with_toms_solution/data_df.csv')
    # validation_indices_dict = {'lung': [], 'prostrate': [], 'spleen': [], 'largeintestine': [], 'kidney':[]}
    validation_indices_dict = {'fold0': None, 'fold1': None, 'fold2' : None, 'fold3': None}
    lung_val_data = train_df[train_df['organ']== 'lung'].sample(int(len(train_df[train_df['organ'] == 'lung'])*0.2)*config['num_folds'], replace=False)['id'].tolist()
    spleen_val_data = train_df[train_df['organ']== 'spleen'].sample(int(len(train_df[train_df['organ'] == 'spleen'])*0.2)*config['num_folds'], replace=False)['id'].tolist()
    prostate_val_data = train_df[train_df['organ']== 'prostate'].sample(int(len(train_df[train_df['organ'] == 'prostate'])*0.2)*config['num_folds'], replace=False)['id'].tolist()
    largeintestine_val_data = train_df[train_df['organ']== 'largeintestine'].sample(int(len(train_df[train_df['organ'] == 'largeintestine'])*0.2)*config['num_folds'], replace=False)['id'].tolist()
    kidney_val_data = train_df[train_df['organ']== 'kidney'].sample(int(len(train_df[train_df['organ'] == 'kidney'])*0.2)*config['num_folds'], replace=False)['id'].tolist()
    # print(len(kidney_val_data), len(largeintestine_val_data), len(prostate_val_data), len(spleen_val_data), len(lung_val_data))
    # for fold in validation_indices_list:
    lung_indices = [lung_val_data[i:i+int(len(lung_val_data)/4)] for i in range(0,len(lung_val_data),int(len(lung_val_data)/4))]
    spleen_indices = [spleen_val_data[i:i+int(len(spleen_val_data)/4)] for i in range(0,len(spleen_val_data),int(len(spleen_val_data)/4))]
    prostate_indices = [prostate_val_data[i:i+int(len(prostate_val_data)/4)] for i in range(0,len(prostate_val_data),int(len(prostate_val_data)/4))]
    largeintestine_indices = [largeintestine_val_data[i: i+int(len(largeintestine_val_data)/4)] for i in range(0,len(largeintestine_val_data),int(len(largeintestine_val_data)/4))]
    kidney_indices = [kidney_val_data[i:i+int(len(kidney_val_data)/4)] for i in range(0,len(kidney_val_data),int(len(kidney_val_data)/4))]
    # print(len(lung_indices), len(spleen_indices), len(prostate_indices), len(largeintestine_indices), len(kidney_indices))
    # print(lung_indices)
    # print(spleen_indices)
    # print(prostate_indices)
    # print(largeintestine_indices)
    # print(kidney_indices)

    for i, fold in enumerate(validation_indices_dict):
        validation_indices_dict[fold] = lung_indices[i] + spleen_indices[i] + prostate_indices[i] + largeintestine_indices[i] \
            + kidney_indices[i]
    with open('D:/coat/segformer_henck/archive/save_dataset_folder/validation_indices_list', 'wb') as file:
        pickle.dump(validation_indices_dict, file)
    # print('datadf_imagename_unique', data_df['image_name'].unique(), len(data_df['image_name'].unique()))
    # for i in validation_indices_dict:
        # print('lund',len(validation_indices_dict[i]))

    
    
    # info_df['image_name'] = info_df['image_file'].apply(lambda x:x.split('.')[0])
    # patient_mapper = {}
    # for (x,y) in info_df[['image_name','patient_number']].values:
    #     patient_mapper[x] = y
    # data_df['patient_number'] = data_df['filename_img'].apply(lambda x:patient_mapper[x.split('_')[0]])
    
    # val_patient_numbers_list = [
    #     [63921], # fold0
    #     [68250], # fold1
    #     [65631], # fold2
    #     [67177], # fold3
    # ]
        
    val_patient_numbers_list = [
        validation_indices_dict['fold0'], # fold0
        validation_indices_dict['fold1'], # fold1
        validation_indices_dict['fold2'], # fold2
        validation_indices_dict['fold3'], # fold3
    ]
    
    # # train
    for seed in config['split_seed_list']:
        trn_idxs_list, val_idxs_list = get_fold_idxs_list(data_df, val_patient_numbers_list)
        # # print('shape shape',val_idxs_list)
        with open(opj(config['OUTPUT_PATH'],f'trn_idxs_list_seed{seed}'), 'wb') as f:
        # # with open(f'trn_idxs_list_seed{seed}', 'wb') as f:
            pickle.dump(trn_idxs_list, f)
        with open(opj(config['OUTPUT_PATH'],f'val_idxs_list_seed{seed}'), 'wb') as f:
        # # with open(f'val_idxs_list_seed{seed}', 'wb') as f:
            pickle.dump(val_idxs_list, f)
        run(seed, data_df, None, trn_idxs_list, val_idxs_list)