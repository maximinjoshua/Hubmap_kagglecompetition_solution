import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from get_config import get_config
config = get_config()

def get_fold_idxs_list(data_df, val_patient_numbers_list):
    # print('val_pati', val_patient_numbers_list)
    trn_idxs_list = []
    val_idxs_list = []
    for fold in range(len(val_patient_numbers_list)):
        trn_idxs = data_df[~data_df['image_name'].isin(val_patient_numbers_list[fold])].index.tolist()
        trn_idxs_list.append(np.array(trn_idxs))
        val_idxs = data_df[data_df['image_name'].isin(val_patient_numbers_list[fold])].index.tolist()
        val_idxs_list.append(np.array(val_idxs))
        print('inside_get_fold_indxs_train', len(trn_idxs))
        print('inside_get_fold_indxs_val', len(val_idxs))
    return trn_idxs_list, val_idxs_list