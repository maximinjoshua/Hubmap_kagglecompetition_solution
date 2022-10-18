from cgitb import enable
from statistics import mean
import time
import pandas as pd
import numpy as np
import gc
from os.path import join as opj
import pickle
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import HuBMAPDatasetTrain
# from models import build_model
# from scheduler import CosineLR
from torch.optim import lr_scheduler
from dicecoefficient import mean_dice_coef
import wandb
from model import *
import os

wandb.init()

wandb.login()

torch.autograd.set_detect_anomaly(True)

checkpoints_path = os.path.join(os.path.dirname(__file__), 'checkpoints/')
os.makedirs(checkpoints_path, exist_ok= True)

from get_config import get_config
config = get_config()

def do_valid(net, valid_loader):

    valid_num = 0
    valid_probability = []
    valid_mask = []
    valid_batch_loss = 0

    net = net.eval()
    start_timer = time.time()
    for t, batch in enumerate(valid_loader):
        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            batch['img'] = batch['img'].cuda()
            batch['mask' ] = batch['mask' ].to(torch.float32).cuda()
            output = net(batch)
            loss0  = output['bce_loss'].mean()

        valid_probability.append(output['probability'].data.cpu().numpy())
        valid_mask.append(batch['mask'].data.cpu().numpy())
        valid_num += batch['img'].size(0)
        valid_batch_loss += batch['img'].size(0)*loss0.item()
        wandb.log({'validation_batch_loss_bce_only': valid_batch_loss})
    valid_loss = valid_batch_loss/valid_num
    validation_metrics = {'valid_loss_bce_only':valid_loss}
    wandb.log(validation_metrics)
    print('valid_bce_loss', valid_loss)
    probability = np.concatenate(valid_probability)
    mask = np.concatenate(valid_mask)
    dice = mean_dice_coef(probability, mask)

    dice = dice.mean()
    print('dice_score', dice)
    wandb.log({'dice_score_per_epoch_in_valid': dice})
    return valid_loss, dice

def run(seed, data_df, pseudo_df, trn_idxs_list, val_idxs_list):

    log_cols = ['fold', 'epoch', 'lr',
            'loss_trn', 'loss_val',
            'trn_score', 'val_score']

    for fold, (trn_idxs, val_idxs) in zip(config['FOLD_LIST'], zip(trn_idxs_list, val_idxs_list)):

        trn_df = data_df.iloc[trn_idxs].reset_index(drop=True)
        val_df = data_df.iloc[val_idxs].reset_index(drop=True)

        log_df = pd.DataFrame(columns=log_cols, dtype=object)
        log_counter = 0

        valid_dataset = HuBMAPDatasetTrain(val_df, config, mode='valid')
        valid_loader  = DataLoader(valid_dataset, batch_size=config['test_batch_size'],
                                shuffle=False, num_workers=0, pin_memory=True)

        net = Net().cuda()
        net.load_pretrain()
                                        
        optimizer = optim.Adam(net.parameters(), lr = 1e-4)
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, eta_min=1e-6)

        val_score_best  = -1e+99    
        val_loss_best = 1e+99
        counter_es = 0

        for epoch in range(1, config['num_epochs']+1):

            trn_df['binned'] = trn_df['binned'].apply(lambda x:config['binned_max'] if x>=config['binned_max'] else x)
            n_sample = trn_df['is_masked'].value_counts().min()
            trn_df_0 = trn_df[trn_df['is_masked']==False].sample(n_sample, replace=False)
            trn_df_1 = trn_df[trn_df['is_masked']==True].sample(n_sample, replace=False)
            n_bin = int(trn_df_1['binned'].value_counts().mean())
            trn_df_list = []
            for bin_size in trn_df_1['binned'].unique():
                trn_df_list.append(trn_df_1[trn_df_1['binned']==bin_size].sample(n_bin, replace=True))
            trn_df_1 = pd.concat(trn_df_list, axis=0)
            trn_df_balanced = pd.concat([trn_df_1, trn_df_0], axis=0).reset_index(drop=True)

            train_dataset = HuBMAPDatasetTrain(trn_df_balanced, config, mode='train')
            train_loader  = DataLoader(train_dataset, batch_size=config['trn_batch_size'],
                                            shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
            for param_group in optimizer.param_groups:
                print("LR %.9f" %param_group['lr'])

            net.train()
            tk0 = tqdm(train_loader, total=int(len(train_loader)))
            num_samples = 0
            batch_loss = 0
            train_probability = []
            train_masks = []
            for i,data in enumerate(tk0):
                data['img'] = data['img'].cuda()
                data['mask'] = data['mask'].to(torch.float32).cuda()
                optimizer.zero_grad()
                batch,c,h,w = data['img'].shape
                output = net(data)
                loss0  = output['bce_loss'].mean()
                loss1  = output['aux2_loss'].mean()
                total_loss = loss0+0.2*loss1
                total_loss.backward()
                optimizer.step()
                train_probability.append(output['probability'].data.cpu().numpy())
                train_masks.append(data['mask'].data.cpu().numpy())
                num_samples += data['img'].size(0)
                batch_loss += total_loss * data['img'].size(0)
                wandb.log({'train_batch_loss':batch_loss})
            scheduler.step()
            train_loss = batch_loss/num_samples
            train_loss = train_loss.data.cpu().numpy()
            print('train loss : ', train_loss)
            train_metrics = {'train_loss': train_loss}
            wandb.log(train_metrics)
            probability = np.concatenate(train_probability)
            masks = np.concatenate(train_masks)
            train_dice_score = mean_dice_coef(probability, masks)
            wandb.log({'train_dice_score': train_dice_score})
            validation_results = do_valid(net, valid_loader)
            if config['early_stopping']:
                if validation_results[0]< val_loss_best:
                    val_loss_best = validation_results[0]
                    counter_es =0
                    torch.save(net.state_dict(), checkpoints_path + f'fold{fold}_epoch{epoch}_bestloss.pth')
                else:
                    counter_es += 1
                
                    if counter_es > config['patience']:
                        print(f'Early stopping in action with best score {val_score_best} and best loss {val_loss_best}')
                        break

            if validation_results[1]> val_score_best:
                val_score_best = validation_results[1]
                torch.save(net.state_dict, checkpoints_path + f'fold{fold}_epoch{epoch}_bestscore.pth')

            log_df.loc[log_counter,log_cols] = np.array([fold, epoch,
                                                            [ group['lr'] for group in optimizer.param_groups ],
                                                            train_loss, validation_results[0], 
                                                            train_dice_score, validation_results[1],
                                                                ], dtype='object')
            log_counter += 1
        log_df.to_csv(checkpoints_path + 'log.csv', index = False)

        print(f"end of fold {fold}")