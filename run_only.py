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
from scheduler import CosineLR
from torch.optim import lr_scheduler
from utils import elapsed_time
from kaggle_hubmap_kv3 import compute_dice_score
from dicecoefficient import mean_dice_coef
# from lovasz_loss import lovasz_hinge
# from losses import criterion_lovasz_hinge_non_empty
# from metrics import dice_sum, dice_sum_2
import wandb
from model import *

# output_path = 'D:/coat/segformer_henck/archive/segformer-mit-b2/checkpoints/'

wandb.init()

wandb.login()

torch.autograd.set_detect_anomaly(True)

is_amp = False

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
            # with torch.cuda.amp.autocast(enabled = is_amp):
            # batch_size = len(batch['index'])
            batch['img'] = batch['img'].cuda()
            batch['mask' ] = batch['mask' ].to(torch.float32).cuda()
            # batch['organ'] = batch['organ'].cuda()

            # output = data_parallel(net, batch) #net(input)#
            output = net(batch)
            loss0  = output['bce_loss'].mean()

        # valid_probability.append(output['probability'].data.cpu().numpy())
        # valid_mask.append(batch['mask'].data.cpu().numpy())
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
    # print(probability)

    # dice = compute_dice_score(probability, mask)
    dice = mean_dice_coef(probability, mask)

    dice = dice.mean()
    print('dice_score', dice)
    wandb.log({'dice_score_per_epoch_in_valid': dice})
    # print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)
    return valid_loss, dice

def run(seed, data_df, pseudo_df, trn_idxs_list, val_idxs_list):

    log_cols = ['fold', 'epoch', 'lr',
            'loss_trn', 'loss_val',
            'trn_score', 'val_score']

    for fold, (trn_idxs, val_idxs) in zip(config['FOLD_LIST'], zip(trn_idxs_list, val_idxs_list)):

        trn_df = data_df.iloc[trn_idxs].reset_index(drop=True)
        val_df = data_df.iloc[val_idxs].reset_index(drop=True)
        # print(trn_df.info())
        # print(val_df.info())

        log_df = pd.DataFrame(columns=log_cols, dtype=object)
        log_counter = 0

        valid_dataset = HuBMAPDatasetTrain(val_df, config, mode='valid')
        valid_loader  = DataLoader(valid_dataset, batch_size=config['test_batch_size'],
                                shuffle=False, num_workers=0, pin_memory=True)

        net = Net().cuda()
        net.load_pretrain()
                                        
        optimizer = optim.Adam(net.parameters(), lr = 1e-4)
        
        # if config['lr_scheduler_name']=='ReduceLROnPlateau':
        #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['lr_scheduler']['ReduceLROnPlateau'])
        # elif config['lr_scheduler_name']=='CosineAnnealingLR':
        #     #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['lr_scheduler']['CosineAnnealingLR'])
        #     scheduler = CosineLR(optimizer, **config['lr_scheduler']['CosineAnnealingLR'])
        # elif config['lr_scheduler_name']=='OneCycleLR':
        #     scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_loader),
        #                                                 **config['lr_scheduler']['OneCycleLR'])
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, eta_min=1e-6)

        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min= 1e-6)

        # scaler = torch.cuda.amp.GradScaler(enabled = is_amp)

        val_score_best  = -1e+99    
        val_loss_best = 1e+99
        counter_es = 0

        for epoch in range(1, config['num_epochs']+1):

            trn_df['binned'] = trn_df['binned'].apply(lambda x:config['binned_max'] if x>=config['binned_max'] else x)
            n_sample = trn_df['is_masked'].value_counts().min()
            # print('n_sample',n_sample)
            trn_df_0 = trn_df[trn_df['is_masked']==False].sample(n_sample, replace=False)
            trn_df_1 = trn_df[trn_df['is_masked']==True].sample(n_sample, replace=False)
            # print('len(trn_df_1', len(trn_df_1))
            n_bin = int(trn_df_1['binned'].value_counts().mean())
            # print('n_bin', n_bin)
            trn_df_list = []
            for bin_size in trn_df_1['binned'].unique():
                trn_df_list.append(trn_df_1[trn_df_1['binned']==bin_size].sample(n_bin, replace=True))
            trn_df_1 = pd.concat(trn_df_list, axis=0)
            # print('after len(trn_df_1)',len(trn_df_1))
            # print('len(trn_df_0)',len(trn_df_0))
            trn_df_balanced = pd.concat([trn_df_1, trn_df_0], axis=0).reset_index(drop=True)

            train_dataset = HuBMAPDatasetTrain(trn_df_balanced, config, mode='train')
            train_loader  = DataLoader(train_dataset, batch_size=config['trn_batch_size'],
                                            shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
            # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))
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
                # data['label'] = data['label'].cuda()
                optimizer.zero_grad()
                # with torch.cuda.amp.autocast(enabled = is_amp):
                batch,c,h,w = data['img'].shape
                output = net(data)
                loss0  = output['bce_loss'].mean()
                loss1  = output['aux2_loss'].mean()
                total_loss = loss0+0.2*loss1
                # scaler.scale(loss0+0.2*loss1).backward()
                total_loss.backward()
				
                # scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                # scaler.step(optimizer)
                optimizer.step()
                # scaler.update()
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
                    # torch.save(net.state_dict(), output_path + f'/fold{fold}_epoch{epoch}_bestloss.pth')
                    torch.save(net.state_dict(), f'fold{fold}_epoch{epoch}_bestloss.pth')
                else:
                    counter_es += 1
                
                    if counter_es > config['patience']:
                        print(f'Early stopping in action with best score {val_score_best} and best loss {val_loss_best}')
                        break

            if validation_results[1]> val_score_best:
                val_score_best = validation_results[1]
                # torch.save(net.state_dict, output_path + f'/fold{fold}_epoch{epoch}_bestscore.pth')
                torch.save(net.state_dict, f'fold{fold}_epoch{epoch}_bestscore.pth')

            log_df.loc[log_counter,log_cols] = np.array([fold, epoch,
                                                            [ group['lr'] for group in optimizer.param_groups ],
                                                            train_loss, validation_results[0], 
                                                            train_dice_score, validation_results[1],
                                                                ], dtype='object')
            log_counter += 1
        log_df.to_csv('D:/coat/segformer_henck/archive/segformer-mit-b2/log/log.csv', index = False)

        print(f"end of fold {fold}")


        # if initial_checkpoint is not None:
        #     f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        #     start_iteration = f['iteration']
        #     start_epoch = f['epoch']
        #     state_dict  = f['state_dict']
        #     net.load_state_dict(state_dict,strict=False)  #True

        # else:
        #     start_iteration = 0
        #     start_epoch = 0
        #     net.load_pretrain()
            

        # valid_loss = np.zeros(4,np.float32)
        # train_loss = np.zeros(2,np.float32)
        # batch_loss = np.zeros_like(train_loss)
        # sum_train_loss = np.zeros_like(train_loss)
        # sum_train = 0

        # start_timer = time.time()
        # iteration = start_iteration
        # epoch = start_epoch
        # rate = 0
        # while iteration < num_iteration:
        #     for t, batch in enumerate(train_loader):
                
        #         if iteration%iter_save==0:
        #             if iteration != start_iteration:
        #                 torch.save({
        #                     'state_dict': net.state_dict(),
        #                     'iteration': iteration,
        #                     'epoch': epoch,
        #                 }, out_dir + '/checkpoint/%08d.model.pth' %  (iteration))
        #                 pass
                
                
        #         if (iteration%iter_valid==0): # or (t==len(train_loader)-1):
        #             #if iteration!=start_iteration:
        #             valid_loss = do_valid(net, valid_loader)  #
        #             pass
                
                
        #         if (iteration%iter_log==0) or (iteration%iter_valid==0):
        #             print('\r', end='', flush=True)
        #             # log.write(message(mode='log') + '\n')
                    
                    
        #         # learning rate schduler ------------
        #         # adjust_learning_rate(optimizer, scheduler(epoch))
        #         # rate = get_learning_rate(optimizer)[0] #scheduler.get_last_lr()[0] #get_learning_rate(optimizer)
                
        #         # one iteration update  -------------
        #         batch_size = len(batch['index'])
        #         batch['image'] = batch['image'].half().cuda()
        #         batch['mask' ] = batch['mask' ].half().cuda()
        #         batch['organ'] = batch['organ'].cuda()
                
                
        #         net.train()
        #         net.output_type = ['loss']
        #         #with torch.autograd.set_detect_anomaly(True):
        #         if 1:
        #             with amp.autocast(enabled = is_amp):
        #                 output = data_parallel(net,batch)
        #                 loss0  = output['bce_loss'].mean()
        #                 loss1  = output['aux2_loss'].mean()
        #             #loss1  = output['lovasz_loss'].mean()
                
        #             optimizer.zero_grad()
        #             scaler.scale(loss0+0.2*loss1).backward()
                    
        #             scaler.unscale_(optimizer)
        #             #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
        #             scaler.step(optimizer)
        #             scaler.update()
                
                
        #         # print statistics  --------
        #         batch_loss[:2] = [loss0.item(),loss1.item()]
        #         sum_train_loss += batch_loss
        #         sum_train += 1
        #         if t % 100 == 0:
        #             train_loss = sum_train_loss / (sum_train + 1e-12)
        #             sum_train_loss[...] = 0
        #             sum_train = 0
                
        #         print('\r', end='', flush=True)
        #         print(message(mode='print'), end='', flush=True)
        #         epoch += 1 / len(train_loader)
        #         iteration += 1