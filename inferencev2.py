import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
from tqdm.notebook import tqdm
import zipfile
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import gc
import glob
from torchvision import transforms
import torch
from model import Net
from get_config import get_config

DATA = 'D:/hacking_the_human_body/hubmap-organ-segmentation/test_images/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sz = 1024
config = get_config()
# reduce = 2
trained_pixel_size = 0.4945/0.4

def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

from torch.utils.data import Dataset

class HuBMAPDataset(Dataset):
    def __init__(self, idx):
        super().__init__()
        self.data = rasterio.open(os.path.join(DATA,str(idx)+'.tiff'),num_threads='all_cpus')
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i,subdataset in enumerate(subdatasets,0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width
        self.input_sz = config['input_resolution']
        self.sz = config['resolution']
        self.pad_sz = config['pad_size'] # add to each input tile
        self.pred_sz = self.sz - 2*self.pad_sz
        self.pad_h = self.pred_sz - self.h % self.pred_sz # add to whole slide
        self.pad_w = self.pred_sz - self.w % self.pred_sz # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.pred_sz
        self.num_w = (self.w + self.pad_w) // self.pred_sz
        
    def __len__(self):
        return self.num_h * self.num_w
    
    def __getitem__(self, idx): # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h*self.pred_sz 
        x = i_w*self.pred_sz
        py0,py1 = max(0,y), min(y+self.pred_sz, self.h)
        px0,px1 = max(0,x), min(x+self.pred_sz, self.w)
        
        # padding coordinate for rasterio
        qy0,qy1 = max(0,y-self.pad_sz), min(y+self.pred_sz+self.pad_sz, self.h)
        qx0,qx1 = max(0,x-self.pad_sz), min(x+self.pred_sz+self.pad_sz, self.w)
        
        # placeholder for input tile (before resize)
        img = np.zeros((self.sz,self.sz,3), np.uint8)
        
        # replace the value
        if self.data.count == 3:
            img[0:qy1-qy0, 0:qx1-qx0] =\
                np.moveaxis(self.data.read([1,2,3], window=Window.from_slices((qy0,qy1),(qx0,qx1))), 0,-1)
        else:
            for i,layer in enumerate(self.layers):
                img[0:qy1-qy0, 0:qx1-qx0, i] =\
                    layer.read(1,window=Window.from_slices((qy0,qy1),(qx0,qx1)))
        if self.sz != self.input_sz:
            img = cv2.resize(img, (self.input_sz, self.input_sz), interpolation=cv2.INTER_AREA)
        # img = self.transforms(image=img)['image'] # to normalized tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img = img.unsqueeze(0)
        return {'img':img, 'p':[py0,py1,px0,px1], 'q':[qy0,qy1,qx0,qx1]}


import cv2 as cv
import pandas as pd
import torch.nn.functional as F
def infer(model, device):
    model = model
    model.eval()
    # checkpoint_path = "../input/checkpoint14hubmap/somethin_for_the_day_epoch_14.pth"
    checkpoint_path = "C:/Users/Maxi/Downloads/fold0_epoch26_bestloss.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    # test_csv = pd.read_csv('../input/hubmap-organ-segmentation/test.csv')
    test_csv = pd.read_csv("D:/new_hubmap_with_toms_solution/inputs_folder/test.csv")
    test_rows = []
    ola = 0
    for index, details in test_csv.iterrows():
        print('index', index)
        dl = HuBMAPDataset(details['id'])
        pred_mask = np.zeros((len(dl),dl.pred_sz,dl.pred_sz), dtype=np.uint8)
        i_data = 0
        number_of_small_imgs = len(dl)

        for i in range(len(dl)):
            pred_list = []
            vflipped_img_dict = {}
            hflipped_img_dict = {}
            image = dl[i]
            image['img'] = image['img'].to(device)
            vflipped_img_dict['img'] = torch.flip(image['img'], dims = [3]).to(device)
            hflipped_img_dict['img'] = torch.flip(image['img'], dims = [1,2]).to(device)
            with torch.no_grad():
                pred = model(image)
                vpred = model(vflipped_img_dict)
                hpred = model(hflipped_img_dict)
            pred_list.extend([pred['probability'], torch.flip(vpred['probability'], dims = [3]), torch.flip(hpred\
                                                ['probability'], dims = [1,2])])
            preds = torch.mean(torch.cat(pred_list, dim = 0), dim = 0)
            pred = preds.permute(1,2,0)
            pred = pred.cpu().numpy()
            pred = cv.resize(pred, (1024,1024), interpolation=cv.INTER_AREA)
            pred = np.where(pred<0.5, 0, 1)
            pred = pred.astype(np.uint8)*255
            py0,py1,px0,px1 = image['p']
            qy0,qy1,qx0,qx1 = image['q']
            pred_mask[i_data,0:py1-py0, 0:px1-px0] = pred[py0-qy0:py1-qy0, px0-qx0:px1-qx0] # (pred_sz,pred_sz)
            i_data += 1
        pred_mask = pred_mask.reshape(dl.num_h*dl.num_w, dl.pred_sz, dl.pred_sz).reshape(dl.num_h, dl.num_w, dl.pred_sz, dl.pred_sz)
        pred_mask = pred_mask.transpose(0,2,1,3).reshape(dl.num_h*dl.pred_sz, dl.num_w*dl.pred_sz)
        pred_mask = pred_mask[:dl.h,:dl.w]
        cv.imwrite('D:/hacking_the_human_body/hubmap-organ-segmentation/finally_a_segmented_image(been_thru_a_lot_fa_dis_mahn)/testtimeaugmentation'+ str(details['id']) + str(ola)+'.png', pred_mask)
        ola+=1
        encoding = mask2enc(pred_mask)
        test_rows.append({
        'id': details['id'],
        'rle': encoding[0] })
        print('encoding',encoding)
    return test_rows

model = Net().to(device)
model.output_type = ['inference']
infer_results = infer(model, device)

submission_df = pd.DataFrame(infer_results)
submission_df.to_csv('submission.csv', index=False)

