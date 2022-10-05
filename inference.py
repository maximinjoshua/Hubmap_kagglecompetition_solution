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

class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, encs=None):
        self.data = rasterio.open(os.path.join(DATA,str(idx)+'.tiff'),num_threads='all_cpus')
        # some images have issues with their format 
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        # self.rescaled_data = cv.resize(self.data, trained_pixel_size, interpolation = cv.INTER_AREA)
        self.shape = self.data.shape
        # self.shape = self.rescaled_data.shape
        # self.reduce = reduce
        self.sz = sz
        self.input_sz = config['input_resolution']
        self.pad0 = (self.sz - self.shape[0]%self.sz)%self.sz
        self.pad1 = (self.sz - self.shape[1]%self.sz)%self.sz
        self.n0max = (self.shape[0] + self.pad0)//self.sz
        self.n1max = (self.shape[1] + self.pad1)//self.sz
        self.mask = enc2mask(encs,(self.shape[1],self.shape[0])) if encs is not None else None

    def original_img_shape(self):
        return self.shape
        
    def __len__(self):
        return self.n0max*self.n1max
    
    def __getitem__(self, idx):
        n0,n1 = idx//self.n1max, idx%self.n1max
#         print('n0', n0, 'n1', n1)
#         print('pad0', self.pad0, 'pad1', self.pad1)#72 72
#         print(self.sz)
        x0,y0 = -self.pad0//2 + n0*self.sz, -self.pad1//2 + n1*self.sz
#         print('x0', x0, 'y0', y0)

        p00,p01 = max(0,x0), min(x0+self.sz,self.shape[0])
#         print('p00', p00, 'p01', p01)
        p10,p11 = max(0,y0), min(y0+self.sz,self.shape[1])
#         print('p10', p10,'p11', p11)
        img = np.zeros((self.sz,self.sz,3),np.uint8)
        mask = np.zeros((self.sz,self.sz),np.uint8)

        if self.data.count == 3:
            img[(p00-x0):(p01-x0),(p10-y0):(p11-y0)] = np.moveaxis(self.data.read([1,2,3], 
                # out_shape = (self.data.count, int(self.data.height*trained_pixel_size), int(self.data.width*trained_pixel_size)),
                window=Window.from_slices((p00,p01),(p10,p11))), 0, -1)
        else:
            for i,layer in enumerate(self.layers):
                img[(p00-x0):(p01-x0),(p10-y0):(p11-y0),i] =\
                  layer.read(1,window=Window.from_slices((p00,p01),(p10,p11)))
        if self.mask is not None: mask[(p00-x0):(p01-x0),(p10-y0):(p11-y0)] = self.mask[p00:p01,p10:p11]
        
        # if self.reduce != 1:
        img = cv2.resize(img,(self.input_sz, self.input_sz),
                            interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.input_sz, self.input_sz),
                            interpolation = cv2.INTER_NEAREST)

#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         h,s,v = cv2.split(hsv)
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img = img.unsqueeze(0)
            
        return {'img':img, 'mask':mask, 'idx':idx}


import math
def reverse_pad(images_array, original_img_shape):
    stitched_img = np.zeros([original_img_shape[0], original_img_shape[1], 1])
    img_shape = images_array.shape[2]
    rows, columns = images_array.shape[0], images_array.shape[1]
    big_image_shape = img_shape*rows
    total_padding = big_image_shape - original_img_shape[0]
#     print('total+padding',total_padding)
    for n0 in range(rows):
        for n1 in range(columns):
            x0, y0 = -total_padding//2 + n0*img_shape, -total_padding//2 + n1*img_shape
#             print('fssfx0', x0, 'fsdfy0', y0)
            p00,p01 = max(0,x0), min(x0+img_shape, original_img_shape[0])
            p10,p11 = max(0,y0), min(y0+img_shape, original_img_shape[1])
            stitched_img[(p00):(p01),(p10):(p11)] = images_array[n0][n1][(p00-x0):(p01-x0),(p10-y0):(p11-y0)]
#     print('stitched_img', stitched_img)
#     print('unique', np.unique(stitched_img))
    return stitched_img

import cv2 as cv
import pandas as pd
import torch.nn.functional as F
def infer(model, device):
    model = model
    model.eval()
    # checkpoint_path = "../input/checkpoint14hubmap/somethin_for_the_day_epoch_14.pth"
    checkpoint_path = "C:/Users/Maxi/Downloads/fold0_epoch17_bestloss.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    # test_csv = pd.read_csv('../input/hubmap-organ-segmentation/test.csv')
    test_csv = pd.read_csv("D:/new_hubmap_with_toms_solution/inputs_folder/test.csv")
#     test_images_list = os.listdir(DATA)
#     test_images_list_indices = list(map(lambda x: x.split('.')[0], test_images_list))
#     print(test_images_list_indices)
    test_rows = []
    ola = 0
    for index, details in test_csv.iterrows():
        # print('index', index)
        dl = HuBMAPDataset(details['id'])
        # print(len(dl))
        original_img_shape = dl.original_img_shape()
        number_of_small_imgs = len(dl)
        rows, columns = int(math.sqrt(number_of_small_imgs)), int(math.sqrt(number_of_small_imgs))
        img_array_for_stitching = np.zeros([rows, columns, sz, sz, 1])
        for i in range(len(dl)):
            image = dl[i]
            image['img'] = image['img'].to(device)
            with torch.no_grad():
                pred = model(image)
                # pred = F.sigmoid(pred)
            pred = pred['probability'].squeeze(0).permute(1,2,0)
            # print(pred.shape)
            pred = pred.cpu().numpy()
            pred = cv.resize(pred, (1024,1024), interpolation=cv.INTER_AREA)
            # cv.imwrite('D:/hacking_the_human_body/hubmap-organ-segmentation/finally_a_segmented_image(been_thru_a_lot_fa_dis_mahn)/segmentedhenckchips'+ str(ola)+'.png', pred)
            pred = np.expand_dims(pred, 2)
            pred = np.where(pred<0.5, 0, 1)
            # pred = pred.astype(np.uint8)*255
            # cv.imwrite('D:/hacking_the_human_body/hubmap-organ-segmentation/finally_a_segmented_image(been_thru_a_lot_fa_dis_mahn)/segmentedhenckchips'+ str(ola)+'.png', pred)
#             print(np.unique(pred))
            # pred = pred*255
            # pred = pred.astype(np.uint8)
            img_array_for_stitching[i//rows][i%rows] = pred
        reversed_image = reverse_pad(img_array_for_stitching, original_img_shape)
        encoding = mask2enc(reversed_image)
        # print(type(encoding))
        test_rows.append({
        'id': details['id'],
        'rle': encoding[0] })
        # print('encoding',encoding)
        # reversed_image = reversed_image*255
        # cv.imwrite('D:/hacking_the_human_body/hubmap-organ-segmentation/finally_a_segmented_image(been_thru_a_lot_fa_dis_mahn)/segmentedhencknewinferencev1'+ str(ola)+'.png', reversed_image)
        # ola += 1
        # break
    return test_rows

model = Net().to(device)
model.output_type = ['inference']
infer_results = infer(model, device)

submission_df = pd.DataFrame(infer_results)
# submission_df.to_csv('submission.csv', index=False)
