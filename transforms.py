import numpy as np
from albumentations import (Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,
                            GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            GaussianBlur,CLAHE,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp)
from albumentations.pytorch import ToTensorV2
import albumentations as A
from get_config import get_config
config = get_config()

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def get_transforms_train():
    transform_train = A.Compose([
                A.augmentations.crops.RandomResizedCrop(height=config['input_resolution'], width=config['input_resolution']),
                A.augmentations.Rotate(limit=90, p=0.5),
                A.augmentations.HorizontalFlip(p=0.5),
                A.augmentations.VerticalFlip(p=0.5),
                A.augmentations.transforms.ColorJitter(p=0.5),
                A.OneOf([
                    A.OpticalDistortion(p=0.5),
                    A.GridDistortion(p=.5),
                    A.PiecewiseAffine(p=0.5),
                ], p=0.5),
                A.OneOf([
                    A.HueSaturationValue(10, 15, 10),
                    A.CLAHE(clip_limit=4),
                    A.RandomBrightnessContrast(),            
                ], p=0.5),                
                A.Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]), 
                  std=(STD[0], STD[1], STD[2])),
                ToTensorV2()
            ])
    return transform_train


def get_transforms_valid():
    transform_valid = Compose([
        Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]), 
                  std=(STD[0], STD[1], STD[2])),
        ToTensorV2(),
    ] )
    return transform_valid


def denormalize(z, mean=MEAN.reshape(-1,1,1), std=STD.reshape(-1,1,1)):
    return std*z + mean