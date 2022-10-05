import numpy as np
import cv2 as cv

# img1 = cv.imread("D:/hacking_the_human_body/hubmap-organ-segmentation/image_masks/mask1168.png")
# img2 = cv.imread("D:/hacking_the_human_body/hubmap-organ-segmentation/image_masks/mask1168.png")
# img1 = np.expand_dims(img1, 0)
# img2 = np.expand_dims(img2, 0)
# img1 = img1/255
# img2 = img2/255

def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

def dice_coef2(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union

def mean_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
    batch_size = y_true.shape[0]
    channel_num = y_true.shape[-1]
    mean_dice_channel = 0.
    for i in range(batch_size):
        for j in range(channel_num):
            channel_dice = single_dice_coef(y_true[i, :, :, j], y_pred_bin[i, :, :, j])
            mean_dice_channel += channel_dice/(channel_num*batch_size)
    return mean_dice_channel

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = (y_true).flatten()
    y_pred_f = (y_pred).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef2beembom(target, prediction, smooth=1):
    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef

def DICE_COElala(mask1, mask2):
    # mask1 = mask1/255
    # mask2 = mask2/255
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading
    return dice 

# # dice = dice_coef2(img1, img2)
# dice = mean_dice_coef(img1, img2)
# # dice = np.sum(img1[img2==255])*2.0 / (np.sum(img1) + np.sum(img2))
# # dice = DICE_COElala(img1, img2)
# print(dice)