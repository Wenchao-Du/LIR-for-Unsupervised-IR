################################################################
# @this file is mainly used to process Pascal Voc data
# @ useness: image resize for super-resolution useness  
# @ author: Duwenchao
################################################################

from __future__ import print_function
import numpy as np
import os
import random
import shutil
import skimage.io as IO
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import cv2

def LR_dataPro(srcfolder, distfolder1, distfolder2):
    if not os.path.exists(srcfolder):
        raise Exception('src folder does not exist! please check it !')
    if not os.path.exists(distfolder1):
        os.makedirs(distfolder1)
    # if not os.path.exists(distfolder2):
    #     os.makedirs(distfolder2)
    filedir = os.listdir(srcfolder)
    count = len(filedir)
    scalelist = np.random.randint(2, 5, count)
    for index, file in enumerate(filedir):
        filepath = srcfolder + '/' + file
        img = IO.imread(filepath)
        size = img.shape
        if size[0] < 64 or size[1] < 64:
            continue
        pad_h = size[0] % 3
        pad_w = size[1] % 3
        img = img[0:(size[0] - pad_h), 0:(size[1] - pad_w),:]
        # outimg = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)), cv2.INTER_LINEAR)
        downimg = rescale(img, 1./3)
        upsample = rescale(downimg, 3)
        # outimg = cv2.resize(outimg, ( img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
        srcsavepath = distfolder1 + '/' + file
        # distsavepath = distfolder2 + '\\' + file.replace('jpg', 'png')
        # IO.imsave(srcsavepath, img)
        IO.imsave(srcsavepath, upsample)
        # cv2.imwrite(srcsavepath, outimg)
        print(index)
        

if __name__ == "__main__":
    folder = '/mnt/725AAA345AA9F54F/Public_DataSet/train2017/SR_Data/SR_testing_datasets/BSDS100'
    distfolder1 = '/mnt/725AAA345AA9F54F/Public_DataSet/train2017/SR_Data/SR_testing_datasets/BSDS100_3S'
    distfolder2 = 'J:\\Public_DataSet\\train2017\\voctestLR_ms_png'
    LR_dataPro(folder, distfolder1, distfolder2)