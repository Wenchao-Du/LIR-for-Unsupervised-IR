from __future__ import print_function
import os
import numpy as np
import random
import shutil

def celeba(folder, subfolder1, subfolder2, subfolder3):
    if not os.path.exists(folder):
        raise Exception('folder does not exist')
    if not os.path.exists(subfolder1):
        os.makedirs(subfolder1)
    if not os.path.exists(subfolder2):
        os.makedirs(subfolder2)
    if not os.path.exists(subfolder3):
        os.makedirs(subfolder3)
    randlist = random.sample(range(1, 202600), 200000)
    sublist = random.sample(randlist, 100000)
    for sfile in range(202600):
        if sfile == 0:
            continue
        filename = str(sfile)
        predix = ''
        if len(filename) != 6:
            for _ in range(6-len(filename)):
                predix += '0'
        predix += filename
        file = folder + '\\' + predix + '.jpg'
        if sfile in sublist:
            shutil.copy2(file, subfolder1)
        elif sfile in randlist:
            shutil.copy2(file, subfolder2)
        else:
            shutil.copy2(file, subfolder3)
        print(sfile)

if __name__ == "__main__":
    folder = 'G:\\AnoData\\Celeba\\img_align_celeba'
    subfolder1 = 'J:\\Public_DataSet\\Celeba_A'
    subfolder2 = 'J:\\Public_DataSet\\Celeba_B'
    subfolder3 = 'J:\\Public_DataSet\\Celeba_Test'
    celeba(folder, subfolder1, subfolder2, subfolder3)