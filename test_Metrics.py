from skimage.measure import compare_psnr, compare_ssim
from PIL import Image as image
import numpy as np
import os

def test(testfolder, gtfolder):
    filelist = os.listdir(testfolder)
    psnr = 0
    ssim = 0
    count = 0
    for index, filepath in enumerate(filelist):
        if filepath == 'bridge.png':
            continue
        testmat = image.open(os.path.join(testfolder, filepath)).convert('RGB')
        testmat = np.array(testmat)
        gtmat = image.open(os.path.join(gtfolder, filepath)).convert('RGB')
        gtmat = np.array(gtmat)
        w,h,_ = testmat.shape
        gtmat = gtmat[0:w,0:h,:]
        psnr += compare_psnr(gtmat, testmat)
        ssim += compare_ssim(testmat, gtmat, multichannel=True)
        count += 1
    print('mean psnr: {}, ssim:{}'.format(psnr / count, ssim / count))

if __name__ == "__main__":
    # testfolder = '/mnt/B290B95290B91E33/Dual_UNIT/Set14_3SR'
    testfolder = '/mnt/B290B95290B91E33/Dual_UNIT/BSDS100_2SR'
    gtfolder = '/mnt/725AAA345AA9F54F/Public_DataSet/train2017/SR_Data/SR_testing_datasets/BSDS100'
    test(testfolder, gtfolder)