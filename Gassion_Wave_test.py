import pywt
import numpy as np
import cv2
from scipy import ndimage


def wave_process(input):
    if input.shape[2] == 3:
        ciffes = pywt.dwtn(input, 'haar')
    else:
        ciffes = pywt.dwt2(input, 'haar')
    for i in range(len(ciffes)):
        cv2.imshow(ciffes[i][:,:,0])
        cv2.waitkey(0)
    return len(ciffes)

def main(file1, file2):
    mat = cv2.imread(file1)
    nmat = cv2.imread(file2)
    blurmat = cv2.GaussianBlur(mat, (5, 5), 3)
    nblurmat = cv2.GaussianBlur(nmat, (25, 25), 50)
    sx = ndimage.sobel(mat, axis = 0, mode = 'constant')
    sy = ndimage.sobel(mat, axis = 1, mode = 'constant')
    sobel = np.hypot(sx, sy).mean()
    print(sobel)
    sx = ndimage.sobel(nmat, axis = 0, mode = 'constant')
    sy = ndimage.sobel(nmat, axis = 1, mode = 'constant')
    sobel = np.hypot(sx, sy).mean()
    print(sobel)
    cv2.imshow("", blurmat)
    cv2.imshow("n", nblurmat)
    cv2.waitKey(0)


if __name__ == "__main__":
    file1 = 'datasets\\noi2clr\\testA\\0060.png'
    file2 = 'datasets\\noi2clr\\testB\\0060.png'
    main(file1, file2)
