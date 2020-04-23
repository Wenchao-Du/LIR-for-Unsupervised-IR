import math
import numbers
import torch
from torch import nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt 

class GaussionSmoothLayer(nn.Module):
    def __init__(self, channel, kernel_size, sigma, dim = 2):
        super(GaussionSmoothLayer, self).__init__()
        kernel_x = cv2.getGaussianKernel(kernel_size, sigma)
        kernel_y = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel_x * kernel_y.T
        self.groups = channel
        if dim == 1:
            self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                    groups= channel, bias= False)
        elif dim == 2: 
            self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                    groups= channel, bias= False)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                    groups= channel, bias= False)
        else:
            raise RuntimeError(
                'input dim is not supported !, please check it !'
            )
        self.conv.weight.requires_grad = False
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel))
        self.pad = int((kernel_size - 1) / 2)
    def forward(self, input):
        intdata = input
        intdata = F.pad(intdata, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        output = self.conv(intdata)
        return output

class LapLasGradient(nn.Module):
    def __init__(self, indim, outdim):
        super(LapLasGradient, self).__init__()
        # @ define the sobel filter for x and y axis 
        kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]
            ]
        )
        kernel2 = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]
            ]
        )
        kernel3 = torch.tensor(
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]
            ]
        )
        kernel4 = torch.tensor(
            [[1, 1, 1],
             [1, -8, 1],
             [1, 1, 1]
            ]
        )
        self.conv = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)
        self.conv.weight.data.copy_(kernel4)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        grad = self.conv(x)
        return grad



class GradientLoss(nn.Module):
    def __init__(self, indim, outdim):
        super(GradientLoss, self).__init__()
        # @ define the sobel filter for x and y axis 
        x_kernel = torch.tensor(
            [ [1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]
            ]
        )
        y_kernel = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]
            ]
        )
        self.conv_x = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)
        self.conv_y = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)
        self.conv_x.weight.data.copy_(x_kernel)
        self.conv_y.weight.data.copy_(y_kernel)
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        gradient = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
        return gradient

def main(file1,file2):
    mat = cv2.imread(file1)
    nmat = cv2.imread(file2)
    tensor = torch.from_numpy(mat).float()
    tensor1 = torch.from_numpy(nmat).float()

    # 11, 17, 25, 50 
    blurkernel = GaussionSmoothLayer(3, 11, 50)
    gradloss = GradientLoss(3, 3)
    

    tensor = tensor.permute(2, 0, 1)
    tensor = torch.unsqueeze(tensor, dim = 0)
    
    tensor1 = tensor1.permute(2, 0, 1)
    tensor1 = torch.unsqueeze(tensor1, dim = 0)
    
    out = blurkernel(tensor)
    out1 = blurkernel(tensor1)

    loss = gradloss(out)
    loss1 = gradloss(out1)

    out = out.permute(0, 2, 3, 1).int()
    out = out.numpy().squeeze().astype(np.uint8)

    out1 = out1.permute(0, 2, 3, 1).int()
    out1 = out1.numpy().squeeze().astype(np.uint8)

    cv2.imshow("1", out)
    cv2.imshow("2", out1)
    cv2.waitKey(0)

#   \
#                                          
def testPIL(file1, file2):
    transform = transforms.Compose([
                                    transforms.ToTensor()
                                    ])
    image11 = transform(Image.open(file1).convert('RGB')).unsqueeze(0)
    image22 = transform(Image.open(file2).convert('RGB')).unsqueeze(0)
    # blurkernel = GaussionSmoothLayer(3, 11, 15)
    # gradloss = LapLasGradient(3, 3)
    gradloss2 = GradientLoss(3,3)
    # image1 = blurkernel(image11)
    image1 = gradloss2(image11)
    image1 = image1.numpy().squeeze()
    image1 = np.transpose(image1, (1,2,0))
    # image2 = blurkernel(image22)
    image2 = gradloss2(image22)
    image2 = image2.numpy().squeeze()
    image2 = np.transpose(image2, (1,2,0))
    plt.figure('1')
    plt.imshow(image1, interpolation='nearest')
    plt.figure('2')
    plt.imshow(image2, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    file1 = 'K:\\EDNGAN\\results\\expimages\\504\\hdtestRed_ssim_.png'
    file2 = 'K:\\EDNGAN\\results\\expimages\\504\\ldtestRed_ssim_.png'
    # main(file1, file2)
    testPIL(file1, file2)
    
    
