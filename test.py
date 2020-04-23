"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config
from trainer import UNIT_Trainer
import matplotlib.pyplot as plt
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/unit_noise2clear-bn-Deblur.yaml', help="net configuration")
parser.add_argument('--input', type=str, default = '/mnt/725AAA345AA9F54F/Public_DataSet/cvpr16_deblur_study_real_dataset/real_dataset', help="input image path")

parser.add_argument('--output_folder', type=str, default='./Deblur_Real', help="output image path")
parser.add_argument('--checkpoint', type=str, default='/mnt/B290B95290B91E33/Dual_UNIT/outputs/unit_noise2clear-bn-Deblur/checkpoints/gen_00300000.pt',
                    help="checkpoint of autoencoders") 
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
parser.add_argument('--psnr', action="store_false", help='is used to compare psnr')
parser.add_argument('--ref', type=str, default='J:\\Public_DataSet\\Kodak\\original\\kodim04.png', help='cmpared refferd image')
opts = parser.parse_args()


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loaderopts.trainer == 'UNIT':
trainer = UNIT_Trainer(config)
state_dict = torch.load(opts.checkpoint, map_location='cpu')
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode_cont  # encode function
decode = trainer.gen_b.dec_cont  # decode function

if not os.path.exists(opts.input):
    raise Exception('input path is not exists!')
imglist = os.listdir(opts.input)
transform = transforms.Compose([transforms.ToTensor()])
for i, file in enumerate(imglist):
    if not file.endswith('.jpg'):
        continue
    print(file)
    filepath = opts.input + '/' + file
    image = transform(Image.open(
        filepath).convert('RGB')).unsqueeze(0).cuda()
        # Start testing
    h,w = image.size(2),image.size(3)
    if h > 800 or w > 800:
        continue
    pad_h = h % 4
    pad_w = w % 4
    image = image[:,:,0:h-pad_h, 0:w - pad_w]
    content = encode(image)
    outputs = decode(content)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)
    path = os.path.join(opts.output_folder, file)
    outputs_back = outputs.clone()
    vutils.save_image(outputs.data, path, padding=0, normalize=True)
    # if opts.psnr:
    #     outputs = torch.squeeze(outputs_back)
    #     outputs = outputs.permute(1, 2, 0).to('cpu', torch.float32).numpy()
    #     # outputs = outputs.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #     ref = Image.open(opts.ref).convert('RGB')
    #     ref = np.array(ref) / 255.
    #     noi = Image.open(opts.input).convert('RGB')
    #     noi = np.array(noi) / 255.
    #     rmpad_h = noi.shape[0] % 4
    #     rmpad_w = noi.shape[1] % 4

    #     pad_h = ref.shape[0] % 4
    #     pad_w = ref.shape[1] % 4

        # if rmpad_h != 0 or pad_h != 0:
        #     noi = noi[0:noi.shape[0]-rmpad_h,:,:]
        #     ref = ref[0:ref.shape[0]-pad_h,:,:]
        # if rmpad_w != 0 or pad_w != 0:
        #     noi = noi[:, 0:noi.shape[1]-rmpad_w,:]
        #     ref = ref[:, 0:ref.shape[1]-pad_w,:]
            
        # psnr = compare_psnr(ref, outputs)
        # ssim = compare_ssim(ref, outputs, multichannel=True)
        # print('psnr:{}, ssim:{}'.format(psnr, ssim))
        # plt.figure('ref')
        # plt.imshow(ref, interpolation='nearest')
        # plt.figure('out')
        # plt.imshow(outputs, interpolation='nearest')
        # plt.figure('in')
        # plt.imshow(noi, interpolation='nearest')
        # plt.show()

    
    

