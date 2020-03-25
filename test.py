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
parser.add_argument('--config', type=str, default='configs/unit_noise2clear-base.yaml', help="net configuration")
parser.add_argument('--input', type=str, default = 'F:\\CBSD68-dataset\\CBSD68\\CropBSD68\\noisy25', help="input image path")

parser.add_argument('--output_folder', type=str, default='./outputs/BSDN25-base', help="output image path")
parser.add_argument('--checkpoint', type=str, default='outputs\\unit_noise2clear-base\\checkpoints\\gen_00300000.pt',
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
    filepath = opts.input + '\\' + file
    image = Variable(transform(Image.open(
        filepath).convert('RGB')).unsqueeze(0).cuda())
    # Start testing
    content = encode(image)
    outputs = decode(content)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)
    path = os.path.join(opts.output_folder, file)
    outputs_back = outputs.clone()
    vutils.save_image(outputs.data, path, padding=0, normalize=False)

    
    

