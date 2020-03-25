"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, data_prefetcher
import argparse
from torch.autograd import Variable
from trainer import UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/unit_noise2self.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

def main():
    cudnn.benchmark = True
    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path
    trainer = UNIT_Trainer(config)
    if torch.cuda.is_available():
        trainer.cuda(config['gpuID'])
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    
    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    writer = SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
    
    print('start training !!')
    # Start training
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    
    TraindataA = data_prefetcher(train_loader_a) 
    TraindataB = data_prefetcher(train_loader_b)
    testdataA = data_prefetcher(test_loader_a)
    testdataB = data_prefetcher(test_loader_b)

    while True:
        dataA = TraindataA.next()
        dataB = TraindataB.next()
        if dataA is None or dataB is None:
            TraindataA = data_prefetcher(train_loader_a) 
            TraindataB = data_prefetcher(train_loader_b)
            dataA = TraindataA.next()
            dataB = TraindataB.next()
        with Timer("Elapsed time in update: %f"):
            trainer.dis_update(dataA, dataB, config)
            trainer.gen_update(dataA, config)
            # torch.cuda.synchronize()

            # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, writer)
            # writer.add_graph(trainer, (dataA, dataB))
            # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            testa = testdataA.next()
            testb = testdataB.next()
            if testa is None or testb is None:
                testdataA = data_prefetcher(test_loader_a) 
                testdataB = data_prefetcher(test_loader_b)
                testa = testdataA.next()
                testb = testdataB.next()
            with torch.no_grad():
                test_image_outputs = trainer.sample(testa, testb)
                train_image_outputs = trainer.sample(dataA, dataB)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                # HTML
                # write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(dataA, dataB)
            if image_outputs is not None:
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)
        trainer.update_learning_rate()
        iterations += 1
        if iterations >= max_iter:
            writer.close()
            sys.exit('Finish training')
        

if __name__ == "__main__":
    main()    