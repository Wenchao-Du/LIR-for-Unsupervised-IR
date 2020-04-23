################################################################
# @this file is mainly used to process Pascal Voc data
# @ useness: split src dataset into trainset, testset  
# @ author: Duwenchao
################################################################

from __future__ import print_function
import numpy as np
import os
import random
import shutil
import skimage.io as IO
import skimage.data as data
import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
from skimage import data


def add_noise(img, mode, var):
	sigma = var
	sigma_ = sigma / 255.
	sigma_ = sigma_ * sigma_
	if np.max(img) > 1:		img = img / 255.
	if mode is None:
		raise Exception('please add the noise type !!')
	if var is None:
		noisemat = random_noise(img, mode = mode)
	elif mode == 'poisson':
		noisemat = random_noise(img, mode = mode)
	else:
		noisemat = random_noise(img, mode = mode, var=sigma_)
	return noisemat

def bernoulli_noise(size, p):
	mask = np.random.binomial(1, p, size)
	return mask

def add_bernoulli_noise(folder, distfolder):
	if not os.path.exists(folder):
		raise Exception('folder is not exist!')
	if not os.path.exists(distfolder):
		os.makedirs(distfolder)
	filelist = os.listdir(folder)
	# plist = np.random.uniform(0.35, 1, len(filelist))
	p=0.95
	for index, file in enumerate(filelist):
		filepath = folder + '\\' + file
		# img = IO.imread(filepath)
		img = data.load(filepath)
		if len(img.shape) < 3:
			os.remove(filepath)
			continue
		width, height = img.shape[0], img.shape[1]
		if width < 64 or height < 64:
			os.remove(filepath)
			continue
		mask = bernoulli_noise((width, height, 1), p)
		mask = np.concatenate((mask, mask, mask), 2)
		noiseimg = mask * img
		# IO.imshow(noiseimg)
		# IO.show()
		# return noiseimg
		outpath = distfolder + '\\' +  file
		IO.imsave(outpath, noiseimg)
		print(index)

def splitlist(folder, trainfolder1, trainfolder2, testfolder):
	if not os.path.exists(folder):
		raise Exception('input folder dose not exist ! please check it !!')
	if not os.path.exists(trainfolder1):
		os.makedirs(trainfolder1)
	if not os.path.exists(trainfolder2):
		os.makedirs(trainfolder2)
	if not os.path.exists(testfolder):
		os.makedirs(testfolder)

	filelist = os.listdir(folder)
	sublist = random.sample(filelist, 110000)
	sublist_a = random.sample(sublist, 55000)
	for i, sfile in enumerate(filelist):
		fpath = folder + '\\' + sfile
		if sfile in sublist_a:
			shutil.copy2(fpath, trainfolder1)
		elif sfile in sublist:
			shutil.copy2(fpath, trainfolder2)
		else:
			shutil.copy2(fpath, testfolder)
		print(i)

def Imagetransform(srcfolder, destfolder, mode):
	if not os.path.exists(srcfolder):
		raise Exception('input srcfolder does not exists! ')
	if not os.path.exists(destfolder):
		os.makedirs(destfolder)
	filelist = os.listdir(srcfolder)
	count = len(filelist)
	varlist = np.random.randint(5, 51, count)
	IO.use_plugin('pil') # SET the specific plugin, default: imgio 
	for i, sfile in enumerate(filelist):
		print("{}, {}".format(i, varlist[i]))
		filename = srcfolder + '\\' + sfile
		mat = IO.imread(filename)
		# mat = data.load(filename)
		if len(mat.shape) < 3:
			continue
		w, h, c = mat.shape
		if h < 64 or w < 64:
			continue
		noimat = add_noise(mat, mode, 50) # varlist[i]
		# plt.figure('noi')
		# plt.imshow(noimat, interpolation='nearest')
		# plt.show()
		outfile = destfolder + '\\' + sfile
		IO.imsave(outfile, noimat)

def checksize(folder, size):
	if not os.path.exists(folder):
		raise Exception('folder is not exist!')
	filelist = os.listdir(folder)
	for index, file in enumerate(filelist):
		filepath = folder + '\\' + file
		mat = IO.imread(filepath)
		if len(mat.shape) < 3:
			os.remove(filepath)
			print("{}: is deleted".format(file))
			continue
		if mat.shape[0] < size or mat.shape[1] < size:
			os.remove(filepath)
			print("{}: is deleted".format(file))
			continue
		else:
			print(index)
if __name__ == '__main__':
	folder = 'J:\\Public_DataSet\\Kodak\\original'
	folder1 = 'J:\\Public_DataSet\\train2017\\voctest'
	folder2 = 'J:\\Public_DataSet\\train2017\\trainA_JPG'
	folder3 = 'J:\\Public_DataSet\\SR_testing_datasets\\Kodak\\original'
	destfolder = 'J:\\Public_DataSet\\SR_testing_datasets\\Kodak\\boulli_p8'
	# splitlist(folder, folder1, folder2, folder3)
	mode = 'gaussian'
	# mode2 = 'poisson'
	# Imagetransform(folder, destfolder, mode)
	# file = 'J:\\Public_DataSet\\SR_testing_datasets\\Set5\\baby.png'
	add_bernoulli_noise(folder3, destfolder)
	# checksize(folder, 64)
	