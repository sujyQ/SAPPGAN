import os
import random
import numpy as np
import shutil
from shutil import copyfile
import math
import numbers
from PIL import Image
from scipy import signal
import scipy.io
from collections import OrderedDict


BLUR_RANGE = ['k13', 'k15', 'k17', 'k19', 'k21', 'k23', 'k25', 'k27']
PADDING = dict(k13=6, k15=7, k17=8, k19=9, k21=10, k23=11, k25=12, k27=13)
DATASET = ['CelebA', 'CelebA-HQ', 'FFHQ']

ROOT_DIR = '/home/hsj/drive/hsj/dataset/testsets/MSPL'

KERNEL_PATH = os.path.join(ROOT_DIR, 'test250kernels')

CelebA = dict(
    gt_dir = os.path.join(ROOT_DIR, 'Test_data_ver01_CentorCrop','Test_data_{}'.format(DATASET[0]), '{}_gt'.format(DATASET[0])),
    blur_dir = os.path.join(ROOT_DIR, 'Test_data_ver01_CentorCrop','Test_data_{}'.format(DATASET[0]), '{}_blur'.format(DATASET[0])),
)

CelebA_HQ = dict(
    gt_dir = os.path.join(ROOT_DIR, 'Test_data_ver01_CentorCrop', 'Test_data_{}'.format(DATASET[1]), '{}_gt'.format(DATASET[1])),
    blur_dir = os.path.join(ROOT_DIR, 'Test_data_ver01_CentorCrop','Test_data_{}'.format(DATASET[1]), '{}_blur'.format(DATASET[1])),
)

FFHQ = dict(
    gt_dir = os.path.join(ROOT_DIR,'Test_data_ver01_CentorCrop', 'Test_data_{}'.format(DATASET[2]), '{}_gt'.format(DATASET[2])),
    blur_dir = os.path.join(ROOT_DIR, 'Test_data_ver01_CentorCrop','Test_data_{}'.format(DATASET[2]), '{}_blur'.format(DATASET[2])),
)


# Read Blur kernels
KERNEL_DICT = OrderedDict()
for blur_size in BLUR_RANGE:                
    k_filename_kernel =os.path.join(KERNEL_PATH, 'blur_{}.mat'.format(blur_size))
    kernel_file = scipy.io.loadmat(k_filename_kernel)
    kernels = np.array(kernel_file['blurs_{}'.format(blur_size)])
    KERNEL_DICT['{}'.format(blur_size)] = kernels


gtlist0 = os.listdir(CelebA['gt_dir'])
gtlist1 = os.listdir(CelebA_HQ['gt_dir'])
gtlist2 = os.listdir(FFHQ['gt_dir'])

assert len(gtlist0) == len(gtlist1) == len(gtlist2) == 80


for i, gt in enumerate(gtlist0):
    impath = os.path.join(CelebA['gt_dir'], gt)
    imname = os.path.splitext(os.path.split(impath)[-1])[0]
    
    gt_img = Image.open(impath)
    gt_img_np = np.array(gt_img)
    gt_img_np = (gt_img_np/255.0).astype(np.float32)
        
    for b_range, kernels in list(KERNEL_DICT.items()):              
        for index in range(0, 10):
            blur_img_np = np.copy(gt_img_np)
            copy_img_np = np.copy(gt_img_np)

            pad_size = PADDING['{}'.format(b_range)]
            n_pad = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
            copy_img_np = np.pad(copy_img_np, n_pad, 'reflect')

            # convolve blur kernels
            blur_img_np[:,:,0] = signal.convolve(copy_img_np[:,:,0], kernels[:,:,index], mode='valid')
            blur_img_np[:,:,1] = signal.convolve(copy_img_np[:,:,1], kernels[:,:,index], mode='valid')
            blur_img_np[:,:,2] = signal.convolve(copy_img_np[:,:,2], kernels[:,:,index], mode='valid')
            
            blur_img_np = (blur_img_np * 255.0).round().clip(0, 255).astype(np.uint8)
            blur_img = Image.fromarray(blur_img_np)
            
            w, h = blur_img.size
            assert w == 128 and h == 128


            savename = '{}_ker{:0>2}_blur_{}.png'.format(imname, index+1, b_range)
            savefile = os.path.join(CelebA['blur_dir'], savename)
            blur_img.save(savefile)
            print(i+1, savename)

        
for i, gt in enumerate(gtlist1):
    impath = os.path.join(CelebA_HQ['gt_dir'], gt)
    imname = os.path.splitext(os.path.split(impath)[-1])[0]
    
    gt_img = Image.open(impath)
    gt_img_np = np.array(gt_img)
    gt_img_np = (gt_img_np/255.0).astype(np.float32)
        
    for b_range, kernels in list(KERNEL_DICT.items()):
        for index in range(10, 20):
            blur_img_np = np.copy(gt_img_np)
            copy_img_np = np.copy(gt_img_np)

            pad_size = PADDING['{}'.format(b_range)]
            n_pad = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
            copy_img_np = np.pad(copy_img_np, n_pad, 'reflect')
            
            # convolve blur kernels
            blur_img_np[:,:,0] = signal.convolve(copy_img_np[:,:,0], kernels[:,:,index], mode='valid')
            blur_img_np[:,:,1] = signal.convolve(copy_img_np[:,:,1], kernels[:,:,index], mode='valid')
            blur_img_np[:,:,2] = signal.convolve(copy_img_np[:,:,2], kernels[:,:,index], mode='valid')
            
            blur_img_np = (blur_img_np * 255.0).round().clip(0, 255).astype(np.uint8)
            blur_img = Image.fromarray(blur_img_np)
            
            w, h = blur_img.size
            assert w == 128 and h == 128

            savename = '{}_ker{:0>2}_blur_{}.png'.format(imname, index+1, b_range)
            savefile = os.path.join(CelebA_HQ['blur_dir'], savename)
            blur_img.save(savefile)
            print(i+1, savename)


for i, gt in enumerate(gtlist2):
    impath = os.path.join(FFHQ['gt_dir'], gt)
    imname = os.path.splitext(os.path.split(impath)[-1])[0]
    
    gt_img = Image.open(impath)
    gt_img_np = np.array(gt_img)
    gt_img_np = (gt_img_np/255.0).astype(np.float32)
        
    for b_range, kernels in list(KERNEL_DICT.items()):
        for index in range(20, 30):
            blur_img_np = np.copy(gt_img_np)
            copy_img_np = np.copy(gt_img_np)

            pad_size = PADDING['{}'.format(b_range)]
            n_pad = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
            copy_img_np = np.pad(copy_img_np, n_pad, 'reflect')
            
            # convolve blur kernels
            blur_img_np[:,:,0] = signal.convolve(copy_img_np[:,:,0], kernels[:,:,index], mode='valid')
            blur_img_np[:,:,1] = signal.convolve(copy_img_np[:,:,1], kernels[:,:,index], mode='valid')
            blur_img_np[:,:,2] = signal.convolve(copy_img_np[:,:,2], kernels[:,:,index], mode='valid')
            
            blur_img_np = (blur_img_np * 255.0).round().clip(0, 255).astype(np.uint8)
            blur_img = Image.fromarray(blur_img_np)
            
            w, h = blur_img.size
            assert w == 128 and h == 128

            savename = '{}_ker{:0>2}_blur_{}.png'.format(imname, index+1, b_range)
            savefile = os.path.join(FFHQ['blur_dir'], savename)
            blur_img.save(savefile)
            print(i+1, savename)


