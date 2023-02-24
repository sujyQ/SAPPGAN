import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.models as py_models

import os
import numpy
import copy
from skimage.io import imread

import lpips

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# from perceptual import PerceptualLoss16

BLUR_RANGE = ['k13', 'k15', 'k17', 'k19', 'k21', 'k23', 'k25', 'k27']

def main(orig_dir, blur_dir, deblur_dir, save_dir, dataset_type, dataset_name, use_gpu=True):

    class PerceptualLoss16():
        def __init__(self, loss, gpu=0, p_layer=14):
            super(PerceptualLoss16, self).__init__()
            self.criterion = loss
    #         conv_3_3_layer = 14
            checkpoint = torch.load('VGGFace16.pth')
            vgg16 = py_models.vgg16(num_classes=2622)
            vgg16.load_state_dict(checkpoint['state_dict'])
            cnn = vgg16.features
            cnn = cnn.cuda()
    #         cnn = cnn.to(gpu)
            model = nn.Sequential()
            model = model.cuda()
            for i,layer in enumerate(list(cnn)):
    #             print(layer)
                model.add_module(str(i),layer)
                if i == p_layer:
                    break
            self.contentFunc = model   
            del vgg16, cnn, checkpoint

        def getloss(self, fakeIm, realIm):
            if isinstance(fakeIm, numpy.ndarray):
                fakeIm = torch.from_numpy(fakeIm).permute(2, 0, 1).unsqueeze(0).float().cuda()

            if isinstance(realIm, numpy.ndarray) :
                realIm = torch.from_numpy(realIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
            
            f_fake = self.contentFunc.forward(fakeIm)
            f_real = self.contentFunc.forward(realIm)
            f_real_no_grad = f_real.detach()
            loss = self.criterion(f_fake, f_real_no_grad)
            return loss

    perceptualLoss = PerceptualLoss16(nn.MSELoss().cuda(),p_layer=30)

    lpipsLoss = lpips.LPIPS(net='vgg', version='0.1').cuda()

    dVGG_blur = dict(all=[])
    dVGG_deblur = dict(all=[])
    lpips_list = dict(all=[])
    
    for b_range in BLUR_RANGE:
        dVGG_blur['{}'.format(b_range)] = []
        dVGG_deblur['{}'.format(b_range)] = []
        lpips_list['{}'.format(b_range)] = []

    f_test = open(save_dir, 'w')

    orig_list = sorted(os.listdir(orig_dir))
    deblur_list = sorted(os.listdir(deblur_dir))
    blur_list = sorted(os.listdir(blur_dir))

    n = 0
    for i, gt_path in enumerate(orig_list):
        gt_path = os.path.join(orig_dir, gt_path)
        gt_filename = os.path.splitext(os.path.split(gt_path)[-1])[0]
        print(gt_path)
        gt_img = imread(gt_path)
        for name in blur_list:
            blur_path = os.path.join(blur_dir, name)
            blur_filename = os.path.splitext(os.path.split(blur_path)[-1])[0]
            # # Nips
            # deblur_path = os.path.join(deblur_dir, deblur_list[n])
            # deblur_path = os.path.join(deblur_dir, blur_filename + '.png')
    #         # CVPR 2018
    #         deblur_path = os.path.join(deblur_dir, blur_filename + '_random.png')
    #         # ours
            # deblur_path = os.path.join(deblur_dir, blur_filename + '_out2.png')
            deblur_path = os.path.join(deblur_dir, blur_filename + '_out.png')
            # deblur_path = os.path.join(deblur_dir, blur_filename+'.png')
            deblur_filename = os.path.splitext(os.path.split(deblur_path)[-1])[0]
            if gt_filename in blur_filename:
                blur_img = imread(blur_path)
                deblur_img = imread(deblur_path)
                with torch.no_grad():
                    temp1 = perceptualLoss.getloss(blur_img, gt_img)
                    temp2 = perceptualLoss.getloss(deblur_img, gt_img)

                    if isinstance(gt_img, numpy.ndarray) :
                        gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
                    deblur_img = torch.from_numpy(deblur_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
                    lpips_loss = lpipsLoss.forward(deblur_img, gt_img)

                    dVGG_blur['all'].append(temp1)
                    dVGG_deblur['all'].append(temp2)
                    lpips_list['all'].append(lpips_loss.item())

                    for b_range in BLUR_RANGE:
                        if b_range in blur_filename:
                            dVGG_blur['{}'.format(b_range)].append(temp1)
                            dVGG_deblur['{}'.format(b_range)].append(temp2)
                            lpips_list['{}'.format(b_range)].append(lpips_loss)
                    log = '[GT : {}] [BLUR : {}] [blur dVGG : {}] [deblur dVGG : {}] [deblur LPIPS : {}]'.format(
                        gt_filename,
                        blur_filename,
                        temp1.item(),
                        temp2.item(),
                        lpips_loss.item(),
                    )
                    f_test.write(str(log))
                    f_test.write('\n') 
                    n += 1
    
    log = '='*60
    print(log)
    f_test.write(str(log))
    f_test.write('\n')

    for b_range in BLUR_RANGE:
        if len(dVGG_blur['{}'.format(b_range)]) is not 0:
            log = '[Blur Range: {}] Average -- [blur dVGG: {:.6f}] [deblur dVGG: {:.6f}] [deblur LPIPS : {}]'.format(
                b_range,
                sum(dVGG_blur['{}'.format(b_range)])/len(dVGG_blur['{}'.format(b_range)]),
                sum(dVGG_deblur['{}'.format(b_range)])/len(dVGG_deblur['{}'.format(b_range)]),
                sum(lpips_list['{}'.format(b_range)])/len(lpips_list['{}'.format(b_range)])
            )

            print(log)
            f_test.write(str(log))
            f_test.write('\n')

    log = '='*60
    print(log)
    f_test.write(str(log))
    f_test.write('\n')

    if len(dVGG_blur['{}'.format(b_range)]) is not 0:
        log = 'Total Average -- [blur dVGG: {:.6f}] [deblur dVGG: {:.6f}] [deblur LPIPS : {}]'.format(
                sum(dVGG_blur['all']) / len(dVGG_blur['all']),
                sum(dVGG_deblur['all']) / len(dVGG_deblur['all']),
                sum(lpips_list['all']) / len(lpips_list['all'])
        )
        print(log)
        f_test.write(str(log))
        f_test.write('\n')
        f_test.close()

if __name__ == '__main__':

    testset = ['CelebA', 'CelebA-HQ', 'FFHQ']
    type = ['center', 'random']

    # testset = ['CelebA-HQ']
    # type = ['center']

    experiment = '220616_V05_DFPGNet+V04_Weighted_resume_LPIPS'

    for t in type :
        for tests in testset :
            ORIG_DIR = os.path.join('/media/data1/hsj/dataset/MSPL_TestData', 'Test_data_{}'.format(t),'Test_data_{}'.format(tests),'{}_gt'.format(tests))
            BLUR_DIR = os.path.join('/media/data1/hsj/dataset/MSPL_TestData', 'Test_data_{}'.format(t),'Test_data_{}'.format(tests),'{}_blur'.format(tests))
            # DEBLUR_DIR = '/media/data1/hsj/codes/deblurring_unetgan/single_deblur_unetgan/test/Test_data_CelebA-HQ'
            DEBLUR_DIR = os.path.join('/media/data1/hsj/codes/deblurring_unetgan/single_deblur_unetgan/results', experiment, 'MSPL-{}'.format(t), tests)
            SAVE_DIR = os.path.join('/media/data1/hsj/codes/deblurring_unetgan/single_deblur_unetgan/results', experiment, 'MSPL-{}'.format(t),'{}_perceptual.txt'.format(tests))
            # SAVE_DIR = os.path.join('/media/data1/hsj/codes/deblurring_unetgan/single_deblur_unetgan/test/Test_data_CelebA-HQ', '{}_perceptual3.txt'.format(tests))

            # DATASET_TYPE must be => 'MSPL' or 'CVPR18'
            DATASET_TYPE = 'MSPL'
            DATASET_NAME = 'FFHQ'

            # if is_visual is true => save result images. else just save psnr result txtfile
            USE_GPU = True

            main(ORIG_DIR, BLUR_DIR, DEBLUR_DIR, SAVE_DIR, DATASET_TYPE, DATASET_NAME, USE_GPU)