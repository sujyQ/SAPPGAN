from ctypes import util
import os
import numpy as np
import cv2
from PIL import Image

import torch
from torchvision import transforms
import torchvision.utils

from util import *
import mimo, unet, DFPGNet_G

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

experiment = '220723_V08_GwoFD_projection_weighted_encdecLinear_dec_adv003'
best = 'LPIPS'
# best = 'PSNR'

MODEL_G_DIR = os.path.join('/media/data1/hsj/codes/deblurring_unetgan/single_deblur_unetgan/experiment', experiment, 'weights/G_{}_best.pth'.format(best))
# MODEL_G_DIR = '/media/data1/hsj/codes/deblurring_unetgan/single_deblur_unetgan/experiment/220619_Jung_official/checkpoints/netL_best.pth'
# MODEL_D_DIR = os.path.join('/media/data1/hsj/codes/deblurring_unetgan/single_deblur_unetgan/experiment', experiment, 'weights/D_{}_best.pth'.format(best))

testset = ['CelebA', 'CelebA-HQ', 'FFHQ']
type = ['center', 'random']

# INPUT_IMG_DIR = '/home/hsj/drive/hsj/dataset/MSPL_TrainingData/validationset/blur'
# REAL_IMG_DIR = '/home/hsj/drive/hsj/dataset/MSPL_TrainingData/validationset/gt'
# OUTPUT_IMG_DIR = '/home/hsj/drive/hsj/codes/codes/deblurring-unetgan/single_deblur_unetgan/results/220428_V03_RF_projection_notOnehot/MSPL-center/val'
# make_dir(OUTPUT_IMG_DIR)

device = torch.device('cuda')

net_G = mimo.Generator().to(device)
net_G = DFPGNet_G.Generator().to(device)
# net_D = unet.Discriminator(output_dim=1 ,conditional=True).to(device)

rgb_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

net_G.eval()
net_G.load_state_dict(torch.load(MODEL_G_DIR), strict=True)
# net_D.eval()
# net_D.load_state_dict(torch.load(MODEL_D_DIR), strict=True)

for t in type :
    for tests in testset :
        INPUT_IMG_DIR = os.path.join('/media/data1/hsj/dataset/MSPL_TestData', 'Test_data_{}'.format(t), 'Test_data_{}'.format(tests), '{}_blur'.format(tests))
        REAL_IMG_DIR = os.path.join('/media/data1/hsj/dataset/MSPL_TestData', 'Test_data_{}'.format(t), 'Test_data_{}'.format(tests), '{}_gt'.format(tests))
        OUTPUT_IMG_DIR = os.path.join('/media/data1/hsj/codes/deblurring_unetgan/single_deblur_unetgan/results', '{}_{}'.format(experiment, best), 'MSPL-{}'.format(t), tests)
        make_dir(OUTPUT_IMG_DIR)

        # Discriminator Test
        # count = 0
        # for imgs in os.listdir(REAL_IMG_DIR) :
        #     img_path = os.path.join(REAL_IMG_DIR, imgs)
        #     img_filename = os.path.splitext(os.path.split(img_path)[-1])[0]

        #     with torch.no_grad() :
        #         img_pil = Image.open(img_path)
        #         img_tensor = rgb_to_tensor(img_pil)
        #         img_tensor = torch.unsqueeze(img_tensor, 0)
        #         img_tensor = img_tensor.to(device)

        #         _, out = net_D(img_tensor)

        #         # labels_predict_plain = generate_label_plain(out)
        #         # labels_predict_color = generate_label(labels_predict_plain)
        #         d_map = generate_D_map(out)

        #         img_filename = img_filename.split('_')[0]
        #         print(img_filename)
        #         save_filename = os.path.join(OUTPUT_IMG_DIR, '{}_parse.png'.format(img_filename))
        #         # print(labels_predict_color.shape)
        #         # cv2.imwrite(save_filename, cv2.cvtColor(labels_predict_color, cv2.COLOR_RGB2BGR))
        #         torchvision.utils.save_image(d_map, save_filename)
        #         count = count+1
        #         print("[{}] {} is done!".format(count, img_filename))

        count = 0
        for imgs in os.listdir(INPUT_IMG_DIR) :
            img_path = os.path.join(INPUT_IMG_DIR, imgs)
            img_filename = os.path.splitext(os.path.split(img_path)[-1])[0]

            with torch.no_grad() :
                img_pil = Image.open(img_path)
                img_tensor = rgb_to_tensor(img_pil)
                img_tensor = torch.unsqueeze(img_tensor, 0)
                img_tensor = img_tensor.to(device)

                out = net_G(img_tensor)

                # for i, out_tensor in enumerate(out_list) :
                #     out_tensor = denorm(out_tensor)
                #     out_img = tensor2img(out_tensor)

                #     save_filename = os.path.join(OUTPUT_IMG_DIR, '{}_out{}.png'.format(img_filename, i))
                #     cv2.imwrite(save_filename, out_img)

                out_tensor = denorm(out[0])
                out_img = tensor2img(out_tensor)

                save_filename = os.path.join(OUTPUT_IMG_DIR, '{}_out.png'.format(img_filename))
                cv2.imwrite(save_filename, out_img)

                # _, out = net_D(out_list[-1])
                # parsing = out.squeeze(0).cpu().numpy().argmax(0)
                # print(np.unique(parsing))

                # labels_predict_plain = generate_label_plain(out)
                # labels_predict_color = generate_label(labels_predict_plain)
                # d_map = generate_D_map(out)

                save_filename = os.path.join(OUTPUT_IMG_DIR, '{}_parse.png'.format(img_filename))

                # cv2.imwrite(save_filename,  cv2.cvtColor(labels_predict_color, cv2.COLOR_RGB2BGR))
                # torchvision.utils.save_image(d_map, save_filename)
                count = count+1
                print(img_filename)
                print("[{}] {} is done!".format(count, img_filename))
