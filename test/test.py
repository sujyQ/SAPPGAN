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

experiment = 'your/experiment/name'
best = 'LPIPS'

MODEL_G_DIR = os.path.join('/path/to/yours/experiment', experiment, 'weights/G_{}_best.pth'.format(best))

testset = ['CelebA', 'CelebA-HQ', 'FFHQ']
type = ['center', 'random']


device = torch.device('cuda')

net_G = mimo.Generator().to(device)
net_G = DFPGNet_G.Generator().to(device)

rgb_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

net_G.eval()
net_G.load_state_dict(torch.load(MODEL_G_DIR), strict=True)

for t in type :
    for tests in testset :
        INPUT_IMG_DIR = os.path.join('/path/to/test/dataset', 'Test_data_{}'.format(t), 'Test_data_{}'.format(tests), '{}_blur'.format(tests))
        REAL_IMG_DIR = os.path.join('/path/to/test/dataset', 'Test_data_{}'.format(t), 'Test_data_{}'.format(tests), '{}_gt'.format(tests))
        OUTPUT_IMG_DIR = os.path.join('/path/to/yours/results', '{}_{}'.format(experiment, best), 'MSPL-{}'.format(t), tests)
        make_dir(OUTPUT_IMG_DIR)

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

                out_tensor = denorm(out[0])
                out_img = tensor2img(out_tensor)

                save_filename = os.path.join(OUTPUT_IMG_DIR, '{}_out.png'.format(img_filename))
                cv2.imwrite(save_filename, out_img)

                count = count+1
                print(img_filename)
                print("[{}] {} is done!".format(count, img_filename))
