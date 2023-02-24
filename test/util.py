import os
import numpy as np
from torchvision.utils import make_grid
import math
import cv2
import torch
import torch.nn.functional as F

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), is_img=True):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.detach().numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.detach().numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        if not is_img:
            img_np = (img_np + 128.0).round().clip(0, 255)
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def vis_parsing_maps(parsing_anno, stride=1):
    part_colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], 
                    [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], 
                    [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], 
                    [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], 
                    [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]


    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    return vis_parsing_anno_color

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 19: # CelebAMask-HQ
        cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                     (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                     (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                     (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153), 
                     (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)], 
                     dtype=np.uint8) 

    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def tensor2label(label_tensor, n_label, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    print(label_tensor.shape)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    print(label_numpy.shape)
    # label_numpy = label_tensor.numpy()
    # label_numpy = label_numpy / 255.0

    return label_numpy


def generate_label(label_plain, imsize=128):
    label_plain = np.array(label_plain)
    label_plain = torch.from_numpy(label_plain)
    label_plain = label_plain.view(1, imsize, imsize)
    label_color = tensor2label(label_plain, 19)
    label_color = np.array(label_color)
    # label_batch = torch.from_numpy(label_batch)	
    return label_color

def generate_label_plain(label, imsize=128):
    label = label.view(1, 19, imsize, imsize)
    label = label.data.max(1)[1].cpu().numpy()  # 1x128x128
    return label

def generate_D_map(D_maps) :
    D_maps = F.sigmoid(D_maps)

    for i in range(D_maps.size(0)) :
        D_maps[i] = D_maps[i] - D_maps[i].min()
        D_maps[i] = D_maps[i]/D_maps[i].max()

    return D_maps.float().cpu()