
import torch
import numpy as np
import random
import torch.optim as optim
import datetime
import os
import shutil
from collections import OrderedDict
import scipy.io
from scipy import signal
import torch.nn as nn
from torchvision.utils import make_grid
import math
import numpy as np
import cv2

# print config
def print_config(config) :
    keys = sorted(config.keys())
    print("=======================config=======================")
    for k in keys:
        print(str(k).ljust(30,"."), config[k] )
    print("====================================================")


# set seed number
def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if use multi-gpus (?)
    np.random.seed(seed)
    random.seed(seed)


# Function to join strings or ignore them
def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])


# load model's weights, optimizer, and the state_dict
def load_weights(G, D, state_dict, resume_root, name_suffix=None, strict=True):

    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, resume_root))
    else:
        print('Loading weights from %s...' % resume_root)

    if G is not None:
        G.load_state_dict(
            torch.load('%s/%s.pth' % (resume_root, join_strings('_', ['G', name_suffix]))),
                                        strict=strict)

    if D is not None:
        D.load_state_dict(
            torch.load('%s/%s.pth' % (resume_root, join_strings('_', ['D', name_suffix]))),
                                        strict=strict)


def load_optim_weights(G_optim, D_optim, G_scheduler, D_scheduler, root, name_suffix=None) :
    if name_suffix :
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading weights from %s...' % root)

    if G_optim is not None :
        G_optim.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix]))))
        G_scheduler.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_scheduler', name_suffix]))))
            
    if D_optim is not None :
        D_optim.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix]))))
        D_scheduler.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['D_scheduler', name_suffix]))))

    return G_optim, D_optim, G_scheduler, D_scheduler


def prepare_optim(G, D, config) :
    G_optim = optim.Adam(params= G.parameters(), lr=config['G_lr'], betas=(config['G_B1'], config['G_B2']), weight_decay=0, eps=config['adam_eps'] )
    D_optim = optim.Adam(params= D.parameters(), lr=config['D_lr'], betas=(config['D_B1'], config['D_B2']), weight_decay=0, eps=config['adam_eps'] )
    if config['lr_scheduler'] == "ExponentialLR" :
        G_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_optim, config['lr_gamma'])
        D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optim, config['lr_gamma'])
    if config['resume'] :
        G_optim, D_optim, G_scheduler, D_scheduler = load_optim_weights(G_optim, D_optim, G_scheduler, D_scheduler, config['resume_from'], name_suffix=config['name_suffix'])

    return G_optim, D_optim, G_scheduler, D_scheduler


# Write some metadata to the logs directory
def write_metadata(logs_root, experiment_name, config, state_dict):
  with open(('%s/%s/metalog.txt' %
             (logs_root, experiment_name)), 'w') as writefile:
    writefile.write('datetime: %s\n' % str(datetime.datetime.now()))
    writefile.write("=======================config=======================")
    keys = sorted(config.keys())
    for k in keys:
        writefile.write(str(k).ljust(30,".")+" : "+str(config[k]) )
    writefile.write(("===================================================="))
    # writefile.write('config: %s\n' % str(config))
    writefile.write('state: %s\n' %str(state_dict))


# write some import python files to the logs directory
def write_files(logs_root, experiment_name, model_name) :
    if os.path.isfile('./train_fns.py') :
        shutil.copy('./train_fns.py', os.path.join(logs_root, experiment_name))
    
    if os.path.isfile('./loss.py') :
        shutil.copy('./loss.py', os.path.join(logs_root, experiment_name))

    if os.path.isfile('./train.py') :
        shutil.copy('./train.py', os.path.join(logs_root, experiment_name))
    
    if os.path.isfile('./util/parser.py') :
        shutil.copy('./util/parser.py', os.path.join(logs_root, experiment_name))
    
    if os.path.isfile('./model/base_model.py') :
        shutil.copy('./model/base_model.py', os.path.join(logs_root, experiment_name))

    if os.path.isfile('./model/DFPGNet_D.py') :
        shutil.copy('./model/DFPGNet_D.py', os.path.join(logs_root, experiment_name))

    nets = model_name.split('+')
    for net in nets :
        if os.path.isfile('./model/'+net+'.py') :
            shutil.copy('./model/'+net+'.py', os.path.join(logs_root, experiment_name))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Logstyle is either:
# '%#.#f' for floating point representation in text
# '%#.#e' for exponent representation in text
class MyLogger(object):
    def __init__(self, fname, reinitialize=False, logstyle='%3.6f'):
        self.root = fname
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.reinitialize = reinitialize
        self.metrics = []
        self.logstyle = logstyle # One of '%3.3f' or like '%3.3e'

    # Delete log if re-starting and log already exists
    def reinit(self, item):
        if os.path.exists('%s/%s.log' % (self.root, item)):
            if self.reinitialize:
                # Only print the removal mess
                if 'sv' in item :
                    if not any('sv' in item for item in self.metrics):
                        print('Deleting singular value logs...')
                else:
                    print('{} exists, deleting...'.format('%s_%s.log' % (self.root, item)))
                os.remove('%s/%s.log' % (self.root, item))

    # Log in plaintext; this is designed for being read in MATLAB(sorry not sorry)
    def log(self, itr, **kwargs):
        for arg in kwargs:
            if isinstance(kwargs[arg],list):
                mylist = "[ " + ",".join([str(e) for e in kwargs[arg]]) + " ]"
                kwargs[arg] = mylist
            if arg not in self.metrics:
                if self.reinitialize:
                    self.reinit(arg)
                self.metrics += [arg]

            with open('%s/%s.log' % (self.root, arg), 'a') as f:
                if isinstance(kwargs[arg],str):
                    f.write( str(itr) + ": "  +  kwargs[arg] + "\n")
                else:
                    f.write('%d: %s\n' % (itr, self.logstyle % kwargs[arg]))


# get blur kernels and return them
def get_blurkernels(kernel_path):
    blur_range = ['k0', 'k13', 'k15', 'k17', 'k19', 'k21', 'k23', 'k25', 'k27']
    kernel_dict = OrderedDict()
    for blur_size in blur_range: 
        if blur_size == 'k0':
            kernel_dict['{}'.format(blur_size)] = None
        else:        
            k_filename_kernel =os.path.join(kernel_path, 'blur_{}.mat'.format(blur_size))
            kernel_file = scipy.io.loadmat(k_filename_kernel)
            kernels = np.array(kernel_file['blurs_{}'.format(blur_size)])
            kernels = kernels.transpose([2,0,1])
            kernel_dict['{}'.format(blur_size)] = kernels
    return kernel_dict


def get_blurtensor(image_tensor, b_range, kernels):
    batch_size = image_tensor.size(0)
    image_clone = image_tensor.clone() 
    image_np = image_clone.numpy()

    blur_img_np = np.copy(image_np)
    
    if b_range != 'k0':
        copy_img_np = np.copy(image_np)
        PADDING = dict(k13=6, k15=7, k17=8, k19=9, k21=10, k23=11, k25=12, k27=13)
        pad_size = PADDING['{}'.format(b_range)]
        n_pad = ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size))
        copy_img_np = np.pad(copy_img_np, n_pad, 'reflect')

        if kernels is not None:
            for j in range (batch_size):
                index = random.randint(0,2249)
                blur_img_np[j,0,:,:]= signal.convolve(copy_img_np[j,0,:,:],kernels[index,:,:],mode='valid')
                blur_img_np[j,1,:,:]= signal.convolve(copy_img_np[j,1,:,:],kernels[index,:,:],mode='valid')
                blur_img_np[j,2,:,:]= signal.convolve(copy_img_np[j,2,:,:],kernels[index,:,:],mode='valid')
    blur_img_np = blur_img_np + (1.0/255.0)* np.random.normal(0,4,blur_img_np.shape) 
    
    blur_tensor = torch.from_numpy(blur_img_np)  
    blur_tensor = blur_tensor.float()

    assert image_tensor.size() == blur_tensor.size()
    return blur_tensor 


# convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off


# save model's weight, optimizer and scheduler
def save_network(G, D, G_optim, D_optim, G_scheduler, D_scheduler, weights_root, name_suffix = None):  

    if name_suffix is not None:
        save_filename = os.path.join(weights_root, 'G_{}_best.pth'.format(name_suffix))
        torch.save(G.state_dict(), save_filename) 
        
        save_filename = os.path.join(weights_root, 'D_{}_best.pth'.format(name_suffix))
        torch.save(D.state_dict(), save_filename)

        save_filename = os.path.join(weights_root, 'G_optim_{}_best.pth'.format(name_suffix))
        torch.save(G_optim.state_dict(), save_filename)

        save_filename = os.path.join(weights_root, 'D_optim_{}_best.pth'.format(name_suffix))
        torch.save(D_optim.state_dict(), save_filename)

        save_filename = os.path.join(weights_root, 'G_scheduler_{}_best.pth'.format(name_suffix))
        torch.save(G_scheduler.state_dict(), save_filename)

        save_filename = os.path.join(weights_root, 'D_scheduler_{}_best.pth'.format(name_suffix))
        torch.save(D_scheduler.state_dict(), save_filename)

    else:                     
        save_filename = os.path.join(weights_root, 'G_last.pth')
        torch.save(G.state_dict(), save_filename) 
        
        save_filename = os.path.join(weights_root, 'D_last.pth')
        torch.save(D.state_dict(), save_filename)

        save_filename = os.path.join(weights_root, 'G_scheduler_last.pth')
        torch.save(G_scheduler.state_dict(), save_filename)

        save_filename = os.path.join(weights_root, 'D_scheduler_last.pth')
        torch.save(D_scheduler.state_dict(), save_filename)

        save_filename = os.path.join(weights_root, 'G_optim_last.pth')
        torch.save(G_optim.state_dict(), save_filename)

        save_filename = os.path.join(weights_root, 'D_optim_last.pth')
        torch.save(D_optim.state_dict(), save_filename)


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

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def tensor2img_list(tensor_list):    
    def _tensor2img_list(img_tensor):
        img_denorm = denorm(img_tensor)
        img_np = tensor2img(img_denorm)
        return img_np
    return [_tensor2img_list(_l) for _l in tensor_list]


def tensor2label(label_tensor, n_label=19, imtype = np.uint8) :
    c_map = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    c_map = np.array(c_map, dtype=np.uint8)
    c_map = torch.from_numpy(c_map[:n_label])

    if n_label == 0 :
        return None

    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1 :
        label_tensor = label_tensor.max(0, keepdim=True)[1]

    size = label_tensor.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

    for label in range(0, n_label) :
        mask = (label == label_tensor[0]).cpu()
        color_image[0][mask] = c_map[label][0]
        color_image[1][mask] = c_map[label][1]
        color_image[2][mask] = c_map[label][2]
    
    label_numpy = tensor2img(color_image)

    return label_numpy


def vis_parsing_maps(parsing_anno, stride=1):
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    return vis_parsing_anno_color


def extract_ampl_phase(fft_im) :
    # print(fft_im.size())
    fft_amp = fft_im[:, :, :, :, 0] ** 2 + fft_im[:, :, :, :, 1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3])
    return in_feat/(norm_factor.expand_as(in_feat)+eps)