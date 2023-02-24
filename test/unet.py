"""
    Reference:
        https://github.com/ajbrock/BigGAN-PyTorch (MIT license)
        https://github.com/boschresearch/unetgan
"""
from multiprocessing.connection import deliver_challenge
from unittest import skip

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# from model import spectral
import spectral

import numpy as np

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


# Residual block for the discriminator
class DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, wide=True, 
                downsample=False, upsample=False):
        super(DBlock, self).__init__()        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.activation = nn.ReLU(inplace=False)
        
        if (downsample == True) and (upsample == True):
            raise ValueError("Both 'downsample' and 'upsample' can not be True" ) 
        self.sample_mode = False
        if downsample:
            self.sample_mode = 'down'
        elif upsample:
            self.sample_mode = 'up'

        self.conv1 = spectral.SNConv2d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = spectral.SNConv2d(self.hidden_channels, self.out_channels, kernel_size=3, padding=1)

        if self.learnable_sc :
            self.conv_sc = spectral.SNConv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
    
    def up_forward(self, x):
        h = self.activation(x)
        h = F.interpolate(h, scale_factor=2)        
        h = self.conv1(h)        
        h = self.conv2(self.activation(h))

        x = F.interpolate(x, scale_factor=2)
        if self.learnable_sc:            
            x = self.conv_sc(x)
        
        return h + x

    def dn_forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        h = F.avg_pool2d(h, 2)
        
        if self.learnable_sc:            
            x = self.conv_sc(x)
            x = F.avg_pool2d(x, 2)
        
        return h + x

    def forward(self, x):        
        if self.sample_mode == 'up':
            out = self.up_forward(x)
        elif self.sample_mode == 'down':
            out = self.dn_forward(x)
        else:
            h = self.conv1(x)
            h = self.conv2(self.activation(h))

            if self.learnable_sc:
                x = self.conv_sc(x)
            out = h + x
        
        return out


def D_unet_arch(ch=64, attention='64'):
    arch = {}
    n = 2
    # covers bigger perceptual fields
    arch[128] = {'in_channels' :       [3] + [ch*item for item in       [1, 2, 4, 8, 16, 8*n, 4*2, 2*2, 1*2,1]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 8,   4,   2,    1,  1]],
                             'downsample' : [True]*5 + [False]*5,
                             'upsample':    [False]*5+ [True] *5,
                             'resolution' : [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,11)}}


    arch[256] = {'in_channels' :            [3] + [ch*item for item in [1, 2, 4, 8, 8, 16, 8*2, 8*2, 4*2, 2*2, 1*2  , 1         ]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 8,   8,   4,   2,   1,   1          ]],
                             'downsample' : [True] *6 + [False]*6 ,
                             'upsample':    [False]*6 + [True] *6,
                             'resolution' : [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256 ],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,13)}}

    return arch


class Discriminator(nn.Module):
    def __init__(self, resolution=128, n_classes = 19, output_dim=1, init='ortho', conditional = True):
        super(Discriminator, self).__init__()        
        # Init channel
        self.ch = 64
        # Resolution
        self.resolution = resolution
        # ReLu...
        self.activation = nn.ReLU(inplace=False)
        
        self.output_dim = output_dim
        self.init = init
        self.conditional = conditional
        self.n_classes = n_classes
                 
        if self.resolution==128:
            self.save_features = [0,1,2,3,4]
        elif self.resolution==256:
            self.save_features = [0,1,2,3,4,5]

        self.out_channel_multiplier = 1#4
        # Architecture
        self.arch = D_unet_arch(self.ch)[resolution]
        
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            if self.arch["downsample"][index]:
                self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                        out_channels=self.arch['out_channels'][index],
                                        wide=True,
                                        downsample=True if self.arch['downsample'][index] else False,
                                        upsample=False)]]

            elif self.arch["upsample"][index]:               
                self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                        out_channels=self.arch['out_channels'][index],
                                        wide=True,
                                        downsample=False, 
                                        upsample=True if self.arch['upsample'][index] else False )]]

            # If attention on this block, attach it to the end
            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition: #index < 5
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                print("index = ", index)
                self.blocks[-1] += [spectral.Attention(self.arch['out_channels'][index])]
            
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.blocks.append(nn.Conv2d(self.ch, self.output_dim, kernel_size=1))

        self.enc_linear = spectral.SNLinear(16*self.ch, 1)
        self.dec_linear = spectral.SNLinear(self.arch['out_channels'][-1], self.output_dim)

        if self.conditional :
            self.embedding = spectral.SNEmbedding(self.n_classes, self.arch['out_channels'][-1])
            # self.proj_linear = spectral.SNLinear(16*self.ch, self.arch['out_channels'][-1])

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                else:
                    print('Init style not recognized...')

    def forward(self, x, label=None):
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        enc_out = None
        dec_out = None
        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index==6 :
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==7:
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==8:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[1]),dim=1)

            if self.resolution == 256:
                if index==7:
                    h = torch.cat((h,residual_features[5]),dim=1)
                elif index==8:
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==10:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==11:
                    h = torch.cat((h,residual_features[1]),dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index==self.save_features[-1]:
                # enc_out = self.activation(h)
                h_ = torch.sum(self.activation(h), [2,3])
                enc_out = self.enc_linear(h_)
                if self.conditional :
                    h_keep = h_
                    

        dec_out = self.blocks[-1](h)

        if self.conditional and label is not None :
            # print('*'*30)
            # print(label.shape)
            emb = self.embedding(label)
            # print(emb.shape)
            emb = emb.view(emb.size(0), emb.size(-1), self.resolution, self.resolution)
            # print(emb.shape)
            # print(h.shape)
            # print(emb.shape == h.shape)
            # print(torch.mul(emb, h).shape)
            # print(h_keep.size())
            # print( self.proj_linear(h_keep).unsqueeze(dim=2).unsqueeze(dim=3).size())
            # print(emb.size())
            # print(h.size())

            # h_keep = self.proj_linear(h_keep).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(emb)
            # proj = torch.sum(emb * h_keep, 1, keepdim=True)
            proj = torch.sum(emb * h, 1, keepdim=True)
            # print(proj.shape)
            dec_out += proj
            # print(dec_out.shape)
        dec_out = dec_out.view(dec_out.size(0), self.output_dim, self.resolution, self.resolution)
    
        return enc_out, dec_out


#-----------------------------------------
# Debug
if __name__ == '__main__':
    import os
    import time
    os.environ["CUDA_VISIBLE_DEVICES"] =  "0"
    torch.backends.cudnn.benckmark = True           
    device = torch.device('cuda')

    image_size = 128
    batch_size = 16

    input_tensor = torch.rand([batch_size, 3, image_size, image_size]).to(device)
    conditional_input = torch.ones([input_tensor.size(0)], dtype=torch.long)
    conditional_input_tensor = torch.LongTensor(conditional_input).cuda()
    netD = Discriminator().to(device)
    P_enc_out, P_dec_out = netD(input_tensor, conditional_input_tensor)
    print("="*30)    
    print(P_enc_out.size())
    print(P_dec_out.size())