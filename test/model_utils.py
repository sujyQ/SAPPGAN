import torch

def get_norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def get_act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def normalize_tensor(in_feat,eps=1e-10):
    # norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3]).repeat(1,in_feat.size()[1],1,1)
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3])
    return in_feat/(norm_factor.expand_as(in_feat)+eps)

import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm


class SEModule(nn.Module):
    def __init__(self, n_feats=64, reduction=16):
        super(SEModule, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.channel_attention = nn.Sequential(
                nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.channel_attention(y)
        return x * y


class ResBlock(nn.Module):
    """ ResNet Block composed of 2 conv blocks"""
    def __init__(self, n_feats, norm_type=None, act_type='leakyrelu', use_channel_attention=True):
        super(ResBlock, self).__init__()

        blocks = []
        for i in range(2):
            if norm_type == 'spectral':
                blocks.append(SpectralNorm(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)))
            else:
                blocks.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True))
                if norm_type:
                    blocks.append(get_norm(norm_type, n_feats))
            if i == 0 and act_type:
                blocks.append(get_act(act_type))        
        if use_channel_attention:
            blocks.append(SEModule(n_feats))
        self.blocks = nn.Sequential(*blocks) 

    def forward(self, x):
        res = self.blocks(x)
        output = res + x
        return output


class ResGroup(nn.Module):    
    def __init__(self, 
                 n_feats=64, 
                 n_resblocks=2,
                 norm_type=None, 
                 act_type='leakyrelu'):
        super(ResGroup, self).__init__()
        blocks = []
        for i in range(n_resblocks):            
            blocks.append(ResBlock(n_feats, norm_type, act_type))
        self.main = nn.Sequential(*blocks) 
        
    def forward(self, x):
        res = self.main(x)
        output = res + x
        return output