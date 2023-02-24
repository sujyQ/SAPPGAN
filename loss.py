import torch.nn.functional as F
import torch.nn as nn
import torch
import util.utils as util
import torchvision.models as py_models

def BCEloss(D_fake, D_real, d_real_target, d_fake_target) :
    real = F.binary_cross_entropy_with_logits(D_real, d_real_target.expand_as(D_real))
    fake = F.binary_cross_entropy_with_logits(D_fake, d_fake_target.expand_as(D_fake))
    return fake, real


def BCEfakeloss(D_fake,target):
    return F.binary_cross_entropy_with_logits(D_fake, target.expand_as(D_fake))


def loss_weighted_dcgan_dis(dis_fake, dis_real):
    _dis_fake = F.softplus(dis_fake)
    _dis_real = F.softplus(-dis_real)
    prob_fake = torch.sigmoid(dis_fake)
    prob_real = torch.sigmoid(dis_real)
    weight_real = 1 - prob_real
    weight_fake = prob_fake
    weight_real = weight_real / (weight_real.sum()+1e-8)
    weight_fake = weight_fake / (weight_fake.sum()+1e-8)
    L_real = torch.sum(_dis_real * weight_real)
    L_fake = torch.sum(_dis_fake * weight_fake)
    return L_fake,  L_real


def loss_weighted_dcgan_gen(dis_fake):
    _dis_fake = F.softplus(-dis_fake)
    # prob_fake = -torch.exp(-dis_fake)
    prob_fake = torch.sigmoid(dis_fake)
    weight = 1 - prob_fake
    weight = weight / (weight.sum()+1e-8)
    loss = torch.sum(_dis_fake * weight)
    return loss


def NegativeCosAmpPhaseloss(img1, img2) :
    fft1 = torch.rfft(img1, signal_ndim=2, onesided=False, normalized=True)
    fft2 = torch.rfft(img2, signal_ndim=2, onesided=False, normalized=True)

    amp1, pha1 = util.extract_ampl_phase(fft1)
    amp2, pha2 = util.extract_ampl_phase(fft2)
    amp2max, _ = torch.max(amp2, dim=2, keepdim=True)
    amp2max, _ = torch.max(amp2max, dim=3, keepdim=True)

    w2 = amp2 / (amp2max + 1e-20)

    # print(amp1.size(), pha1.size())
    inner_product = (pha1 * pha2)
    norm1 = (pha1.pow(2)+1e-20).pow(0.5)
    norm2 = (pha2.pow(2)+1e-20).pow(0.5)

    cos_pha = inner_product / (norm1*norm2 + 1e-20)

    inner_product = (amp1 * amp2)
    norm1 = (amp1.pow(2)+1e-20).pow(0.5)
    norm2 = (amp2.pow(2)+1e-20).pow(0.5)

    cos_amp = inner_product / (norm1*norm2 + 1e-20)
    
    cos = cos_pha + cos_amp

    # print(w2.size())
    # print(cos.size())
    cos = cos * w2

    return -1.0 * cos.mean() 


# for DFPGNet
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        
        checkpoint = torch.load(model_path)
        vgg16 = py_models.vgg16(num_classes=2622)
        vgg16.load_state_dict(checkpoint['state_dict'])
        vgg16.eval()
        
        self.features1 = nn.Sequential(                
                *list(vgg16.features.children())[:3]
            )
        self.features2 = nn.Sequential(                
                *list(vgg16.features.children())[3:8]
            )
        self.features3 = nn.Sequential(                
                *list(vgg16.features.children())[8:15]
            )
        
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = False
        for param in self.features3.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # C=64, W, H
        x1 = self.features1(x)
        # C=128, W/2, H/2
        x2 = self.features2(x1)
        # C=256, W/4, H/4
        x3 = self.features3(x2)
        return x1, x2, x3


class NetUniformLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetUniformLinLayer, self).__init__()
        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        if chn_in==64:
            w = torch.ones(64)/64
            w = w.unsqueeze(0)
            w = w.unsqueeze(2)
            w = w.unsqueeze(3)
            layers[len(layers)-1].weight.data = w
            # print(layers[len(layers)-1].weight.data)
        elif chn_in==128:
            w = torch.ones(128)/128
            w = w.unsqueeze(0)
            w = w.unsqueeze(2)
            w = w.unsqueeze(3)
            layers[len(layers)-1].weight.data = w
            # print(layers[len(layers)-1].weight.data)
        elif chn_in==256:
            w = torch.ones(256)/256
            w = w.unsqueeze(0)
            w = w.unsqueeze(2)
            w = w.unsqueeze(3)
            layers[len(layers)-1].weight.data = w
            # print(layers[len(layers)-1].weight.data)
        self.model = nn.Sequential(*layers)
        

class FeatUniform(nn.Module):
    def __init__(self):
        super(FeatUniform, self).__init__()
        n_feats = [64, 128, 256]
        self.lin0 = NetUniformLinLayer(n_feats[0],use_dropout=True)
        self.lin1 = NetUniformLinLayer(n_feats[1],use_dropout=True)
        self.lin2 = NetUniformLinLayer(n_feats[2],use_dropout=True)  

    def forward(self, diffs): 
        val1 = torch.mean(torch.mean((self.lin0.model(diffs[0])),dim=3),dim=2)
        val2 = torch.mean(torch.mean((self.lin1.model(diffs[1])),dim=3),dim=2)
        val3 = torch.mean(torch.mean((self.lin2.model(diffs[2])),dim=3),dim=2)

        val = val1 + val2 + val3
        val_out = val.view(val.size()[0],val.size()[1],1,1)

        val_out2 = [val1, val2, val3]

        return val_out, val_out2