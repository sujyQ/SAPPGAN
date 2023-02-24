import torch.nn as nn
import torch
import model_utils
from torch.nn import init

class self_FTLayer(nn.Module):
    def __init__(self, n_feats):
        super(self_FTLayer, self).__init__()

        self.scale = nn.Sequential(            
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats,n_feats, 3, 1, 1),
        )

        self.shift = nn.Sequential(            
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats,n_feats, 3, 1, 1),
        )

    def forward(self, x):

        shift = self.shift(x)
        scale = self.scale(x)
        
        return x * (scale + 1) + shift


class feature_FTLayer(nn.Module):
    def __init__(self, n_feats):
        super(feature_FTLayer, self).__init__()

        self.scale = nn.Sequential(            
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats,n_feats, 3, 1, 1),
        )

        self.shift = nn.Sequential(            
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats,n_feats, 3, 1, 1),
        )

    def forward(self, x, feat):      
        scale = self.scale(feat)
        shift = self.shift(feat)        
        return x * (scale + 1) + shift
    

class Generator(nn.Module):
    def __init__(self, num_baseblocks=5, num_learnerblocks=10, norm_type=None, act_type='leakyrelu', init='ortho'):
        super(Generator, self).__init__()
        self.init = init
        n_feats = [64, 128, 256]
        # head
        self.from_rgb = nn.Conv2d(3, n_feats[0], 3, 1, 1) 
        # tail
        self.to_rgb = nn.Conv2d(n_feats[0], 3, 1, 1)
        # encoder
        en1 = [model_utils.ResGroup(n_feats[0], num_baseblocks, norm_type, act_type)]
        self.en1 = nn.Sequential(*en1)           
        
        self.down1 = nn.Conv2d(n_feats[0], n_feats[1], kernel_size=4, stride=2, padding=1)
        en2 = [model_utils.ResGroup(n_feats[1], num_baseblocks, norm_type, act_type)]
        self.en2 = nn.Sequential(*en2)  
        
        self.down2 = nn.Conv2d(n_feats[1], n_feats[2], kernel_size=4, stride=2, padding=1)
        en3 = [model_utils.ResGroup(n_feats[2], num_baseblocks, norm_type, act_type)]
        self.en3 = nn.Sequential(*en3)

        # vgg learners
        vggf1 = [self_FTLayer(n_feats[0])]
        vggf1.append(model_utils.ResGroup(n_feats[0], num_learnerblocks, norm_type, act_type))
        self.vggf1 = nn.Sequential(*vggf1)

        vggf2 = [self_FTLayer(n_feats[1])]
        vggf2.append(model_utils.ResGroup(n_feats[1], num_learnerblocks, norm_type, act_type))
        self.vggf2 = nn.Sequential(*vggf2)

        vggf3 = [self_FTLayer(n_feats[2])]
        vggf3.append(model_utils.ResGroup(n_feats[2], num_learnerblocks, norm_type, act_type))
        self.vggf3 = nn.Sequential(*vggf3)

        # middle
        middle = [model_utils.ResGroup(n_feats[2], num_baseblocks, norm_type, act_type)]
        self.middle = nn.Sequential(*middle)
            
        # decoder            
        self.ft1 = feature_FTLayer(n_feats[2])    
        de1 = [model_utils.ResGroup(n_feats[2], num_baseblocks, norm_type, act_type)]
        self.de1 = nn.Sequential(*de1)       
        
        self.up1 = nn.ConvTranspose2d(n_feats[2], n_feats[1], kernel_size=4, stride=2, padding=1)
        self.ft2 = feature_FTLayer(n_feats[1]) 
        de2 = [model_utils.ResGroup(n_feats[1], num_baseblocks, norm_type, act_type)]
        self.de2 = nn.Sequential(*de2)

        self.up2 = nn.ConvTranspose2d(n_feats[1], n_feats[0], kernel_size=4, stride=2, padding=1)
        self.ft3 = feature_FTLayer(n_feats[0]) 
        de3 = [model_utils.ResGroup(n_feats[0], num_baseblocks, norm_type, act_type)]
        self.de3 = nn.Sequential(*de3)
        # self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                else:
                    print('Init style not recognized...')
   
    def forward(self, x): 
        x = self.from_rgb(x)
        res = x

        e1 = self.en1(x)
        vggf1 = self.vggf1(e1)
        
        x = self.down1(e1)
        e2 = self.en2(x)
        vggf2 = self.vggf2(e2)

        x = self.down2(e2)
        e3 = self.en3(x)
        vggf3 = self.vggf3(e3)

        mid = self.middle(e3)

        f1= self.ft1(mid, vggf3)        
        x = self.de1(f1)
        x = self.up1(x)

        f2 = self.ft2(x, vggf2)
        x = self.de2(f2)
        x = self.up2(x)

        f3 = self.ft3(x, vggf1)        
        x = self.de3(f3)

        x += res
        x = self.to_rgb(x)
        x = torch.tanh(x) 

        return x, vggf1, vggf2, vggf3