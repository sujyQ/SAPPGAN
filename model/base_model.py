"""
    Reference:
        https://github.com/ajbrock/BigGAN-PyTorch (MIT license)
        https://github.com/boschresearch/unetgan
"""
import torch.nn as nn
import torch
import os

class G_D(nn.Module):
    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D


    def forward(self, blur_img, gt_img, label=None, train_G=False):
        # print(hq_img.size())
        # If training G, enable grad tape        
        
        with torch.set_grad_enabled(True):
            if train_G:
                # fake image
                fake_img, outf1, outf2, outf3 = self.G(blur_img)
                out_feats = [outf1, outf2, outf3]
                D_enc_out, D_dec_out = self.D(fake_img, label)

                return (
                    fake_img, D_enc_out, D_dec_out, out_feats                  
                    )               

            else:
                # real image  
                D_real_enc_out, D_real_dec_out = self.D(gt_img, label)

                # fake image
                fake_img, _, _, _ = self.G(blur_img)
                D_fake_enc_out, D_fake_dec_out = self.D(fake_img, label)
                
                return (
                    fake_img, D_fake_enc_out, D_fake_dec_out, D_real_enc_out, D_real_dec_out
                    )