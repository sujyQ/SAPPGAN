import torch
import functools
import torch.nn as nn
from collections import OrderedDict
import os
from math import log10
import torch.nn.functional as F
import lpips
import loss
import util.utils as util
import cv2
import numpy as np


def training_function(G, D, GD, config, vgg_face, FeatUniform) :

    def train(blur_img, gt_img, gt_segmap, G_optim, D_optim, epoch) :

        # zero grad optims
        G_optim.zero_grad()
        D_optim.zero_grad()

        # define real/fake target
        d_real_target = torch.tensor([1.0]).cuda()
        d_fake_target = torch.tensor([0.0]).cuda()
        g_target = torch.tensor([1.0]).cuda()

        # define loss functions
        discriminator_loss = functools.partial(loss.BCEloss, d_real_target=d_real_target, d_fake_target=d_fake_target)
        adversarial_loss = functools.partial(loss.BCEfakeloss, target=g_target)
        L1_loss = nn.L1Loss().to(config['device'])

        # out contains loss values
        out = {}

        enc_weight = (config['num_epochs'] - epoch) / config['num_epochs'] + 0.5
        dec_weight = 2 - enc_weight

        #####################################
        ######## TRAIN DISCRIMINATOR ########
        #####################################

        # optionally toggle G and D's "require_grad"
        if config['toggle_grads'] :
            util.toggle_grad(D, True)
            util.toggle_grad(G, False)

        # get outputs from network
        _, D_fake_enc_out, D_fake_dec_out, D_real_enc_out, D_real_dec_out = GD(blur_img, gt_img, gt_segmap, train_G=False)

        # calculate losses and backpropagate
        D_fake_enc_loss, D_real_enc_loss = discriminator_loss(D_fake_enc_out, D_real_enc_out)
        D_fake_dec_loss, D_real_dec_loss = loss.loss_weighted_dcgan_dis(D_fake_dec_out, D_real_dec_out)

        D_discriminator_real_loss = D_real_enc_loss + D_real_dec_loss
        D_discriminator_fake_loss = D_fake_enc_loss + D_fake_dec_loss
        D_loss = D_discriminator_real_loss + D_discriminator_fake_loss

        D_loss.backward()
        D_optim.step()

        # save loss items
        D_fake_enc_loss_item = D_fake_enc_loss.detach().item()
        D_real_enc_loss_item = D_real_enc_loss.detach().item()
        D_fake_dec_loss_item = D_fake_dec_loss.detach().item()
        D_real_dec_loss_item = D_real_dec_loss.detach().item()
        D_discriminator_real_loss_item = D_discriminator_real_loss.detach().item()
        D_discriminator_fake_loss_item = D_discriminator_fake_loss.detach().item()
        D_loss_item = D_loss.detach().item()

        del D_loss


        #####################################
        ########## TRAIN GENERATOR ##########
        #####################################

        # optionally toggle D and G's "require_grad"
        if config['toggle_grads'] :
            util.toggle_grad(D, False)
            util.toggle_grad(G, True)

        # get output from the network
        fake_img, D_enc_out, D_dec_out, out_feats = GD(blur_img, gt_img, gt_segmap, train_G=True)

        # calculate losses and backpropagate
        #   adversarial loss
        G_adv_enc_loss = adversarial_loss(D_enc_out)
        G_adv_dec_loss = loss.loss_weighted_dcgan_gen(D_dec_out)
        G_adv_loss = enc_weight * G_adv_enc_loss + dec_weight * G_adv_dec_loss

        #   feature L2 loss
        gtf1, gtf2, gtf3 = vgg_face(gt_img.cuda())
        gt_feats = [gtf1, gtf2, gtf3]
        diffs = []
        for idx in range(len(gt_feats)) :
            gt_normed = util.normalize_tensor(gt_feats[idx])
            pred_normed = util.normalize_tensor(out_feats[idx])
            diffs.append((gt_normed-pred_normed)**2)
        score, _ = FeatUniform(diffs)
        G_feat_L2_loss = torch.mean(score)

        #   pixel loss
        G_pixel_loss = L1_loss(fake_img, gt_img.cuda())

        # G_loss = G_pixel_loss + 0.05*G_adv_loss + 0.1*G_fft_loss + 0.05*G_vgg_loss
        G_loss = G_pixel_loss + 0.03*G_adv_loss + G_feat_L2_loss

        G_loss.backward()
        G_optim.step()

        # save loss items
        G_adv_enc_loss_item = enc_weight * G_adv_enc_loss.detach().item()
        G_pixel_loss_item = G_pixel_loss.detach().item()
        G_adv_dec_loss_item = dec_weight * G_adv_dec_loss.detach().item()
        G_feat_L2_loss_item = G_feat_L2_loss.detach().item()
        G_loss_item = G_loss.detach().item()

        # save intermediate losses
        out['D_fake_enc_loss'] = float(D_fake_enc_loss_item)
        out['D_real_enc_loss'] = float(D_real_enc_loss_item)
        out['D_fake_dec_loss'] = float(D_fake_dec_loss_item)
        out['D_real_dec_loss'] = float(D_real_dec_loss_item)
        out['D_real_loss'] = float(D_discriminator_real_loss_item)
        out['D_fake_loss'] = float(D_discriminator_fake_loss_item)
        out['D_loss'] = float(D_loss_item)
        out['G_adv_enc_loss'] = float(G_adv_enc_loss_item)
        out['G_adv_dec_loss'] = float(G_adv_dec_loss_item)
        out['G_pixel_loss'] = float(G_pixel_loss_item)
        out['G_feat_L2_loss'] = float(G_feat_L2_loss_item)
        out['G_loss'] = float(G_loss_item)

        del G_loss

        return out

    return train


def validation(config, epoch, valid_loader, G, D) :
    G.eval()
    D.eval()

    out = {}

    psnr_result = util.AverageMeter()
    lpips_result = util.AverageMeter()

    lpips_loss = lpips.LPIPS(net='vgg', version='0.1').to(config['device'])

    print(valid_loader.__len__())

    mse = nn.MSELoss().to(config['device'])

    for idx, data in enumerate(valid_loader) :
        gt_img, gt_filename, blur_img, blur_filename = data
        print(idx, blur_filename, gt_filename)
        
        batch_size = gt_img.size(0)
        gt_img = gt_img.to(config['device'])
        blur_img = blur_img.to(config['device'])

        with torch.no_grad() :
            output, _, _, _ = G(blur_img)
            _, segmap_fake = D(output)
            _, segmap_real = D(gt_img)

        loss = mse(gt_img, output)
        psnr = 10 * log10(1/loss.item())
        psnr_result.update(psnr, batch_size)

        lpips_value = lpips_loss.forward(gt_img, output)
        lpips_result.update(lpips_value.item(), batch_size)

        gt_filename = gt_filename[0]
        blur_filename = blur_filename[0]

        save_out_path = os.path.join(config['samples_root'], config['experiment_name'], 'ep_{}'.format(epoch))
        if not os.path.exists(save_out_path) :
            os.makedirs(save_out_path)
        
        out_tensor = output[0, :, :, :]
        out_tensor = util.denorm(out_tensor)
        out_img = util.tensor2img(out_tensor)

        save_name = os.path.join(save_out_path, '{}_out.png'.format(blur_filename))
        cv2.imwrite(save_name, out_img)

        if config['output_dim'] > 1 :
            segmap_fake = segmap_fake.squeeze(0).cpu().numpy().argmax(0)
            segmap_fake_vis = util.vis_parsing_maps(segmap_fake)
            save_name = os.path.join(save_out_path, '{}_parse.png'.format(blur_filename))
            cv2.imwrite(save_name, segmap_fake_vis)

            segmap_real = segmap_real.squeeze(0).cpu().numpy().argmax(0)
            segmap_real_vis = util.vis_parsing_maps(segmap_real)
            save_name = os.path.join(save_out_path, '{}_parse.png'.format(gt_filename))
            cv2.imwrite(save_name, segmap_real_vis)
        
        out['psnr'] = psnr_result.avg
        out['lpips'] = lpips_result.avg

    return out


    