
import numpy as np
import torch
import importlib
import os
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import random

import util.parser as parser
import util.utils as util
from model.base_model import G_D
import train_fns
from dataset.celeba_hq import CelebA_HQ, validface
import loss
import util.initializer as init
from tensorboardX import SummaryWriter

def run(config) :

    # update configs and skip init when resuming training
    config = parser.update_config_dicts(config)
    config = parser.update_config_roots(config)
    if config["resume"] :
        print("Skipping initialization for training resumption...")
        config["skip_init"] = True

    # set random seed
    util.seed_rng(config['seed'])

    # prepare root folders if necessary
    parser.prepare_root(config)

    # tensorboard
    board_writer = SummaryWriter(config['tb_root'])

    # setup cudnn.benchmark for free speed, but only if not more than 4 gpus are used     
    print("number of GPUs:", config["num_gpus"])
    if config["num_gpus"] <= 4:
        torch.backends.cudnn.benchmark = True    
    print(":::::::::::/nCUDNN BENCHMARK", torch.backends.cudnn.benchmark, "::::::::::::::" )

    # import the generator and discriminator
    generator_model, discriminator_model = config['model'].split('+')   # e.g., mimo_conada+unet --> mimo_conada, unet
    print(generator_model, discriminator_model)
    generator = importlib.import_module('model.'+generator_model)                    # import model.mimo_conada 
    discriminator = importlib.import_module('model.'+discriminator_model)            # import model.unet 

    # build generator and discriminator
    G = generator.Generator().to(config['device'])
    D = discriminator.Discriminator(output_dim=config['output_dim']).to(config['device'])
    vgg_face = loss.VGG16FeatureExtractor(model_path=config['VGGFace16']).to(config['device'])
    vgg_face.eval()
    init.init_weights(G, init_type=config['G_init'], scale=0.1)
    init.init_weights(D, init_type=config['D_init'], scale=1)
    GD = G_D(G, D)

    # print information about networks
    print('Number of params in G: {} D: {}'.format(
            *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))

    state_dict = {'itr': 0, 'epoch': 0, 'best_PSNR': 0,'best_SSIM': 1.0, 'best_dvgg' : 999, 'best_lpips' : 1.0, 'config': config}

    # if loading from a pre-trained model, load weights
    if config['resume'] :
        print('Loading weights...')
        if config["epoch_id"] != "" :
            epoch_id = config["epoch_id"]
            util.load_weights(G, D, state_dict, config['resume_from'], name_suffix=config['name_suffix'], strict=True)
            state_dict['epoch'] = epoch_id
            state_dict['itr'] = epoch_id*1511
        print('loading weights completed')

    # parallelize the GD module if parallel
    if config['parallel'] :
        GD = nn.DataParallel(GD)

    # prepare loggers
    test_metrics_fname = '%s/%s' % (config['logs_root'], config['experiment_name'])
    train_metrics_fname = '%s/%s' % (config['logs_root'], config['experiment_name'])
    print('Test Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = util.MyLogger(test_metrics_fname,
                                 reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = util.MyLogger(train_metrics_fname,
                             reinitialize=(not config['resume']))


    # prepare datasets
    if config['dataset'] == 'celebaHQ' :
        train_dataset = CelebA_HQ(data_dir=config['celebahq_folder'], 
                                 mode='train', n_classes = config['n_classes'],
                                 imsize=config['imsize'])            
        train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True, pin_memory=True)
        
        # validation 
        valid_dataset = validface(data_dir=config['valid_folder'])        
        valid_loader = DataLoader(valid_dataset, 1, shuffle=False, num_workers=config['num_workers'], drop_last=True, pin_memory=True)
    else : assert False, "Dataset is not ready"
    print("Loaded ", config["dataset"])

    # prepare blur kernels
    train_kernel_dict = util.get_blurkernels(config['kernel_path'])

    # prepare the training function
    FeatUniform = loss.FeatUniform().to(config['device'])
    train = train_fns.training_function(G, D, GD, config, vgg_face, FeatUniform)

    # prepare optimizers and schedulers
    G_optim, D_optim, G_scheduler, D_scheduler = util.prepare_optim(G, D, config)

    # write metadata and train_fns file before training starts
    util.write_metadata(config['logs_root'], config['experiment_name'], config, state_dict)
    util.write_files(config['logs_root'], config['experiment_name'], config['model'])

    print("Beginning training at epoch %d" % state_dict['epoch'])

    batch_time = util.AverageMeter()

    # init validation (debug do not remove)
    if config['resume'] :
        print("="*30) 
        print('Start Valid')
        metrics = train_fns.validation(config, state_dict['epoch'], valid_loader, G, D)
        psnr_avg = metrics['psnr']
        lpips_avg = metrics['lpips']
        print('[Epoch: 0|{}] [Average PSNR : {:.4f}] [Average LPIPS : {:.4f}]'.format(
            config['num_epochs'], psnr_avg, lpips_avg ))     
        print("="*30) 
        print('End Valid')
        test_log.log(int(state_dict['epoch']), **metrics)
        if psnr_avg > state_dict['best_PSNR'] :
            util.save_network(G, D, G_optim, D_optim, G_scheduler, D_scheduler, config['weights_root'], 'PSNR')
            state_dict['best_PSNR'] = psnr_avg
        if lpips_avg < state_dict['best_lpips'] :
            util.save_network(G, D,G_optim, D_optim, G_scheduler, D_scheduler, config['weights_root'], 'LPIPS')
            state_dict['best_lpips'] = lpips_avg
        util.save_network(G, D, G_optim, D_optim, G_scheduler, D_scheduler, config['weights_root'])

    for epoch in range(state_dict['epoch'], config['num_epochs']) :
        end = time.time()
        for iters, batch_data in enumerate(train_loader) :
            # get batch data
            gt_img, gt_segmap, _ = batch_data
            gt_img.to(config['device'])
            gt_segmap[:, 0, :, :] = gt_segmap[:, 0, :, :]*255.0
            gt_segmap = gt_segmap[:, 0, : ,:].cuda().long()

            # blur gt_img
            blur_range, kernels = random.choice(list(train_kernel_dict.items()))
            blur_img = util.get_blurtensor(gt_img, blur_range, kernels)

            # increment the iteration counter
            state_dict['itr'] += 1

            # make networks are in training mode
            G.train()
            D.train()

            # input imgs don't need grad
            gt_img.requires_grad = False
            gt_segmap.requires_grad = False

            # get metrics values from train_fns
            metrics = train(blur_img, gt_img, gt_segmap, G_optim, D_optim, epoch+1)

            batch_time.update(time.time()-end)
            end = time.time()
            print("="*30)     
            print('Experiment Name:', config['experiment_name'])       
            print('GPU Device:', config['gpus'])            
            print('[Epoch: {}|{}] [Iter: {}|{}({:.3f}s)]'.format(
                epoch, config['num_epochs'], iters, len(train_loader), batch_time.avg
                ))
            for loss_name, value in metrics.items():
                log = '\t[{} : {:.6f}]'.format(loss_name, value)
                print(log)
            if (iters+1)%20 == 0 :
                train_log.log(itr = int(state_dict['itr']), **metrics) 
            for arg in metrics :
                board_writer.add_scalar('{}'.format(arg), metrics[arg], epoch+1)

        # adjust learning rate schedulers
        D_scheduler.step()
        G_scheduler.step()

        # validation every epoch
        print("="*30) 
        print('Start Valid')
        metrics = train_fns.validation(config, epoch, valid_loader, G, D)
        psnr_avg = metrics['psnr']
        lpips_avg = metrics['lpips']
        print('[Epoch: {}|{}] [Average PSNR : {:.4f}]'.format(
            epoch, config['num_epochs'], psnr_avg ))     
        print("="*30) 
        print('End Valid')
        test_log.log(int(epoch), **metrics)
            
        if psnr_avg > state_dict['best_PSNR'] :
            util.save_network(G, D, G_optim, D_optim, G_scheduler, D_scheduler, config['weights_root'], 'PSNR')
            state_dict['best_PSNR'] = psnr_avg
        if lpips_avg < state_dict['best_lpips'] :
            util.save_network(G, D, G_optim, D_optim, G_scheduler, D_scheduler, config['weights_root'], 'LPIPS')
            state_dict['best_lpips'] = lpips_avg
        util.save_network(G, D, G_optim, D_optim, G_scheduler, D_scheduler, config['weights_root'])

        # increment epoch counter at end of epoch
        state_dict['epoch'] += 1


def main() :
    pars = parser.prepare_parser()
    config = vars(pars.parse_args())

    if config["gpus"] !="" :
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
        config["device"] = 'cuda'
    else : config["device"] = 'cpu'

    config["num_gpus"] = len(config["gpus"].replace(",", ""))

    config["random_num"] = str(int(np.random.rand()*1000000)) + "_" + config["id"]

    new_root = os.path.join(config["base_root"], config["experiment_name"])
    if not os.path.isdir(new_root) :
        os.makedirs(new_root)
        os.makedirs(os.path.join(new_root, "samples"))
        os.makedirs(os.path.join(new_root, "weights"))
        os.makedirs(os.path.join(new_root, "logs"))
        print("created ", new_root)
    config["base_root"] = new_root

    if config['experiment_name'] == "" :
        config['experiment_name'] = config['random_num']
    print('Experiment name is %s' % config['experiment_name'])
    print("::: weights will be saved at ", '/'.join([config['weights_root'], config['experiment_name']]) )

    util.print_config(config)

    run(config)


if __name__ == '__main__' :
    main()