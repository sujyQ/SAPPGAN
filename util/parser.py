from argparse import ArgumentParser
import torch.nn as nn
import os


def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)
    parser.add_argument("--celebahq_folder", type=str, default="path / to / your / CelebAHQ / dataset ")
    parser.add_argument("--valid_folder", type=str, default="path / to / your / validation / dataset ")
    parser.add_argument("--kernel_path", type=str, default="path / to / your / training / kernels")
    parser.add_argument("--id",type=str, default="SAPPGAN_official_code") 
    parser.add_argument('--experiment_name', type=str, default='SAPPGAN_official_code',
        help='Optionally override the automatic experiment naming with this arg.')
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--sample_every", type=int,default=1)
    parser.add_argument("--VGGFace16", type=str, default='path/to/VGGFace16.pth')

    parser.add_argument("--resume_from", type = str, default='path/to/weigts/folder')
    parser.add_argument("--epoch_id", type=int, default=100)
    parser.add_argument(
    '--name_suffix', type=str, default='last',
    help='Suffix for experiment name for loading weights for sampling (default: %(default)s)')

    ### Dataset/Dataloader stuff ###
    parser.add_argument(
        '--dataset', type=str, default='celebaHQ',
        help='Which Dataset to train on (default: %(default)s)')
    parser.add_argument(
        '--n_classes', type=int, default=19,
        help='number of classes for label/segmentation  (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of dataloader workers; consider using less for HDF5 '
            '(default: %(default)s)')
    parser.add_argument(
        '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
        help='Pin data into memory through dataloader? (default: %(default)s)')
    parser.add_argument(
        '--load_in_mem', action='store_true', default=False,
        help='Load all data into memory? (default: %(default)s)')
    parser.add_argument(
        '--use_multiepoch_sampler', action='store_true', default=False,
        help='Use the multi-epoch sampler for dataloader? (default: %(default)s)')
    parser.add_argument(
        '--imsize', type=int, default=128,
        help='size of input/output image (default: %(default)s)')

    ### Model stuff ### 
    parser.add_argument(
        '--model', type=str, default='DFPGNet_G+unet',
        help='Name of the model module, combination of generator and discriminator connected by ''+'' (default: %(default)s)')
    parser.add_argument(
        '--discriminator', type=str, default='UNet',
        help='Name of the model module (default: %(default)s)')
    parser.add_argument(
        '--D_param', type=str, default='SN',
        help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
            ' or None (default: %(default)s)')
    parser.add_argument(
        '--D_ch', type=int, default=64,
        help='Channel multiplier for D (default: %(default)s)')
    parser.add_argument(
        '--D_depth', type=int, default=1,
        help='Number of resblocks per stage in D? (default: %(default)s)')
    parser.add_argument(
        '--D_thin', action='store_false', dest='D_wide', default=True,
        help='Use the SN-GAN channel pattern for D? (default: %(default)s)')
    parser.add_argument(
        '--shared_dim', type=int, default=0,
        help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
            '(default: %(default)s)')
    parser.add_argument(
        '--D_nl', type=str, default='relu',
        help='Activation function for D (default: %(default)s)')
    parser.add_argument(
        '--D_attn', type=str, default='0',
        help='What resolutions to use attention on for D (underscore separated) '
            '(default: %(default)s)')
    parser.add_argument(
        '--norm_style', type=str, default='bn',
        help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
            'ln [layernorm], gn [groupnorm] (default: %(default)s)')

    ### Model init stuff ###
    parser.add_argument(
        '--seed', type=int, default=99,
        help='Random seed to use; affects both initialization and '
            ' dataloading. (default: %(default)s)')
    parser.add_argument(
        '--G_init', type=str, default='kaiming',
        help='Init style to use for G (default: %(default)s)')
    parser.add_argument(
        '--D_init', type=str, default='kaiming',
        help='Init style to use for D(default: %(default)s)')
    parser.add_argument(
        '--FD_init', type=str, default='kaiming',
        help='Init style to use for FD(default: %(default)s)')
    parser.add_argument(
        '--skip_init', action='store_true', default=False,
        help='Skip initialization, ideal for testing when ortho init was used '
                '(default: %(default)s)')

    ### Optimizer stuff ###
    parser.add_argument(
        '--G_lr', type=float, default=1e-4,
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_lr', type=float, default=5e-4,
        help='Learning rate to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--FD_lr', type=float, default=1e-5,
        help='Learning rate to use for Feature Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B1', type=float, default=0.9,
        help='Beta1 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B1', type=float, default=0.9,
        help='Beta1 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B2', type=float, default=0.999,
        help='Beta2 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B2', type=float, default=0.999,
        help='Beta2 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--lr_scheduler', type=str, default='ExponentialLR',
        help='Type of learning rate scheduler (default : %(default)s)')
    parser.add_argument(
        '--lr_gamma', type=float, default=0.99,
        help='gamma to use for learning rate scheduler (defalut " %(default)s)')

    ### Batch size, parallel, and precision stuff ###
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Number of blur images per iteration (default: %(default)s)')

    parser.add_argument(
        '--num_epochs', type=int, default=300,
        help='Number of epochs to train for (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', default=True,
        help='Train with multiple GPUs (default: %(default)s)')

    ### Bookkeping stuff ###
    parser.add_argument(
        '--base_root', type=str, default='./experiments',
        help='Default location to store all weights, samples, data, and logs '
            ' (default: %(default)s)')
    parser.add_argument(
        '--weights_root', type=str, default='weights',
        help='Default location to store weights (default: %(default)s)')
    parser.add_argument(
        '--logs_root', type=str, default='logs',
        help='Default location to store logs (default: %(default)s)')
    parser.add_argument(
        '--samples_root', type=str, default='samples',
        help='Default location to store samples (default: %(default)s)')

    ### Numerical precision ###
    parser.add_argument(
        '--adam_eps', type=float, default=1e-8,
        help='epsilon value to use for Adam (default: %(default)s)')
    parser.add_argument(
        '--BN_eps', type=float, default=1e-5,
        help='epsilon value to use for BatchNorm (default: %(default)s)')
    parser.add_argument(
        '--SN_eps', type=float, default=1e-8,
        help='epsilon value to use for Spectral Norm(default: %(default)s)')

    ### Ortho reg stuff ###
    parser.add_argument(
        '--G_ortho', type=float, default=0.0, # 1e-4 is default for BigGAN
        help='Modified ortho reg coefficient in G(default: %(default)s)')
    parser.add_argument(
        '--D_ortho', type=float, default=0.0,
        help='Modified ortho reg coefficient in D (default: %(default)s)')
    parser.add_argument(
        '--toggle_grads', action='store_true', default=True,
        help='Toggle D and G''s "requires_grad" settings when not training them? '
            ' (default: %(default)s)')

    ### Resume training stuff
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Resume training? (default: %(default)s)')

    return parser

# Utility to peg all roots to a base root
# If a base root folder is provided, peg all other root folders to it.
def update_config_roots(config):
    if config['base_root']:
        print('Pegging all root folders to base root %s' % config['base_root'])
        for key in ['weights', 'logs', 'samples']:
            config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
        config['tb_root'] = '%s/%s' % (config['base_root'], 'logs/tensorboard')
    return config


# Utility to prepare root folders if they don't exist; parent folder must exist
def prepare_root(config):
    for key in ['weights_root', 'logs_root', 'samples_root', 'tb_root']:
        if not os.path.exists(config[key]):
            print('Making directory %s for %s...' % (config[key], key))
            os.mkdir(config[key])


# add settings derived from the user-specified configuration into the config-dict 
def update_config_dicts(config) :
    config['D_activation'] = activation_dict[config['D_nl']]

    return config


# Convenience dicts
dset_dict = {'celebaHQMask' : None, '300vw' : None, '300vw+celebaHQ' : None}
imsize_dict = {'celebaHQMask' : 128, '300vw' : 256, '300vw+celebaHQ' : 256}
root_dict = {'celebaHQMask':None, '300vw' : None, '300vw+celebaHQ' : None}
nclass_dict = {'celebaHQMask':1, '300vw':1, '300vw+celebaHQ' : 1}

# Number of classes to put per sample sheet
classes_per_sheet_dict = {'celebaHQMask':1, '300vw' : 1}
activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True),}