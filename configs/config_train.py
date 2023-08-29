# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C
C.seed = 12345

####################################################################################################

C.trilevelnas_x = "B"

mode_lr= "trilevelnas" #1e-4 or 4e-4 900 epochs
#mode_lr= "trilevelnas_long" #1e-4 1800 epochs
#mode_lr = "sota_502010" #4e-4 1800 epochs
#mode_lr = "sota_502012" #8e-4 1800 epochs, higher lr
#mode_lr = "100epochs"

C.attention = True
#C.attention = False

#mode_attn = "no_attn"
mode_attn = "tmp"

#mode_data = "div_flickr"
#mode_data = "div_flickr_224"
mode_data = "div_flickr_120"

C.hard_gt = True
#C.hard_gt = False

#C.continue_train = True
C.continue_train = False

#mode_main = mode_lr = mode_attn = mode_data = "debug"

image_height = {True:None, False:32}
C.image_height = image_height[C.hard_gt]
image_width = {True:None, False:32}
C.image_width = image_width[C.hard_gt]

C.arch_testing = True
#C.arch_testing = False

#"debug":['epab_spatiochannel','vit_spatial_patched32','separable_channel'],
#['epab_spatiochannel','separable_spatial_patched32','vit_spatial_patched32','separable_channel'],
#"tmp":['epab_spatiochannel','separable_spatial_patched16','separable_channel'],
#"tmp":['epab_spatiochannel','separable_spatial_patched16','vit_spatial_patched16_image32','separable_channel'],
#    "snl_spatial":['epab_spatiochannel','separable_spatial','separable_channel'],
#    "patched_snl_spatial32":['epab_spatiochannel','separable_spatial_patched32','separable_channel'],
#    "patched_snl_spatial16":['epab_spatiochannel','separable_spatial_patched16','separable_channel'],
#    "vit_spatial32":['epab_spatiochannel','vit_spatial_patched32','separable_channel'],
#    "vit_spatial16":['epab_spatiochannel','vit_spatial_patched16','separable_channel']
#'separable_spatial_patched32_original'
#"tmp":['vit_spatial_patched16_image32'],
#"tmp":['epab_spatiochannel','separable_spatial_patched32_original','separable_channel']
#"tmp":['vit_spatial_patched4_image32']
primitives_attn = {
    "debug":['epab_spatiochannel','separable_spatial_patched32','vit_spatial_patched32','separable_channel'],
    "no_attn":[],
    "tmp":['vit_spatial_patched4_image32','vit_spatial_map_patched4_image32',
           'vit_channel_image32','vit_channel_map_image32',
           'vit_spatiochannel_patched16_image32','vit_spatiochannel_map_patched16_image32']
}
C.primitives_attn = primitives_attn[mode_attn]

####################################################################################################

"""please config ROOT_dir and user when u first using"""
C.root_dir = os.path.realpath(".")
print("root_dir:%s"%C.root_dir)

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.output_dir = os.path.join(*[C.root_dir,'output','output_'+exp_time])
os.makedirs(C.output_dir, exist_ok=True)
C.exp_name = 'exp_'+exp_time

"""Data Dir and Weight Dir"""
#################################################################################################### LOCAL
#C.dataset_train_path = "/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/sr_training"
# C.dataset_train_path = "/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/DIV2K/DIV2K_train_LR_bicubic"
#C.dataset_valid_path = "/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/DIV2K/DIV2K_valid_LR_bicubic"
#C.dataset_valid_hr_path = "/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/DIV2K/DIV2K_valid_HR"
#C.generator_A2B = 'ESRGAN/RRDB_PSNR_x4.pth'
#################################################################################################### AWS
#C.dataset_path = "/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_3450"
#"debug":"/scratch_net/kringel/hchoong/github/attention-nas/data/set5_eval_matlab",
#"debug":"/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_3450_patched",
dataset_path = {
    "debug":"/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_120",
    "div_flickr":"/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_3450",
    "div_flickr_224":"/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_224",
    "div_flickr_120":"/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_120"
}
C.dataset_path = dataset_path[mode_data]

#C.dataset_train_path = "/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_3450/train"
#C.dataset_valid_path = "/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_3450/val"
#C.dataset_valid_hr_path = "/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_3450/val_hr"

generator_A2B = {
    True: None,
    False: '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS/ESRGAN/RRDB_PSNR_x4.pth'}
C.generator_A2B = generator_A2B[C.hard_gt]

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.join(C.root_dir, 'furnace'))

"""Image Config"""
#C.num_train_imgs = 3450 #32592
#C.num_eval_imgs = 100
num_train_imgs = {"debug":80,"div_flickr":3450,"div_flickr_224":6938,"div_flickr_120":30581}
#num_train_imgs = {"debug":80,"div_flickr":3450,"div_flickr_224":6938,"div_flickr_120":3000}
num_eval_imgs = {"debug":5,"div_flickr":100,"div_flickr_224":100,"div_flickr_120":100}
C.num_train_imgs = num_train_imgs[mode_data]
C.num_eval_imgs = num_eval_imgs[mode_data]

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""

C.opt = 'Adam'

C.momentum = 0.9
C.weight_decay = 5e-4

C.betas = (0.9, 0.999)
C.num_workers = 8


""" Search Config """
C.grad_clip = 5

# C.layers = 30
C.num_cell = 5
C.op_per_cell = 5
#C.num_up = 2
C.num_up = 0
C.num_cell_path_search = 4

C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.]

C.pretrain = False #pretrain=False means no previous pretrain.
#for finetune_pretrain: set pretrain=False. Otherwise put string.
# C.pretrain = 'ckpt/finetune_pretrain/transfer_div2k_1e-5/weights_179.pt'
########################################

batch_size = {"debug":3,"div_flickr":16,"div_flickr_224":1,"div_flickr_120":16}
#batch_size = {"debug":16,"sota":16}
C.batch_size = batch_size[mode_data]


#C.niters_per_epoch = C.num_train_imgs // C.batch_size
niters_per_epoch = {"div_flickr":C.num_train_imgs // C.batch_size,"div_flickr_120":200}
C.niters_per_epoch = niters_per_epoch[mode_data]
#C.latency_weight = [0, 1e-2,]

C.quantize = False
C.ENABLE_BN = False
C.ENABLE_TANH = True
C.loss_func = 'L1'
C.before_act = True

########################################

if C.pretrain == False:
    C.save = 'finetune_pretrain'

    nepochs = {"debug":900,"sota":900,"trilevelnas":900,"trilevelnas_long":1800,
        "sota_502010":1800,"sota_502012":1800,"100epochs":100}
    C.nepochs = nepochs[mode_lr]

    #eval_epoch = {"debug":1,"div_flickr":10,"div_flickr_224":1,"div_flickr_120":10}
    eval_epoch = {"debug":1,"div_flickr":1,"div_flickr_224":1,"div_flickr_120":10}
    C.eval_epoch = eval_epoch[mode_data]

    C.lr_schedule = 'multistep'
    lr = {"debug":1e-4,"sota":1e-4,"trilevelnas":4e-4,"trilevelnas_long":4e-4,
        "sota_502010":4e-4,
        "sota_502012":8e-4,"100epochs":1e-4
        }
    C.lr = lr[mode_lr]

    # linear 
    C.decay_epoch = 300
    # exponential
    C.lr_decay = 0.97
    # multistep
    #C.milestones = [75, 150, 225] #[100, 200 ,300]
    milestones = {"debug":[225, 450, 675],"sota":[225, 450, 675],"trilevelnas":[225, 450, 675],
        "trilevelnas_long":[450, 900, 1350],
        "sota_502010":[450, 900, 1350],"sota_502012":[450, 900, 1350],
        "100epochs":[25, 50, 75]
        }
    #C.milestones = [225, 450, 675]
    #lr=4e-4, milestones=[150,300,450,600], niters_per_epoch=200
    C.milestones = milestones[mode_lr]

    C.gamma = 0.5
    C.loss_weight = [1, 0, 0, 0]

else:
    C.save = "finetune"
    C.nepochs = int(2000)
    C.eval_epoch = int(100)

    C.lr_schedule = 'multistep'
    C.lr = 1e-5
    # linear 
    C.decay_epoch = 300
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [1900]
    C.gamma = 0.5

    C.loss_weight = [1, 0 ,0, 0]   # [1e-2, 0, 1, 5e-8]


C.sparse_type = 'sparsestmax'     #### A #(not applicable to attn part)
C.nf = 64

C.eval_only = False

#C.load_path = 'arch_testing'
#C.load_path = './ckpt/search/search4_psnr_sparsestmax_order_1e-5'    ### B
#C.load_path = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS_attbranch/ckpt/search/exp_2021_12_24_13_54_46/keep_checkpoint_000028.pt'
#False:'./ckpt/search/exp_2021_12_28_06_27_43'
#False:'./ckpt/search/exp_2021_12_30_10_37_57'
#False:'./ckpt/search/exp_2022_04_26_02_25_03'
#False:'./ckpt/search/exp_2022_04_25_18_10_41'}
#False:'./ckpt/search/exp_2022_04_26_02_11_37'}
load_path = {
    True:'arch_testing',
    False:'./ckpt/search/exp_2022_04_25_18_10_41'}
C.load_path = load_path[C.arch_testing]
#C.load_checkpoint_file_name = 'checkpoint.pt'
load_checkpoint_file_name = {True:'arch_testing',False:'checkpoint_000100.pt'}
C.load_checkpoint_file_name = load_checkpoint_file_name[C.arch_testing]

# C.pretrain_load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v2/search/ckpt/finetune/batch2'

C.attn_testing = True
#C.attn_testing = False
C.attn_sink_k = 1
C.attn_levels_t = 1

C.iv_mode = 'image'


#C.continue_train_path = './ckpt/finetune_pretrain/search3_nf32_sparsestmax_order_1e-2_flops_0'
#C.continue_train_path = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS_attbranch/ckpt/finetune_pretrain/exp_2021_12_26_07_04_21/keep_checkpoint_000374.pt'
#C.continue_train_checkpoint_file_name = 'checkpoint.pt'
#True:'ckpt/finetune_pretrain/exp_2021_12_26_07_04_21/keep_checkpoint_000374.pt',
#True:'ckpt/finetune_pretrain/exp_2022_01_03_13_56_22',
#True:'ckpt/finetune_pretrain/exp_2022_01_03_13_56_29',
#True:'ckpt/finetune_pretrain/exp_2022_01_09_10_24_09',
#True:'ckpt/finetune_pretrain/exp_2022_01_19_14_14_38',
#True:'ckpt/finetune_pretrain/exp_2022_01_19_14_15_04',
#True:'ckpt/finetune_pretrain/exp_2022_02_05_17_54_52',
#ckpt/finetune_pretrain/exp_2022_05_07_14_26_53

continue_train_path = {
    True:'ckpt/finetune_pretrain/exp_2022_05_07_14_36_45',
    False:'initial_train'}
C.continue_train_path = continue_train_path[C.continue_train]

#C.continue_train_checkpoint_file_name = 'keep_checkpoint_000740.pt'
C.continue_train_checkpoint_file_name = 'checkpoint_000300.pt'

