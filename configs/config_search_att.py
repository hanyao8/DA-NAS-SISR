# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import os

C = edict()
config = C
cfg = C

C.distributed = False
C.seed = 12345

"""please config ROOT_dir and user when u first using"""
#C.root_dir = '/scratch_net/zinc/wuyan/code/AutoSR/Autodeep_SR'
#C.root_dir = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS'
C.root_dir = osp.realpath(".")
print("root_dir:%s"%C.root_dir)

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

C.output_dir = "/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS/output" + '/output_' + exp_time
os.makedirs(C.output_dir, exist_ok=True)

"""Data Dir and Weight Dir"""
################################################################################################### LOCAL
#C.dataset_path = "/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/sr_training"                        ######## A
#C.generator_A2B = 'ESRGAN/RRDB_PSNR_x4.pth'                                                                          ######## B
#################################################################################################### AWS
# C.dataset_path = "/mnt/efs/fs1/data/X4_sub_3450"
# C.generator_A2B = '/mnt/efs/fs1/AutoSR/ESRGAN/RRDB_PSNR_x4.pth'
####################################################################################################
#C.dataset_path = "/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/sr_training"                        ######## A
#C.dataset_path = "/scratch_net/kringel/hchoong/github/attention-nas/data/div2k"
#C.dataset_path = "/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_combined_2"
#C.dataset_path = "/scratch_net/kringel/hchoong/github/attention-nas/AGD2/AGD_SR/search/data/Set5"

#C.dataset_path = "/scratch_net/kringel/hchoong/github/attention-nas/data/set5_eval_matlab"
#C.dataset_path = "/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_3"
C.dataset_path = "/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_3450"

#C.generator_A2B = 'ESRGAN/RRDB_PSNR_x4.pth'
C.generator_A2B = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS/ESRGAN/RRDB_PSNR_x4.pth'


"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'furnace'))

"""Image Config"""

C.num_train_imgs = 3450
#C.num_train_imgs = 900
#C.num_train_imgs = 3150
#C.num_train_imgs = 5

C.num_eval_imgs = 100
#C.num_eval_imgs = 100
#C.num_eval_imgs = 400
#C.num_eval_imgs = 5


""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""

C.opt = 'Adam'

C.momentum = 0.9
C.weight_decay = 5e-4

C.betas = (0.9, 0.999)
C.num_workers = 2

""" Search Config """
C.grad_clip = 5

C.prun_modes = 'arch_ratio'

C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.]

C.loss_func = 'L1'
#C.loss_func = 'MSE'

C.before_act = True

#######################################################################
C.num_cell = 5
C.op_per_cell = 5
C.num_up = 2

C.num_cell_path_search = 4    ### A

#C.pretrain = True
#C.pretrain = './ckpt/pretrain/test'
#C.pretrain = './ckpt/pretrain/test'
C.pretrain = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS/ckpt/pretrain/exp_2021_12_08_22_36_34'

#######################################
C.nf = 64                    ### B
C.sparse_type = 'sparsestmax'  ### C

#C.exp_name = 'test'   ### D
C.exp_name = 'exp_'+exp_time

C.Lambda_step = 0
C.Lambda_init = 0.001

#C.infer = False
C.infer = True

#C.save_epoch = 10

C.continue_train = False
#C.continue_train = True

#C.continue_train_path = './ckpt/pretrain/search2_nf64'
C.continue_train_path = './ckpt/search/exp_2021_12_09_19_03_00'

C.load_pretrain = False
C.load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v2/search/ckpt/search/AGD_pretrained/'
C.pretrain_load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v2/search/ckpt/finetune/AGD_pretrained/'

#C.calculate_flops=False
C.calculate_flops=True

if C.pretrain == True:
    C.batch_size = 3
    #C.batch_size = 8

    C.niters_per_epoch = C.num_train_imgs // 2 // C.batch_size
    C.latency_weight = [0, 0]
    C.image_height = 32 # this size is after down_sampling
    C.image_width = 32
    C.save = "pretrain"
 
    scale = 0.05


    C.nepochs = 100
    #C.nepochs = 25
    #C.eval_epoch = 20
    C.eval_epoch = 10

    C.lr_schedule = 'multistep'
    C.lr = 1e-4
    # linear 
    C.decay_epoch = 300
    # exponential
    C.lr_decay = 0.97
    # multistep
    # C.milestones = [int(1000*scale), int(2000*scale), int(4000*scale), int(6000*scale)]
    C.milestones = [25, 50, 75]
    C.gamma = 0.5

    # C.loss_weight = [1, 0, 0.006, 2e-8]
    C.loss_weight = [1, 0, 0, 0, 0]

else:
    C.batch_size = 3
    #C.batch_size = 1

    C.niters_per_epoch = C.num_train_imgs // 2 // C.batch_size

    #C.latency_weight = [0, 1e-2,]

    C.image_height = 32 # this size is after down_sampling
    C.image_width = 32
    C.save = "search"

    scale = 0.05

    # C.nepochs = int(8000*scale)
    # C.eval_epoch = int(400*scale)

    C.nepochs = 100
    #C.nepochs = 25
    #C.nepochs = 3

    #C.eval_epoch = 20
    C.eval_epoch = 3
    #C.eval_epoch = 1

    C.lr_schedule = 'multistep'
    C.lr = 1e-4
    # linear 
    C.decay_epoch = 300
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [25, 50, 75]
    C.gamma = 0.5

    # C.loss_weight = [1, 0, 0.006, 2e-8]
    #C.loss_weight = [1, 0, 0, 0, 1e-5]   # 
    C.loss_weight = [1, 0, 0, 0, 0]

########################################

C.ENABLE_BN = False

C.ENABLE_TANH = True

C.quantize = False

C.slimmable = True

C.train_portion = 0.5

C.unrolled = False

C.arch_learning_rate = 3e-4

C.alpha_weight = 2/7
C.ratio_weight = 5/7
C.beta_weight = 0
C.flops_weight = 1e-20                             ### F

C.flops_max = 400e9

C.flops_min = 100e9

# C.loss_weight = [1e-2, 0, 1, 0]



