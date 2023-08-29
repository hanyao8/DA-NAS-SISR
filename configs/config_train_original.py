# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 12345
"""please config ROOT_dir and user when u first using"""
# C.repo_name = 'Autodeep_SR'
# C.abs_dir = osp.realpath(".")
# C.this_dir = C.abs_dir.split(osp.sep)[-1]
# C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.root_dir = '/scratch_net/zinc/wuyan/code/AutoSR/Autodeep_SR'
# C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.this_dir))
# C.log_dir_link = osp.join(C.abs_dir, 'log')
# C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
# C.log_file = C.log_dir + '/log_' + exp_time + '.log'
# C.link_log_file = C.log_file + '/log_last.log'
# C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
# C.link_val_log_file = C.log_dir + '/val_last.log'


"""Data Dir and Weight Dir"""
#################################################################################################### LOCAL
C.dataset_train_path = "/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/sr_training"
# C.dataset_train_path = "/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/DIV2K/DIV2K_train_LR_bicubic"
C.dataset_valid_path = "/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/DIV2K/DIV2K_valid_LR_bicubic"
C.dataset_valid_hr_path = "/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/DIV2K/DIV2K_valid_HR"
C.generator_A2B = 'ESRGAN/RRDB_PSNR_x4.pth'
#################################################################################################### AWS
# C.dataset_train_path = "/mnt/efs/fs1/code/BasicSR/BasicSR/datasets/DIV2K/DIV2K_train_LR_bicubic"
# C.dataset_valid_path = "/mnt/efs/fs1/code/BasicSR/BasicSR/datasets/DIV2K/DIV2K_valid_LR_bicubic"
# C.dataset_valid_hr_path = "/mnt/efs/fs1/code/BasicSR/BasicSR/datasets/DIV2K/DIV2K_valid_HR"
# C.generator_A2B = '/mnt/efs/fs1/AutoSR/ESRGAN/RRDB_PSNR_x4.pth'

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'furnace'))

"""Image Config"""

C.num_train_imgs = 3450#32592
C.num_eval_imgs = 100

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""


# C.lr = 0.0001
# C.lr_decay = 1


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
C.num_up = 2
C.num_cell_path_search = 4

C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.]

C.pretrain = False
# C.pretrain = 'ckpt/finetune_pretrain/transfer_div2k_1e-5/weights_179.pt'
########################################

C.batch_size = 16
C.niters_per_epoch = C.num_train_imgs // C.batch_size
C.latency_weight = [0, 1e-2,]
C.image_height = 32 # this size is after down_sampling
C.image_width = 32

C.quantize = False

C.ENABLE_BN = False

C.ENABLE_TANH = True

C.loss_func = 'L1'

C.before_act = True

########################################

if C.pretrain == False:
    C.save = 'finetune_pretrain'
    C.nepochs = int(300)
    C.eval_epoch = int(30)

    C.lr_schedule = 'multistep'
    C.lr = 2e-4
    # linear 
    C.decay_epoch = 300
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [75, 150, 225] #[100, 200 ,300]
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

C.sparse_type = 'sparsestmax'     #### A
C.load_path = './ckpt/search/search4_psnr_sparsestmax_order_1e-5'    ### B
# C.pretrain_load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v2/search/ckpt/finetune/batch2'
C.continue_train = False
C.continue_train_path = './ckpt/finetune_pretrain/search3_nf32_sparsestmax_order_1e-2_flops_0'

C.output_dir = C.save
C.eval_only = False
C.exp_name = 'test'    ### C
C.nf = 64