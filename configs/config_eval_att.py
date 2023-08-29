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

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
"""please config ROOT_dir and user when u first using"""
# C.repo_name = 'Autodeep_SR'
# C.abs_dir = osp.realpath(".")
# C.this_dir = C.abs_dir.split(osp.sep)[-1]
# C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
# C.root_dir = '/scratch_net/zinc/wuyan/code/AutoSR/Autodeep_SR'
# C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.this_dir))
# C.log_dir_link = osp.join(C.abs_dir, 'log')
# C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
# C.log_file = C.log_dir + '/log_' + exp_time + '.log'
# C.link_log_file = C.log_file + '/log_last.log'
# C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
# C.link_val_log_file = C.log_dir + '/val_last.log'

C.output_dir = "/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS/output/eval" + '/eval_' + exp_time
os.makedirs(C.output_dir, exist_ok=True)

"""Data Dir and Weight Dir"""

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

#add_path(osp.join(C.root_dir, 'furnace'))

"""Image Config"""

#C.num_train_imgs = 3450
#C.num_eval_imgs = 100

C.num_workers = 4

C.num_cell = 5
C.op_per_cell = 5
C.num_up = 2
C.num_cell_path_search = 4

C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.]  # [s]

C.quantize = False

C.ENABLE_BN = False

C.ENABLE_TANH = True

C.generator_A2B = 'ESRGAN/RRDB_ESRGAN_x4.pth'

#C.dataset_path = ['/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/Set5', '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/Set14']
#C.dataset_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/DIV2K/DIV2K_valid'
#C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/data/div2k/DIV2K_valid_LR_bicubic/X4'

#C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/data/div2k'
#C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/AGD2/AGD_SR/search/data/Set5'
#C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/data/urban100_eval'

C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/data/div2k'
#C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/data/set5_eval_matlab'
#C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/data/set14_eval_matlab'
#C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/data/urban100_eval_matlab'

C.real_measurement = False

#######1
# C.load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_2/search/ckpt/search/search4_psnr_sparsestmax_order_1e-5'


# C.ckpt = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_2/search/ckpt/finetune/inherit_search4_psnr_sparsestmax_order_1e-5/weights.pt'

# ##########2
# C.load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_2/search/ckpt/search/search4_psnr_sparsestmax_order_1e-6'


# C.ckpt = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_2/search/ckpt/finetune/inherit_search4_psnr_sparsestmax_order_1e-6/weights_1079.pt'

# #######3
# C.load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_2/search/ckpt/search/search4_psnr_sparsestmax_order_1e-5'


# C.ckpt = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_3/search/ckpt/finetune/transfer_div2k_1e-5/weights_59.pt'

# # ####### 4
# C.load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_3/search/ckpt/search/search4_psnr_div2k_sparsestmax_order_1e-5'


# C.ckpt = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_3/search/ckpt/finetune/inherit_search4_psnr_div2k_sparsestmax_order_1e-5/weights_179.pt'

######### 5

#C.load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_2/search/ckpt/search/search4_psnr_sparsestmax_order_1e-5'
#C.load_path = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS/ckpt/search/search4_psnr_sparsestmax_order_1e-5'
C.load_path = './ckpt/search/exp_2021_12_09_19_03_00'

#C.ckpt = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_3/search/ckpt/finetune/transfer_div2k_1e-5/weights_179.pt'
#C.ckpt = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS/ckpt/finetune/search4_psnr_sparsestmax_order_1e-5_div2k/weights_0.pt'
#C.ckpt = 'ckpt/finetune_pretrain/exp_2021_12_13_20_29_17/weights.pt'
C.ckpt = 'ckpt/finetune/exp_2021_12_14_17_23_48/weights.pt'

C.sparse_type = 'sparsestmax'