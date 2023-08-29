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
C.root_dir = osp.realpath(".")
print("root_dir:%s"%C.root_dir)

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.output_dir = os.path.join(*[C.root_dir,'output','output_'+exp_time])
C.exp_name = 'exp_'+exp_time

"""Data Dir and Weight Dir"""
#C.dataset_path = ['/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/Set5', '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/Set14']
#C.dataset_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/DIV2K/DIV2K_valid'
#C.dataset_path = "/scratch_net/kringel/hchoong/github/attention-nas/data/div_flickr_3450"
#C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/data/set5_eval_matlab'
C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/data/set14_eval_matlab'
#C.dataset_path = '/scratch_net/kringel/hchoong/github/attention-nas/data/urban100_eval_matlab'

C.generator_A2B = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS/ESRGAN/RRDB_PSNR_x4.pth'

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'furnace'))

"""Image Config"""
C.image_height = 32
C.image_width = 32
#C.image_height = 56
#C.image_width = 56
#C.image_height = 120
#C.image_width = 120
#(for patched based inference)

C.num_workers = 4

C.num_cell = 5
C.op_per_cell = 5
#C.num_up = 2
C.num_up = 0
C.num_cell_path_search = 4

C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.]  # [s]

C.quantize = False
C.ENABLE_BN = False
C.ENABLE_TANH = True
C.real_measurement = False

C.nf = 64

########################################################################################################################

#C.trilevelnas_x = "A"
C.trilevelnas_x = "B"

C.attention=True
#C.attention=False

C.attn_testing = True
#C.attn_testing = False
C.attn_sink_k = 1
C.attn_levels_t = 1

C.iv_mode = 'image'

#C.primitives_attn = []
#C.primitives_attn = ['vit_spatial_patched16_image32']
#C.primitives_attn = ['separable_spatial_patched32_wrapper']
#C.primitives_attn = ['epab_spatiochannel', 'vit_spatial_patched16', 'separable_channel']
#C.primitives_attn = ['epab_spatiochannel', 'vit_spatial_patched16_image32', 'separable_channel']
#C.primitives_attn = ['epab_spatiochannel', 'separable_spatial_patched32_original', 'separable_channel']
#C.primitives_attn = ['epab_spatiochannel', 'separable_spatial_patched32', 'separable_channel']
#C.primitives_attn = ['epab_spatiochannel', 'separable_spatial_patched16_wrapper', 'separable_channel']

#C.primitives_attn = ['vit_spatial_patched4_image32']
#C.primitives_attn = ['separable_spatial_patched32_original']

C.primitives_attn = ['vit_spatial_patched4_image32','vit_spatial_map_patched4_image32',
                     'vit_channel_image32','vit_channel_map_image32',
                     'vit_spatiochannel_patched16_image32','vit_spatiochannel_map_patched16_image32']

C.arch_testing = True
#C.arch_testing = False

#C.load_path = 'arch_testing'
#C.load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_2/search/ckpt/search/search4_psnr_sparsestmax_order_1e-5'
#C.load_path = './ckpt/pretrain/exp_2021_12_26_05_47_12/checkpoint_000100.pt'
#False:'./ckpt/search/exp_2021_12_28_06_27_43'}
#False:'./ckpt/search/exp_2021_12_30_10_37_57'}
#False:'./ckpt/search/exp_2022_04_25_18_10_41'}
#ckpt/search/exp_2022_04_26_02_11_37/checkpoint_000100.pt
load_path = {
    True:'arch_testing',
    False:'./ckpt/search/exp_2022_04_26_02_11_37'}
C.load_path = load_path[C.arch_testing]
#C.load_checkpoint_file_name = 'checkpoint.pt'
load_checkpoint_file_name = {True:'arch_testing',False:'checkpoint_000100.pt'}
C.load_checkpoint_file_name = load_checkpoint_file_name[C.arch_testing]

#C.ckpt = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v4_3/search/ckpt/finetune/transfer_div2k_1e-5/weights_179.pt'
#C.ckpt = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS_attbranch/ckpt/finetune_pretrain/exp_2021_12_26_07_04_21/keep_weights_000069.pt'
#C.ckpt = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS_attbranch/ckpt/finetune_pretrain/exp_2021_12_26_07_04_21/keep_weights_000098.pt'
#C.ckpt = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS_attbranch/ckpt/finetune_pretrain/exp_2021_12_26_07_04_21/keep_weights_000182.pt'
#C.ckpt = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS_attbranch/ckpt/finetune_pretrain/exp_2021_12_26_07_04_21/keep_weights_000275.pt'
#C.ckpt = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS_attbranch/ckpt/finetune_pretrain/exp_2021_12_26_07_04_21/keep_weights_000374.pt'
#C.ckpt = '/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS_attbranch/ckpt/finetune_pretrain/exp_2021_12_26_07_04_21/keep_weights_000900.pt'
#C.ckpt = './ckpt/pretrain/exp_2021_12_26_05_47_12/weights_000100.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_03_13_56_22/weights_000394.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_03_13_56_22/weights_000900.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_09_10_24_09/keep_weights_000740.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_09_10_24_09/weights_000990.pt'

#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_20_09_45_09/weights_000900.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_20_09_45_51/weights_000770.pt'

#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_26_02_21_00/weights_001040.pt' 
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_26_02_24_38/weights_001020.pt' #['epab_spatiochannel', 'separable_spatial_patched32', 'separable_channel']
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_26_13_51_46/weights_000940.pt' #['epab_spatiochannel', 'separable_spatial_patched16', 'separable_channel']

#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_05_17_17_30/weights_000900.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_05_17_54_52/weights_001800.pt' #best sota

#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_26_13_51_46/weights_000940.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_26_02_24_38/weights_001020.pt'

#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_12_03_37_56/weights_000170.pt'

#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_30_16_35_26/weights_000044.pt' #512561

#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_30_16_21_36/weights_000036.pt' #(512555, 513231)

#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_08_12_31_29/weights_000670.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_20_09_45_09/weights_000900.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_05_17_54_52/weights_001800.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_01_30_16_39_34/weights_000049.pt' #513185
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_08_12_44_00/weights_001300.pt' #517452
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_08_12_46_00/weights_001240.pt'

#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_23_05_20_42/weights_000290.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_21_22_19_19/weights_000260.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_24_18_18_37/weights_000850.pt' #532433
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_21_20_41_13/weights_000310.pt' #530813
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_02_27_21_36_06/weights_000490.pt'
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_03_17_21_11_27/weights_000082.pt'

#A-1
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_03_18_15_27_12/weights_000029.pt'
#A-2
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_03_18_15_41_55/weights_000027.pt'

#B-1
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_03_18_15_52_19/weights_000100.pt'
#B-2
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_03_18_16_04_55/weights_000088.pt'
#B-3
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_03_18_16_06_11/weights_000096.pt'

#AGD1
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_03_19_06_41_52/weights_000001.pt'
#AGD2
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_03_19_06_59_22/weights_000001.pt'

#566624
#C.ckpt = './ckpt/finetune_pretrain/exp_2022_05_10_06_31_29/weights_000840.pt'

#570773
C.ckpt = './ckpt/finetune_pretrain/exp_2022_05_17_20_30_49/weights_000400.pt'

C.sparse_type = 'sparsestmax'