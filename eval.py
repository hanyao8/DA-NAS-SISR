from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from torchvision.utils import save_image

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_eval import config
from datasets import ImageDataset

from utils.init_func import init_weight
from utils.img_utils import patch_based_infer
import utils.profiling

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_eval import NAS_GAN_Eval

from util_gan.cyclegan import Generator
from util_gan.psnr import compute_psnr,calculate_psnr
from util_gan.lr import LambdaLR

from quantize import QConv2d, QConvTranspose2d, QuantMeasure
from thop import profile
from thop.count_hooks import count_convNd

from RRDBNet_arch import RRDBNet
#from decode import decoder, make_cell_index_matrix

import operations
import model_eval
from sparsemax import Sparsemax
from sparsestmax import sparsestmax
operations.ENABLE_BN = config.ENABLE_BN
model_eval.ENABLE_TANH = config.ENABLE_TANH
sparsemax = Sparsemax(dim=-1)


def count_custom(m, x, y):
    m.total_ops += 0

custom_ops={QConv2d: count_convNd, QConvTranspose2d:count_convNd, QuantMeasure: count_custom, nn.InstanceNorm2d: count_custom}

def path_generator(num_cell, num_up, num_cell_path_search, gamma_index):
    root_cell = [i for i in range(num_cell-num_cell_path_search)]
    root = [0 for _ in range(num_cell-num_cell_path_search)]
    cell_index = num_cell-num_cell_path_search
    paths = []
    cells = []
    feature_map_index = 0
    stack = [(root, root_cell, 0, num_cell-num_cell_path_search)]
    while stack:
        # print(stack)
        temp_path, temp_cell, level, layer = stack.pop()
        feature_map_index += 1
        paths.append(temp_path)
        cells.append(temp_cell)
        if layer < num_cell:
            stack.append((temp_path + [0], temp_cell + [cell_index], level, layer+1))
            cell_index += 1
        if level < num_up:
            stack.append((temp_path + [1], temp_cell, level+1, layer))
    return paths[gamma_index], cells[gamma_index]


def main():
    print("args = %s", str(config))
    profiling_message = utils.profiling.init_info()
    profiling_message = utils.profiling.memory_usage()

    if not(config.arch_testing):
        #state = torch.load(os.path.join(config.load_path, 'checkpoint.pt'))
        #state = torch.load(config.load_path)
        state = torch.load(os.path.join(config.load_path,config.load_checkpoint_file_name))
        #logging.info('loading from:%s', str(os.path.join(config.load_path, 'arch.pt')))
        logging.info('loading from:%s', config.load_path)
        if config.sparse_type == 'sparsemax':
            op_idx_all_list = sparsemax(state['alpha']).argmax(-1)
            quantize_all_list = sparsemax(state['beta']).argmax(-1) == 1
            ratio_all_list = sparsemax(state['ratio']).argmax(-1)
            path_index = torch.argmax(sparsemax(state['gamma']))
            logging.info('gamma=%s', str(sparsemax(state['gamma'])))
        elif config.sparse_type == 'sparsestmax':
            alpha_sparse = sparsestmax(state['alpha'],1)
            op_idx_all_list = alpha_sparse.argmax(-1)

            beta_sparse = sparsestmax(state['beta'],1)
            quantize_all_list = beta_sparse.argmax(-1) == 1

            ratio_sparse = sparsestmax(state['ratio'],1)
            ratio_all_list = ratio_sparse.argmax(-1)

            gamma_sparse = sparsestmax(state['gamma'],1)
            path_index = torch.argmax(gamma_sparse)

            logging.info('gamma_sparse=%s'%str(gamma_sparse))
        else:
            raise NotImplementedError('ivalid sparse type')
        # path_index = 9
        path, cell_index = path_generator(config.num_cell, config.num_up, config.num_cell_path_search, path_index)
        num_up = sum(path)
        num_cell = len(path) - num_up
        # logging.info('gamma=%s', str(path_list))
        op_idx_list = op_idx_all_list[cell_index]
        quantize_list = quantize_all_list[cell_index]
        ratio_list = ratio_all_list[cell_index]
    else:
        if True:

            #570773
            path = [0, 0, 0]
            cell_index = [0, 1, 16]
            num_up = 0
            op_per_cell = 5
            num_cell = 3
            num_cell_path_search = num_cell - 1

            op_idx_list = [[3, 3, 2, 2, 2],
                           [2, 0, 0, 0, 0],
                           [2, 2, 2, 2, 0]]
            quantize_list = [[False,  True,  True, False,  True],
                             [ True,  True, False, False,  True],
                             [False, False, False, False, False]]
            ratio_list = [[1, 4, 4, 4],
                          [0, 0, 4, 1],
                          [1, 4, 0, 0]]


            ##latency test trilevelnas-A no. 1
            #path=[0, 0, 0, 1, 0, 1]
            #cell_index = []
            #num_up = sum(path)
            #num_cell = len(path) - num_up
            #op_idx_list =  [[0, 1, 0, 0, 1],
            #                [3, 3, 3, 1, 2],
            #               [0, 0, 1, 0, 3],
            #                [1, 1, 0, 1, 1]]
            #quantize_list =    [[False, False, False, False, False],
            #                   [False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False]]
            #ratio_list =   [[4, 0, 0, 0],
            #                [1, 1, 4, 0],
            #                [4, 4, 2, 1],
            #                [0, 4, 1, 1]]

            ##latency test trilevelnas-A no. 2
            #path=[0, 0, 0, 0, 0, 1, 1]
            #cell_index = []
            #num_up = sum(path)
            #num_cell = len(path) - num_up
            #op_idx_list =  [[3, 3, 3, 3, 3],
            #                [0, 1, 3, 1, 3],
            #                [3, 3, 3, 2, 3],
            #                [3, 3, 3, 2, 2],
            #                [1, 3, 3, 3, 2]]
            #quantize_list =    [[False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False]]
            #ratio_list =   [[3, 0, 4, 2],
            #                [4, 4, 4, 0],
            #                [1, 1, 2, 2],
            #                [2, 2, 4, 2],
            #                [2, 0, 0, 0]]

            ##latency test trilevelnas-B no. 1
            #path=[0, 0, 0]
            #cell_index = []
            #num_up = sum(path)
            #num_cell = len(path) - num_up
            #op_idx_list =  [[3, 0, 3, 3, 3],
            #                [3, 0, 1, 3, 1],
            #                [2, 0, 0, 0, 0]]
            #quantize_list =    [[False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False]]
            #ratio_list =   [[1, 1, 0, 2],
            #                [0, 3, 0, 0],
            #                [4, 0, 0, 0]]

            #latency test trilevelnas-B no. 2
            #path=[0, 0, 0, 0, 0, 1]
            #cell_index = []
            #num_up = sum(path)
            #num_cell = len(path) - num_up
            #op_idx_list =  [[0, 0, 3, 3, 0],
            #                [3, 2, 3, 1, 1],
            #                [1, 0, 1, 0, 3],
            #                [3, 3, 1, 1, 3],
            #                [3, 1, 1, 1, 1]]
            #quantize_list =    [[False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False]]
            #ratio_list =   [[1, 0, 4, 0],
            #                [1, 2, 3, 2],
            #                [4, 4, 0, 1],
            #                [4, 3, 0, 0],
            #                [1, 4, 4, 0]]

            ##latency test trilevelnas-B no. 3
            #path=[0, 0, 1, 1]
            #cell_index = []
            #num_up = sum(path)
            #num_cell = len(path) - num_up
            #op_idx_list =  [[0, 0, 0, 2, 0],
            #                [3, 3, 1, 1, 0]]
            #quantize_list =    [[False, False, False, False, False],
            #                    [False, False, False, False, False]]
            #ratio_list =   [[2, 0, 3, 3],
            #                [4, 3, 1, 0]]

            ##latency test agd no. 1 visualization
            #print('agd1')
            #path=[0, 0, 0, 0, 0, 1, 1]
            #cell_index = []
            #num_up = sum(path)
            #num_cell = len(path) - num_up
            #op_idx_list =  [[0, 0, 0, 0, 0],
            #                [3, 3, 2, 1, 2],
            #                [0, 0, 0, 0, 0],
            #                [1, 3, 3, 3, 2],
            #                [0, 0, 0, 0, 0]]
            #quantize_list =    [[False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False]]
            #ratio_list =   [[0, 0, 0, 0],
            #                [0, 0, 4, 0],
            #                [0, 0, 0, 0],
            #                [1, 1, 0, 0],
            #                [0, 0, 0, 0]]

            ##latency test agd no. 2 psnr
            #path=[0, 0, 0, 0, 0, 1, 1]
            #cell_index = []
            #num_up = sum(path)
            #num_cell = len(path) - num_up
            #op_idx_list =  [[3, 3, 3, 3, 2],
            #                [3, 2, 0, 0, 1],
            #                [0, 0, 0, 0, 0],
            #                [3, 3, 3, 3, 2],
            #                [0, 0, 0, 0, 0]]
            #quantize_list =    [[False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False]]
            #ratio_list =   [[1, 1, 3, 0],
            #                [0, 1, 1, 0],
            #                [0, 0, 0, 0],
            #                [1, 3, 0, 0],
            #                [0, 0, 0, 0]]


###################################################################################################

            #path=[0, 0, 0, 1, 0, 1]
            #cell_index = []
            #num_up = sum(path)
            #num_cell = len(path) - num_up
            #op_idx_list =  [[0, 1, 0, 0, 1],
            #                [3, 3, 3, 1, 2],
            #                [0, 0, 1, 0, 3],
            #                [1, 1, 0, 1, 1]]
            #quantize_list =    [[False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False],
            #                    [False, False, False, False, False]]
            #ratio_list =   [[4, 0, 0, 0],
            #                [1, 1, 4, 0],
            #                [4, 4, 2, 1],
            #                [0, 4, 1, 1]]


            #path=[0, 0, 0, 1]
            #cell_index=[0, 1, 16]
            #num_up = sum(path)
            #num_cell = len(path) - num_up
            #op_idx_list =  [[3, 3, 2, 2, 2],
            #                [2, 0, 0, 0, 0],
            #                [2, 2, 2, 2, 0]]
            #quantize_list =    [[False,  True,  True, False,  True],
            #                    [ True,  True, False, False,  True],
            #                    [False, False, False, False, False]]
            #ratio_list =   [[1, 4, 4, 4],
            #                [0, 0, 4, 1],
            #                [1, 4, 0, 0]]
        else:
            path=[0, 0, 0, 0, 1]
            cell_index=[0, 1, 16, 26]
            num_up = sum(path)
            num_cell = len(path) - num_up
            op_idx_list =  [[3, 3, 2, 2, 2],
                            [2, 0, 0, 0, 0],
                            [2, 2, 2, 2, 0],
                            [2, 2, 2, 3, 3]]
            quantize_list =    [[False,  True,  True, False,  True],
                                [ True,  True, False, False,  True],
                                [False, False, False, False, False],
                                [False,  True,  True,  True, False]]
            ratio_list =   [[1, 4, 4, 4],
                            [0, 0, 4, 1],
                            [1, 4, 0, 0],
                            [4, 4, 4, 2]]

        #path=[0, 0, 0, 0, 0, 1]
        #cell_index=[0, 1, 16, 26, 32]
        #num_up = sum(path)
        #num_cell = len(path) - num_up
        #op_idx_list = [
        #    [1, 3, 2, 2, 1],
        #    [1, 3, 1, 2, 3],
        #    [1, 3, 3, 3, 3],
        #    [0, 0, 2, 1, 2],
        #    [1, 3, 2, 1, 0]]
        #quantize_list = [
        #    [False,  True,  True, False,  True],
        #    [ True,  True, False, False,  True],
        #    [False, False, False, False, False],
        #    [False,  True,  True,  True, False],
        #    [ True, False, False,  True, False]]
        #ratio_list = [
        #    [2, 4, 4, 3],
        #    [4, 1, 0, 1],
        #    [2, 4, 3, 0],
        #    [4, 0, 3, 4],
        #    [1, 0, 0, 1]]

        #op_idx_list = torch.Tensor(op_idx_list)

    #num_attn_cells=1 num_attn_op_per_cell=1 num_attn_levels=2 num_attn_op=3

    if config.attn_testing and config.attention:
        if len(config.primitives_attn)==1:
            alpha_sink_attn1 = torch.tensor([[ [0.0000, 0.7000] ]])
            alpha_sink_attn = alpha_sink_attn1
            alpha_levels_attn1 = torch.tensor([[[ [     [-0.7175] ] ]]])
            alpha_levels_attn = alpha_levels_attn1
        elif len(config.primitives_attn)==2:
            alpha_sink_attn2 = torch.tensor([[ [0.0000, 0.0000, 0.9000, 0.7000] ]])
            alpha_sink_attn = alpha_sink_attn2
            alpha_levels_attn2 = torch.tensor([[[ [     [-2.0440, -0.4560],
                                                        [-0.7175,  1.3922] ] ]]])
            alpha_levels_attn = alpha_levels_attn2
        elif len(config.primitives_attn)==3:
            alpha_sink_attn3 = torch.tensor([[ [0.0000, 0.0000, 0.0000, 0.6000, 0.9000, 0.7000] ]])
            alpha_sink_attn = alpha_sink_attn3
            alpha_levels_attn3 = torch.tensor([[[ [     [1.4271, -1.8701, -1.1962],
                                                        [-2.0440, -0.4560, -1.4295],
                                                        [-0.7175,  1.3922,  0.0811] ] ]]])
            alpha_levels_attn = alpha_levels_attn3
        elif len(config.primitives_attn)==4:
            alpha_sink_attn4 = torch.tensor([[ [0.0000, 0.0000, 0.0000, 0.0000,
                                                0.6000, 0.9000, 0.7000, 0.8000] ]])
            alpha_sink_attn = alpha_sink_attn4
            alpha_levels_attn4 = torch.tensor([[[ [     [1.4271, 0.38344, -1.8701, -1.1962],
                                                        [1.2834, -2.0440, -0.4560, -1.4295],
                                                        [2.0124, 0.31947, -1.8342, -0.3921],
                                                        [-0.1984, -0.7175,  1.3922,  0.0811] ] ]]])
            alpha_levels_attn = alpha_levels_attn4

        #570773
        alpha_sink_attn = [[[0.0804, 0.0839, 0.0782, 0.0762, 0.0794, 0.0767,
                             0.0851, 0.0751, 0.1252, 0.0766, 0.0868, 0.0764]]]

        alpha_levels_attn = [[[[[0.1340, 0.1342, 0.2997, 0.1342, 0.1636, 0.1342],
                                [0.1346, 0.1346, 0.1267, 0.1346, 0.3103, 0.1592],
                                [0.1278, 0.1304, 0.1212, 0.1321, 0.3593, 0.1291],
                                [0.2627, 0.1364, 0.1363, 0.1926, 0.1358, 0.1364],
                                [0.1846, 0.1994, 0.1397, 0.1627, 0.1399, 0.1738],
                                [0.1351, 0.1354, 0.2951, 0.1694, 0.1330, 0.1321]]]]]

        alpha_sink_attn = torch.tensor(alpha_sink_attn)
        alpha_levels_attn = torch.tensor(alpha_levels_attn)

        print("alpha_sink_attn: %s"%str(alpha_sink_attn))
        print(alpha_sink_attn.shape)
        print("alpha_attn_levels: %s"%str(alpha_levels_attn))
        print(alpha_levels_attn.shape)
        #attn_sink_k = alpha_levels_attn.shape[-1]
        #attn_levels_t = alpha_levels_attn.shape[-1]
        attn_sink_k = config.attn_sink_k
        attn_levels_t = config.attn_levels_t
    elif not(config.attn_testing) and config.attention:
        alpha_sink_attn = state['alpha_sink_attn']
        alpha_levels_attn = state['alpha_levels_attn']
        attn_sink_k = config.attn_sink_k
        attn_levels_t = config.attn_levels_t
    else:
        alpha_sink_attn = None
        alpha_levels_attn = None
        attn_sink_k = None
        attn_levels_t = None

    print('path=%s', path)
    print('cell index=%s', str(cell_index))
    print("op = %s"%str(op_idx_list))
    print("quantize = %s"%str(quantize_list))
    print('ratio=%s'%str(ratio_list))

    logging.info('main | path=%s'%(str(path)))
    logging.info('main | cell_index=%s'%(str(cell_index)))
    logging.info('main | num_up=%s'%(str(num_up)))
    logging.info('main | num_cell=%s'%(str(num_cell)))
    logging.info('main | op_idx_list=%s'%(str(op_idx_list)))
    logging.info('main | quantize_list=%s'%(str(quantize_list)))
    logging.info('main | ratio_list=%s'%(str(ratio_list)))

    # Model #######################################

    #model = NAS_GAN_Eval(op_idx_list, quantize_list, ratio_list, path, num_cell=num_cell, op_per_cell=config.op_per_cell, num_up=config.num_up, num_cell_path_search=config.num_cell_path_search,
    #                     width_mult_list=config.width_mult_list, quantize=config.quantize)
    model = NAS_GAN_Eval(op_idx_list, quantize_list, ratio_list, path, num_cell=num_cell, op_per_cell=config.op_per_cell, num_up=config.num_up, num_cell_path_search=config.num_cell_path_search,
                         width_mult_list=config.width_mult_list, quantize=config.quantize,
                         iv_mode=config.iv_mode,attention=config.attention,alpha_sink_attn=alpha_sink_attn,alpha_levels_attn=alpha_levels_attn,
                         sink_k=attn_sink_k,levels_t=attn_levels_t,primitives_attn=config.primitives_attn,trilevelnas_x=config.trilevelnas_x)

    total_num_params = 0
    non_vgg_params = 0
    backbone_params = 0
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.size())).item()
        print(name, param.size(),'num_params:',num_params)
        total_num_params += num_params
        if not('vgg' in name):
            non_vgg_params += num_params
        if not(('vgg' in name) or ('_ops_attn' in name)):
            backbone_params += num_params
    print(param)
    print('total_num_params:',total_num_params)
    print('non_vgg_params:',non_vgg_params)
    print('backbone_params:',backbone_params)
    print('\n')

    flops, params = profile(model, inputs=(torch.randn(1, 3, 256, 256),), custom_ops=custom_ops)
    logging.info("params = %fM, FLOPs = %fGFlops"%(params / 1e6, flops / 1e9))
    print("params = %fM, FLOPs = %fGFlops"%(params / 1e6, flops / 1e9))

    #flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), custom_ops=custom_ops)
    #flops, params = profile(model, inputs=(torch.randn(1, 3, 128, 128),), custom_ops=custom_ops)
    #flops, params = profile(model, inputs=(torch.randn(1, 3, 256, 256),), custom_ops=custom_ops)

    #flops = model.forward_flops(size=(3, 510, 350))
    #print("params = %fMB, FLOPs = %fGFlops" % (params / 1e6, flops / 1e9))

    #latency = model.forward_latency(size=(3,32,32))

    latency = model.forward_latency(size=(3,256,256))
    print("latency=",latency)

    print(config.output_dir)

    model = torch.nn.DataParallel(model).cuda()    
    if config.ckpt:
        state_dict = torch.load(config.ckpt)
        model.load_state_dict(state_dict, strict=False)


    transforms_ = [ transforms.ToTensor()]
    #sets = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/data/DIV2K/DIV2K_valid'
    # for sets in config.dataset_path:
    #test_loader = DataLoader(ImageDataset(sets, transforms_=transforms_, mode='val'), 
    #                        batch_size=1, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='val'), 
                        batch_size=1, shuffle=False, num_workers=config.num_workers)
    print('line 251 before calling infer()')
    if True:
        with torch.no_grad():
            #valid_psnr_rgb, valid_psnr_y = infer(model, test_loader, sets)
            valid_psnr_rgb, valid_psnr_y = infer(model, test_loader, config.dataset_path)
        #print(sets)
        print(config.dataset_path)
        print(config.output_dir)
        print('PSNR_RGB:', valid_psnr_rgb)
        print('PSNR_Y:', valid_psnr_y)

    if False:
        from RRDBNet_arch import RRDBNet
        teacher_model = RRDBNet(3, 3, 64, 23, gc=32)
        teacher_model.load_state_dict(torch.load(config.generator_A2B), strict=True)

        total_num_params = 0
        for name, param in teacher_model.named_parameters():
            num_params = torch.prod(torch.tensor(param.size())).item()
            print(name, param.size(),'num_params:',num_params)
            total_num_params += num_params
        print(param)
        print('teacher_model total_num_params:',total_num_params)
        print('\n')

        flops, params = profile(teacher_model, inputs=(torch.randn(1, 3, 256, 256),), custom_ops=custom_ops)
        logging.info("teacher_model params = %fM, FLOPs = %fGFlops"%(params / 1e6, flops / 1e9))
        print("teacher_model params = %fM, FLOPs = %fGFlops"%(params / 1e6, flops / 1e9))

        teacher_model = torch.nn.DataParallel(teacher_model).cuda()
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.require_grads = False

        with torch.no_grad():
            #valid_psnr_rgb, valid_psnr_y = infer(model, test_loader, sets)
            valid_psnr_rgb, valid_psnr_y = infer(teacher_model, test_loader, config.dataset_path)
        #print(sets)
        print(config.dataset_path)
        print(config.output_dir)
        print('PSNR_RGB:', valid_psnr_rgb)
        print('PSNR_Y:', valid_psnr_y)


def infer(model, test_loader, dataset):
    model.eval()

    for i, batch in enumerate(test_loader):
        #if i==1:
        #    raise(Exception)
        print("eval.py, infer %d"%i)
        img_name = "%06d"%i + "." + "png"

        real_A = batch['A'].cuda()
        #if i==0:
        #    real_A = real_A[:,:,32:32+32,80:80+32]
        print(type(real_A))
        print(real_A.shape)

        if config.attention:
            fake_B = model(real_A).data.float().clamp_(0, 1)
            os.makedirs(os.path.join(*[config.output_dir,'direct']), exist_ok=True)
            save_image(fake_B, os.path.join(*[config.output_dir,'direct',img_name]))

            fake_B_patched = patch_based_infer(real_A,model,
                    patch_shape=(config.image_height,config.image_width)).data.float().clamp_(0, 1)
            os.makedirs(os.path.join(*[config.output_dir,'patched']), exist_ok=True)
            save_image(fake_B_patched, os.path.join(*[config.output_dir,'patched',"patched_"+img_name]))
        else:
            fake_B = model(real_A).data.float().clamp_(0, 1)
            os.makedirs(os.path.join(*[config.output_dir,'direct']), exist_ok=True)
            save_image(fake_B, os.path.join(*[config.output_dir,'direct',img_name]))

            fake_B_patched = patch_based_infer(real_A,model,
                    patch_shape=(config.image_height,config.image_width)).data.float().clamp_(0, 1)
            os.makedirs(os.path.join(*[config.output_dir,'patched']), exist_ok=True)
            save_image(fake_B_patched, os.path.join(*[config.output_dir,'patched',"patched_"+img_name]))
        print("\n")

    #if not config.real_measurement:
    #    # psnr = compute_psnr('output/eval', config.dataset_path)
    #    #psnr_rgb = compute_psnr('output/eval', dataset + '/val_hr', test_Y=False, crop_border=0)
    #    #psnr_y = compute_psnr('output/eval', dataset + '/val_hr', test_Y=True, crop_border=0)
    #    psnr_rgb = compute_psnr(config.output_dir, dataset + '/val_hr', test_Y=False, crop_border=0)
    #    psnr_y = compute_psnr(config.output_dir, dataset + '/val_hr', test_Y=True, crop_border=0)
    #else:
    #    psnr_rgb = 0
    #    psnr_y = 0

    if config.attention:
        psnr_rgb_direct = compute_psnr(config.output_dir+'/direct', dataset + '/val_hr', test_Y=False, crop_border=0)
        psnr_y_direct = compute_psnr(config.output_dir+'/direct', dataset + '/val_hr', test_Y=True, crop_border=0)
        psnr_rgb_patched = compute_psnr(config.output_dir+'/patched', dataset + '/val_hr', test_Y=False, crop_border=0)
        psnr_y_patched = compute_psnr(config.output_dir+'/patched', dataset + '/val_hr', test_Y=True, crop_border=0)
        print('PSNR_RBG (direct):', psnr_rgb_direct)
        print('PSNR_Y (direct):', psnr_y_direct)
        print('PSNR_RGB (patched):', psnr_rgb_patched)
        print('PSNR_Y (patched):', psnr_y_patched)
        print('patch_based_infer patch size: (%d,%d)'%(config.image_height,config.image_width))
        return psnr_rgb_direct, psnr_y_direct
    else:
        psnr_rgb_direct = compute_psnr(config.output_dir+'/direct', dataset + '/val_hr', test_Y=False, crop_border=0)
        psnr_y_direct = compute_psnr(config.output_dir+'/direct', dataset + '/val_hr', test_Y=True, crop_border=0)
        psnr_rgb_patched = compute_psnr(config.output_dir+'/patched', dataset + '/val_hr', test_Y=False, crop_border=0)
        psnr_y_patched = compute_psnr(config.output_dir+'/patched', dataset + '/val_hr', test_Y=True, crop_border=0)
        print('PSNR_RBG (direct):', psnr_rgb_direct)
        print('PSNR_Y (direct):', psnr_y_direct)
        print('PSNR_RGB (patched):', psnr_rgb_patched)
        print('PSNR_Y (patched):', psnr_y_patched)
        print('patch_based_infer patch size: (%d,%d)'%(config.image_height,config.image_width))
        return psnr_rgb_direct, psnr_y_direct

if __name__ == '__main__':
    main() 
