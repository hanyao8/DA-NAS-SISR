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

from config_train import config
from datasets import ImageDataset,ImageDatasetGT

from utils.init_func import init_weight
from utils.img_utils import patch_based_infer
import utils.profiling

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat, save_checkpoint
from model_infer import NAS_GAN_Infer

from util_gan.cyclegan import Generator
from util_gan.psnr import compute_psnr,calculate_psnr
from util_gan.lr import LambdaLR

from quantize import QConv2d, QConvTranspose2d, QuantMeasure
from thop import profile
from thop.count_hooks import count_convNd

from RRDBNet_arch import RRDBNet
import operations
import model_infer
from sparsemax import Sparsemax
from sparsestmax import sparsestmax
operations.ENABLE_BN = config.ENABLE_BN
model_infer.ENABLE_TANH = config.ENABLE_TANH
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
    profiling_message = utils.profiling.init_info()
    profiling_message = utils.profiling.memory_usage()

    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
#################################################################################################### Main
    if not(config.arch_testing):
        #state = torch.load(os.path.join(config.load_path, 'checkpoint.pt'))
        #state = torch.load(config.load_path)
        state = torch.load(os.path.join(config.load_path, config.load_checkpoint_file_name))

        if config.sparse_type == 'sparsemax':
            op_idx_all_list = sparsemax(state['alpha']).argmax(-1)
            quantize_all_list = sparsemax(state['beta']).argmax(-1) == 1
            ratio_all_list = sparsemax(state['ratio']).argmax(-1)
            path_index = torch.argmax(sparsemax(state['gamma']))
            # logging.info('gamma=%s', str(sparsemax(state['gamma'])))
        elif config.sparse_type == 'sparsestmax':
            op_idx_all_list = sparsestmax(state['alpha'], 1).argmax(-1)
            quantize_all_list = sparsestmax(state['beta'], 1).argmax(-1) == 1
            ratio_all_list = sparsestmax(state['ratio'], 1).argmax(-1)
            path_index = torch.argmax(sparsestmax(state['gamma'], 1))
            # logging.info('gamma=%s', str(sparsestmax(state['gamma'], 1)))
            print('line 108 shapes')
            print(op_idx_all_list.shape)
            print(state['alpha'].shape)
            print(state['beta'].shape)
            print(state['ratio'].shape)
            print(path_index.shape)
            print(path_index)
            print(state['gamma'].shape)
        elif config.sparse_type == 'softmax':
            op_idx_all_list = F.softmax(state['alpha'], dim=-1).argmax(-1)
            quantize_all_list = F.softmax(state['beta'], dim=-1).argmax(-1) == 1
            ratio_all_list = F.softmax(state['ratio'], dim=-1).argmax(-1)
            path_index = torch.argmax(F.softmax(state['gamma'], dim=-1))
            # logging.info('gamma=%s', str(F.softmax(state['gamma'])))
        else:
            raise NotImplementedError('ivalid sparse type')
        
        path, cell_index = path_generator(config.num_cell, config.num_up, config.num_cell_path_search, path_index)
        num_up = sum(path)
        num_cell = len(path) - num_up
        op_per_cell = config.op_per_cell
        num_cell_path_search = config.num_cell_path_search


        # logging.info('gamma=%s', str(path_list))
        op_idx_list = op_idx_all_list[cell_index]
        quantize_list = quantize_all_list[cell_index]
        ratio_list = ratio_all_list[cell_index]
    else:

        path = [0, 0]
        cell_index = []
        num_up = 0
        op_per_cell = 5
        num_cell = 2
        num_cell_path_search = num_cell - 1

        op_idx_list = [[2, 0, 0, 0, 0],
                       [2, 2, 2, 2, 0]]
        quantize_list = [[ True,  True, False, False,  True],
                         [False, False, False, False, False]]
        ratio_list = [[0, 0, 4, 1],
                      [1, 4, 0, 0]]

#        path = [0, 0, 0]
#        cell_index = [0, 1, 16]
#        num_up = 0
#        op_per_cell = 5
#        num_cell = 3
#        num_cell_path_search = num_cell - 1

#        op_idx_list = [[3, 3, 2, 2, 2],
#                       [2, 0, 0, 0, 0],
#                       [2, 2, 2, 2, 0]]
#        quantize_list = [[False,  True,  True, False,  True],
#                         [ True,  True, False, False,  True],
#                         [False, False, False, False, False]]
#        ratio_list = [[1, 4, 4, 4],
#                     [0, 0, 4, 1],
#                     [1, 4, 0, 0]]

        #latency test agd no. 2 psnr
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

        ##latency test agd no. 1 visualization
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

        ##latency test trilevelnas-A no. 1
        #path=[0, 0, 0, 1, 0, 1]
        #cell_index = []
        #num_up = sum(path)
        #num_cell = len(path) - num_up
        #op_idx_list =  [[0, 1, 0, 0, 1],
        #                [3, 3, 3, 1, 2],
        #                [0, 0, 1, 0, 3],
        #                [1, 1, 0, 1, 1]]
        #quantize_list =    [[False, False, False, False, False],
        #                   [False, False, False, False, False],
        #                    [False, False, False, False, False],
        #                    [False, False, False, False, False]]
        #ratio_list =   [[4, 0, 0, 0],
        #                [1, 1, 4, 0],
        #                [4, 4, 2, 1],
        #                [0, 4, 1, 1]]

        #path=[0, 0, 0, 0, 1]
        #    #cell_index=[0, 1, 16, 26]
        #num_up = sum(path)
        #num_cell = len(path) - num_up
        #op_idx_list =  [[3, 3, 2, 2, 2],
        #                [2, 0, 0, 0, 0],
        #                [2, 2, 2, 2, 0],
        #                [2, 2, 2, 3, 3]]
        #quantize_list =    [[False,  True,  True, False,  True],
        #                    [ True,  True, False, False,  True],
        #                    [False, False, False, False, False],
        #                    [False,  True,  True,  True, False]]
        #ratio_list =   [[1, 4, 4, 4],
        #                [0, 0, 4, 1],
        #                [1, 4, 0, 0],
        #                [4, 4, 4, 2]]

        #path=[0, 0, 0, 0, 1]
        #    #cell_index=[0, 1, 16, 26]
        #num_up = sum(path)
        #num_cell = len(path) - num_up
        #op_idx_list =  [[3, 3, 2, 2, 2],
        #                [2, 0, 0, 0, 0],
        #                [2, 2, 2, 2, 0],
        #                [2, 2, 2, 3, 3]]
        #quantize_list =    [[False,  True,  True, False,  True],
        #                    [ True,  True, False, False,  True],
        #                    [False, False, False, False, False],
        #                    [False,  True,  True,  True, False]]
        #ratio_list =   [[1, 4, 4, 4],
        #                [0, 0, 4, 1],
        #                [1, 4, 0, 0],
        #                [4, 4, 4, 2]]

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

        op_idx_list = torch.tensor(op_idx_list)
        quantize_list = torch.tensor(quantize_list)
        ratio_list = torch.tensor(ratio_list)

    print("initialize attention arch params")
    if config.attn_testing and config.attention:
        #num_attn_cells=1
        #num_attn_op_per_cell=1
        #num_attn_levels=2
        #num_attn_op=3

        #alpha_attn_sink = torch.zeros((num_attn_cells,num_attn_op_per_cell,
        #    num_attn_levels*num_attn_op))
        #last dimension is num_attn_levels*num_attn_op because 
        # normalization is applied on that dimension (i.e. sparsestmax or softmax)

        #alpha_attn_levels = torch.randn((num_attn_cells,num_attn_op_per_cell,
        #    num_attn_levels-1,num_attn_op,num_attn_op))

        #fix parameter tensors for reproducibility and fair comparison
        #all settings result to fully connected attention cell

        alpha_sink_attn = [[[0.1252, 0.0851, 0.0751, 0.0766, 0.0868, 0.0764,
                             0.0804, 0.0839, 0.0782, 0.0762, 0.0794, 0.0767]]]

        #alpha_sink_attn = [[[0.0804, 0.0839, 0.0782, 0.0762, 0.0794, 0.0767,
        #                     0.0851, 0.0751, 0.1252, 0.0766, 0.0868, 0.0764]]]

        alpha_levels_attn = [[[[[0.1340, 0.1342, 0.2997, 0.1342, 0.1636, 0.1342],
                                [0.1346, 0.1346, 0.1267, 0.1346, 0.3103, 0.1592],
                                [0.1278, 0.1304, 0.1212, 0.1321, 0.3593, 0.1291],
                                [0.2627, 0.1364, 0.1363, 0.1926, 0.1358, 0.1364],
                                [0.1846, 0.1994, 0.1397, 0.1627, 0.1399, 0.1738],
                                [0.1351, 0.1354, 0.2951, 0.1694, 0.1330, 0.1321]]]]]

#        if len(config.primitives_attn)==1:
#            alpha_sink_attn1 = torch.tensor([[ [0.0000, 0.7000] ]])
#            alpha_sink_attn = alpha_sink_attn1
#            alpha_levels_attn1 = torch.tensor([[[ [     [-0.7175] ] ]]])
#            alpha_levels_attn = alpha_levels_attn1
#        elif len(config.primitives_attn)==2:
#            alpha_sink_attn2 = torch.tensor([[ [0.0000, 0.0000, 0.9000, 0.7000] ]])
#            alpha_sink_attn = alpha_sink_attn2
#            alpha_levels_attn2 = torch.tensor([[[ [     [-2.0440, -0.4560],
#                                                        [-0.7175,  1.3922] ] ]]])
#            alpha_levels_attn = alpha_levels_attn2
#        elif len(config.primitives_attn)==3:
#            alpha_sink_attn3 = torch.tensor([[ [0.0000, 0.0000, 0.0000, 0.6000, 0.9000, 0.7000] ]])
#            alpha_sink_attn = alpha_sink_attn3
#            alpha_levels_attn3 = torch.tensor([[[ [     [1.4271, -1.8701, -1.1962],
#                                                        [-2.0440, -0.4560, -1.4295],
#                                                        [-0.7175,  1.3922,  0.0811] ] ]]])
#            alpha_levels_attn = alpha_levels_attn3
#        elif len(config.primitives_attn)==4:
#            alpha_sink_attn4 = torch.tensor([[ [0.0000, 0.0000, 0.0000, 0.0000,
#                                                0.6000, 0.9000, 0.7000, 0.8000] ]])
#            alpha_sink_attn = alpha_sink_attn4
#            alpha_levels_attn4 = torch.tensor([[[ [     [1.4271, 0.38344, -1.8701, -1.1962],
#                                                        [1.2834, -2.0440, -0.4560, -1.4295],
#                                                        [2.0124, 0.31947, -1.8342, -0.3921],
#                                                        [-0.1984, -0.7175,  1.3922,  0.0811] ] ]]])
#            alpha_levels_attn = alpha_levels_attn4

        alpha_sink_attn = torch.tensor(alpha_sink_attn)
        alpha_levels_attn = torch.tensor(alpha_levels_attn)

        print("alpha_sink_attn: %s"%str(alpha_sink_attn))
        print(alpha_sink_attn.shape)
        print("alpha_attn_levels: %s"%str(alpha_levels_attn))
        print(alpha_levels_attn.shape)

        #attn_sink_k = alpha_levels_attn.shape[-1]
        #attn_sink_k = alpha_sink_attn.shape[-1]
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


    logging.info('main | path=%s'%(str(path)))
    logging.info('main | cell_index=%s'%(str(cell_index)))
    logging.info('main | num_up=%s'%(str(num_up)))
    logging.info('main | num_cell=%s'%(str(num_cell)))
    logging.info('main | op_idx_list=%s'%(str(op_idx_list)))
    logging.info('main | quantize_list=%s'%(str(quantize_list)))
    logging.info('main | ratio_list=%s'%(str(ratio_list)))

    print('main | path=%s'%(str(path)))
    print('main | cell_index=%s'%(str(cell_index)))
    print('main | num_up=%s'%(str(num_up)))
    print('main | num_cell=%s'%(str(num_cell)))
    print('main | op_idx_list=%s'%(str(op_idx_list)))
    print('main | quantize_list=%s'%(str(quantize_list)))
    print('main | ratio_list=%s'%(str(ratio_list)))

    # Model #######################################
    print("instantiate NAS_GAN_Infer")
    model = NAS_GAN_Infer(op_idx_list, quantize_list, ratio_list, path,
                          num_cell=num_cell, op_per_cell=op_per_cell, num_up=num_up, num_cell_path_search=num_cell_path_search,
                          width_mult_list=config.width_mult_list, loss_weight=config.loss_weight, loss_func=config.loss_func,
                          before_act=config.before_act, quantize=config.quantize, nf=config.nf,
                          iv_mode=config.iv_mode,attention=config.attention,alpha_sink_attn=alpha_sink_attn,alpha_levels_attn=alpha_levels_attn,
                          sink_k=attn_sink_k,levels_t=attn_levels_t,primitives_attn=config.primitives_attn,trilevelnas_x=config.trilevelnas_x)

                          #For attention extension alphas are directly passed into model_infer
                          #because attention cell derivation depends on alpha values at various levels in the attn cell

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

    #if not(config.attention):
    
    #flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), custom_ops=custom_ops)
    #flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),), custom_ops=custom_ops)

    #logging.info("FLOPs = %fGFlops (thop profile 3,256,256)", flops / 1e9)
    #print("FLOPs = %fGFlops (thop profile 3,256,256)"%(flops / 1e9))
    #logging.info("params = %fMB, FLOPs = %fGFlops", params / 1e6, flops / 1e9)
    #print("params = %fMB, FLOPs = %fGFlops"%(params / 1e6, flops / 1e9))

    #flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), custom_ops=custom_ops)
    #logging.info("FLOPs = %fGFlops (thop profile 3,32,32)", flops / 1e9)
    #print("FLOPs = %fGFlops (thop profile 3,32,32)"%(flops / 1e9))

    #flops = model.forward_flops(size=(3, 256, 256))
    #flops = model.forward_flops(size=(3, 128, 128))
    #logging.info("FLOPs = %fGFlops (model_infer)", flops / 1e9)
    #print("FLOPs = %fGFlops (model_infer)", flops / 1e9)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 256, 256),), custom_ops=custom_ops)
    logging.info("params = %fM, FLOPs = %fGFlops"%(params / 1e6, flops / 1e9))
    print("params = %fM, FLOPs = %fGFlops"%(params / 1e6, flops / 1e9))

    model = torch.nn.DataParallel(model).cuda()

    if type(config.pretrain) == str:
        state_dict = torch.load(config.pretrain)
        model.load_state_dict(state_dict)

    if config.hard_gt:
        teacher_model = None
    else:
        teacher_model = RRDBNet(3, 3, 64, 23, gc=32)
        teacher_model.load_state_dict(torch.load(config.generator_A2B), strict=True)
        teacher_model = torch.nn.DataParallel(teacher_model).cuda()
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.require_grads = False

    # Optimizer ###################################
    start_epoch = 0
    base_lr = config.lr
    parameters = []
    parameters += list(model.module.cells.parameters())
    parameters += list(model.module.conv_first.parameters())
    parameters += list(model.module.conv_last.parameters())

    ################################################################################
    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=base_lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            parameters,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        print("Wrong Optimizer Type.")
        sys.exit()

    # lr policy ##############################
    #total_iteration = config.nepochs * config.niters_per_epoch

    if config.lr_schedule == 'linear':
        lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(config.nepochs, 0, config.decay_epoch).step)
    elif config.lr_schedule == 'exponential':
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    elif config.lr_schedule == 'multistep':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    else:
        print("Wrong Learning Rate Schedule Type.")
        sys.exit()

# Continue train ##############################
    if config.continue_train:
        print(f'=> resuming from {config.continue_train_path}')
        assert os.path.exists(config.continue_train_path)
        checkpoint_file = os.path.join(config.continue_train_path,config.continue_train_checkpoint_file_name)
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_policy.load_state_dict(checkpoint['lr_policy'])
        config.save = checkpoint['save_path']
    else:
        # create new log dir
        config.save = 'ckpt/{}/{}'.format(config.save, config.exp_name)

# config.save = 'ckpt/{}/{}'.format(config.save, config.exp_name)
    create_exp_dir(config.save)
    logger = SummaryWriter(config.save)
    config.output_dir = os.path.join(config.save, 'output/')
    create_exp_dir(config.output_dir)

    #log_format = '%(asctime)s %(message)s'
    log_format = "%(asctime)s | %(message)s"

    print("logging root handlers:")
    print(logging.root.handlers)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    #logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))
    logging.info('path=%s', path)
    logging.info("op = %s", str(op_idx_list))
    logging.info("quantize = %s", str(quantize_list))
    logging.info('ratio=%s', str(ratio_list))

    # data loader ############################

    if config.hard_gt:
        transforms_ = [ transforms.ToTensor()]
    else:
        transforms_ = [ transforms.RandomCrop(config.image_height),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()]

    #train_loader_model = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, unaligned=True),
    #                    batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    #train_loader_model = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='train'),
    #                    batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    #if config.hard_gt:
    #    train_hr_loader_model = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='train_hr'), 
    #                    batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    #else:
    #    train_hr_loader_model = None

    if config.hard_gt:
        train_loader_model = DataLoader(ImageDatasetGT(config.dataset_path, transforms_=transforms_, mode='train'),
                        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    else:
        train_loader_model = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='train'),
                            batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    transforms_ = [ transforms.ToTensor()]
    test_loader = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='val'), 
                        batch_size=1, shuffle=False, num_workers=config.num_workers)

    if config.eval_only:
        logging.info('Eval: psnr = %f', infer(0, model, test_loader, logger))
        sys.exit(0)
    tbar = tqdm(range(start_epoch, config.nepochs), ncols=80)
    for epoch in tbar:
        print('Epoch;', epoch)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(train_loader_model, model, teacher_model, optimizer, lr_policy, logger, epoch)
        # torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        #if epoch and not (epoch+1) % config.eval_epoch:
        if ((epoch+1)%config.eval_epoch)==0:
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
            
            with torch.no_grad():
                model.prun_mode = None

                valid_psnr = infer(epoch, model, test_loader, logger)
                logging.info("Epoch %d: valid_psnr %.3f"%(epoch, valid_psnr))
                logging.info("Epoch %d: flops %.3f"%(epoch, flops))
                logging.info('validation | {"epoch":%d,"valid_psnr":%.3f,"flops":%f} '%(
                            int(epoch),float(valid_psnr),float(flops)))

            os.system("rm " + os.path.join(config.save, 'weights_*.pt'))
            save(model, os.path.join(config.save, 'weights_%06d.pt'%(epoch+1)))
            os.system("rm " + os.path.join(config.save, 'checkpoint_*.pt'))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_policy': lr_policy.state_dict(),
                'save_path': config.save,
            }, False, os.path.join(config.save, "checkpoint_%06d.pt"%(epoch+1)))


def train(train_loader_model, model, teacher_model, optimizer, lr_policy, logger, epoch):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    #if config.hard_gt:
    #    hr_dataloader_model = iter(train_hr_loader_model)

    for step in pbar:
        lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()
        minibatch = dataloader_model.next()
        input = minibatch['A']
        print("train.py train() input shape check: input.shape=%s"%str(input.shape))
        input = input.cuda(non_blocking=True)

        if config.hard_gt:
            #minibatch = hr_dataloader_model.next()
            #target = minibatch['A']
            target = minibatch['A_GT']
            target = target.cuda(non_blocking=True)
        else:
            target = teacher_model(input)
        print("train.py train() target shape check: input.shape=%s"%str(target.shape))

        loss = model.module._loss(input, target)
        logging.info('train | {"epoch":%d,"step":%d,"loss":%.6f} '%(
                int(epoch),int(epoch*len(pbar)+step),float(loss)))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))
        print('\n')

    torch.cuda.empty_cache()
    del loss


def infer(epoch, model, test_loader, logger):
    print("\ntrain.py infer()\n")
    model.eval()
    for i, batch in enumerate(test_loader):
        img_name = "%06d"%i + "." + "png"
        print('infer | %s'%(img_name))
        logging.info('infer | %s'%(img_name))

        # Set model input
        real_A = Variable(batch['A']).cuda()
        fake_B = model(real_A).data.float().clamp_(0, 1)
        #save_image(fake_B, os.path.join(config.output_dir, img_name))
        os.makedirs(os.path.join(*[config.output_dir,'direct']), exist_ok=True)
        save_image(fake_B, os.path.join(*[config.output_dir,'direct',img_name]))

        fake_B_patched = patch_based_infer(real_A,model,
                patch_shape=(32,32)).data.float().clamp_(0, 1)
        os.makedirs(os.path.join(*[config.output_dir,'patched']), exist_ok=True)
        save_image(fake_B_patched, os.path.join(*[config.output_dir,'patched',"patched_"+img_name]))

    #psnr = compute_psnr(config.output_dir, config.dataset_valid_hr_path, test_Y=False, crop_border=0)
    #psnr_y = compute_psnr(config.output_dir, config.dataset_valid_hr_path, test_Y=True, crop_border=0)
    #psnr_rgb = compute_psnr(config.output_dir, config.dataset_path+'/val_hr', test_Y=False)
    #psnr_y = compute_psnr(config.output_dir, config.dataset_path+'/val_hr', test_Y=True)

    #logging.info('infer | psnr_rgb=%s'%(str(psnr_rgb)))
    #logging.info('infer | psnr_y=%s'%(str(psnr_y)))

    psnr_rgb_direct = compute_psnr(config.output_dir+'/direct', config.dataset_path+'/val_hr', test_Y=False, crop_border=0)
    psnr_y_direct = compute_psnr(config.output_dir+'/direct', config.dataset_path+'/val_hr', test_Y=True, crop_border=0)
    psnr_rgb_patched = compute_psnr(config.output_dir+'/patched', config.dataset_path+'/val_hr', test_Y=False, crop_border=0)
    psnr_y_patched = compute_psnr(config.output_dir+'/patched', config.dataset_path+'/val_hr', test_Y=True, crop_border=0)
    print('PSNR_RBG (direct):', psnr_rgb_direct)
    print('PSNR_Y (direct):', psnr_y_direct)
    print('PSNR_RGB (patched):', psnr_rgb_patched)
    print('PSNR_Y (patched):', psnr_y_patched)
    print('patch_based_infer patch size: (%d,%d)'%(32,32))

    logging.info('infer | psnr_rgb_direct=%s'%(str(psnr_rgb_direct)))
    logging.info('infer | psnr_y_direct=%s'%(str(psnr_y_direct)))
    logging.info('infer | psnr_rgb_patched=%s'%(str(psnr_rgb_patched)))
    logging.info('infer | psnr_y_patched=%s'%(str(psnr_y_patched)))

    return psnr_y_direct


if __name__ == '__main__':
    main() 
