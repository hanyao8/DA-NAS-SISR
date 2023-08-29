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

import time

from tensorboardX import SummaryWriter

from torchvision.utils import save_image

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_get_inherit_weight import config
from datasets import ImageDataset


from architect import Architect
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat, save_checkpoint
from model_search_flexible import NAS_GAN as Network
from model_infer import NAS_GAN_Infer


from RRDBNet_arch import RRDBNet
#from decode import decoder, make_cell_index_matrix
import operations
import model_search_flexible
import model_infer
from sparsemax import Sparsemax
from sparsestmax import sparsestmax
operations.ENABLE_BN = config.ENABLE_BN
model_infer.ENABLE_TANH = config.ENABLE_TANH
sparsemax = Sparsemax(dim=-1)

def path_generator(num_cell, num_up, num_cell_path_search, gamma_index):
    root_cell = [i for i in range(num_cell-num_cell_path_search)]
    root = [0 for _ in range(num_cell-num_cell_path_search)]
    root_up = []
    cell_index = num_cell-num_cell_path_search
    up_index = 0
    paths = []
    cells = []
    ups = []
    feature_map_index = 0
    stack = [(root, root_cell, root_up, 0, num_cell-num_cell_path_search)]
    while stack:
        # print(stack)
        temp_path, temp_cell, temp_up, level, layer = stack.pop()
        feature_map_index += 1
        paths.append(temp_path)
        cells.append(temp_cell)
        ups.append(temp_up)
        if layer < num_cell:
            stack.append((temp_path + [0], temp_cell + [cell_index], temp_up, level, layer+1))
            cell_index += 1
        if level < num_up:
            stack.append((temp_path + [1], temp_cell, temp_up + [up_index], level+1, layer))
            up_index += 1
    return paths[gamma_index], cells[gamma_index], ups[gamma_index]

#def main(pretrain=True):
def main():
    #assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    start_epoch = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    num_gpus = torch.cuda.device_count()
    print('num of gpus:', num_gpus)

    gpus = [0]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))

    ### Load cell architecture
    print('load cell architecture')
    state = torch.load(os.path.join(config.load_path, 'checkpoint.pt'))
    print('epoch number from checkpoint: %d'%(state['epoch']))
    print('saving...')
    torch.save(state['model_state_dict'], os.path.join(config.load_path, 'weights.pt'))
    print('save done')
    # logging.info('loading from:%s', str(os.path.join(config.load_path, 'arch.pt')))
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
    elif config.sparse_type == 'softmax':
        op_idx_all_list = F.softmax(state['alpha'], dim=-1).argmax(-1)
        quantize_all_list = F.softmax(state['beta'], dim=-1).argmax(-1) == 1
        ratio_all_list = F.softmax(state['ratio'], dim=-1).argmax(-1)
        path_index = torch.argmax(F.softmax(state['gamma'], dim=-1))
        # logging.info('gamma=%s', str(F.softmax(state['gamma'])))
    else:
        raise NotImplementedError('ivalid sparse type')
    path, cell_index, up_index = path_generator(config.num_cell, config.num_up, config.num_cell_path_search, path_index)
    num_up = sum(path)
    num_cell = len(path) - num_up
    # logging.info('gamma=%s', str(path_list))
    op_idx_list = op_idx_all_list[cell_index]
    quantize_list = quantize_all_list[cell_index]
    ratio_list = ratio_all_list[cell_index]

    print('path:', path)
    print('cell_index:', cell_index)
    print('up_index:', up_index)

    # Model #######################################
    model = Network(config.num_cell, config.op_per_cell, config.num_up, config.num_cell_path_search, config.nf, state=None, slimmable=config.slimmable, width_mult_list=config.width_mult_list, loss_weight=config.loss_weight, 
                    prun_modes=config.prun_modes, loss_func=config.loss_func, before_act=config.before_act, quantize=config.quantize, config=config)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    print('length of cell', len(model.module.cells))
    print('length of ups', len(model.module.ups))
    # state_dict = state['model_state_dict']
    partial = torch.load(os.path.join(config.load_path, 'weights.pt'))
    # partial = torch.load(state_dict)
    state = model.state_dict()
    pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
    state.update(pretrained_dict)
    model.load_state_dict(state)
    cell = 0
    up = 0
    cells = []
    for action in path:
        if action == 1:
            cells.append(model.module.ups[up_index[up]].state_dict())
            up += 1
        else:
            cells.append(model.module.cells[cell_index[cell]].state_dict())
            cell += 1

    save_checkpoint({
            'conv_first': model.module.conv_first.state_dict(),
            'cells': cells,
            'conv_last': model.module.conv_last.state_dict()
            }, False, os.path.join(config.load_path, "inherit_weight.pt")) 

if __name__ == '__main__':
    #main(pretrain=config.pretrain) 
    main()
