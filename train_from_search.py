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

from config_train_from_search import config
from datasets import ImageDataset

from utils.init_func import init_weight

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat, save_checkpoint
from model_infer import NAS_GAN_Infer

from util_gan.cyclegan import Generator
from util_gan.psnr import compute_psnr
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
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    state = torch.load(os.path.join(config.load_path, 'checkpoint.pt'))
    if config.sparse_type == 'sparsemax':
        op_idx_all_list = sparsemax(state['alpha']).argmax(-1)
        quantize_all_list = sparsemax(state['beta']).argmax(-1) == 1
        ratio_all_list = sparsemax(state['ratio']).argmax(-1)
        path_index = torch.argmax(sparsemax(state['gamma']))
    elif config.sparse_type == 'sparsestmax':
        op_idx_all_list = sparsestmax(state['alpha'], 1).argmax(-1)
        quantize_all_list = sparsestmax(state['beta'], 1).argmax(-1) == 1
        ratio_all_list = sparsestmax(state['ratio'], 1).argmax(-1)
        path_index = torch.argmax(sparsestmax(state['gamma'], 1))
    elif config.sparse_type == 'softmax':
        op_idx_all_list = F.softmax(state['alpha'], dim=-1).argmax(-1)
        quantize_all_list = F.softmax(state['beta'], dim=-1).argmax(-1) == 1
        ratio_all_list = F.softmax(state['ratio'], dim=-1).argmax(-1)
        path_index = torch.argmax(F.softmax(state['gamma'], dim=-1))
    else:
        raise NotImplementedError('ivalid sparse type')
    path, cell_index = path_generator(config.num_cell, config.num_up, config.num_cell_path_search, path_index)
    num_up = sum(path)
    num_cell = len(path) - num_up
    # logging.info('gamma=%s', str(path_list))
    op_idx_list = op_idx_all_list[cell_index]
    quantize_list = quantize_all_list[cell_index]
    ratio_list = ratio_all_list[cell_index]
  
    # Model #######################################
    model = NAS_GAN_Infer(op_idx_list, quantize_list, ratio_list, path, num_cell=num_cell, op_per_cell=config.op_per_cell, num_up=num_up, num_cell_path_search=config.num_cell_path_search, 
                          width_mult_list=config.width_mult_list, 
                          loss_weight=config.loss_weight, loss_func=config.loss_func, before_act=config.before_act, quantize=config.quantize, nf=config.nf)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 256, 256),), custom_ops=custom_ops)
    flops = model.forward_flops(size=(3, 256, 256))
    # logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)

    model = torch.nn.DataParallel(model).cuda()

    if type(config.pretrain) == str:
        state_dict = torch.load(config.pretrain)
        model.load_state_dict(state_dict)
        print('finetune phase')
    else:
        print('pretrain phase')
        # print('original:', model.module.cells[3].upconv.conv.weight)
        inherit_weight = torch.load(os.path.join(config.load_path, 'inherit_weight.pt'))
        model.module.conv_first.load_state_dict(inherit_weight['conv_first'])
        assert len(inherit_weight['cells']) == len(model.module.cells)
        print('length of inherited cells:', len(inherit_weight['cells']))
        # print('original:', model.module.cells[1].upconv.conv.weight)
        for cell in range(len(inherit_weight['cells'])):
            partial = inherit_weight['cells'][cell]
            temp_state = model.module.cells[cell].state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in temp_state and temp_state[k].size() == partial[k].size()}
            temp_state.update(pretrained_dict)
            model.module.cells[cell].load_state_dict(temp_state)
        model.module.conv_last.load_state_dict(inherit_weight['conv_last'])
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
    total_iteration = config.nepochs * config.niters_per_epoch

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
        checkpoint_file = os.path.join(config.continue_train_path, 'checkpoint.pt')
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

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    #logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))
    logging.info('num_cell=%s', str(config.num_cell))
    logging.info('num_cell_search_path=%s', str(config.num_cell_path_search))

    logging.info('path index=%s', str(path_index))
    logging.info('num cell=%s', str(num_cell))
    logging.info('path=%s', path)
    logging.info('cell index=%s', str(cell_index))

    if config.sparse_type == 'sparsemax':
        logging.info('op params = %s', str(sparsemax(state['alpha'])[cell_index]))
    elif config.sparse_type == 'sparsestmax':
        logging.info('op params = %s', str(sparsestmax(state['alpha'], 1)[cell_index]))
    logging.info("op = %s", str(op_idx_list))
    if config.sparse_type == 'sparsemax':
        logging.info('quantize params = %s', str(sparsemax(state['beta'])[cell_index]))
    elif config.sparse_type == 'sparsetmax':
        logging.info('quantize params = %s', str(sparsestmax(state['beta'], 1)[cell_index]))
    logging.info("quantize = %s", str(quantize_list))
    if config.sparse_type == 'sparsemax':
        logging.info('ratio params = %s', str(sparsemax(state['ratio'])[cell_index]))
    elif config.sparse_type == 'sparsestmax':
        logging.info('ratio params = %s', str(sparsestmax(state['ratio'], 1)[cell_index]))
    logging.info('ratio=%s', str(ratio_list))

    # data loader ############################

    transforms_ = [ transforms.RandomCrop(config.image_height),  # config.image_height=32
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor()]
    train_loader_model = DataLoader(ImageDataset(config.dataset_train_path, transforms_=transforms_, unaligned=True), 
                        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    transforms_ = [ transforms.ToTensor()]
    test_loader = DataLoader(ImageDataset(config.dataset_valid_path, transforms_=transforms_, mode='val'), 
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
        if epoch and not (epoch+1) % config.eval_epoch:
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
            
            with torch.no_grad():
                model.prun_mode = None

                valid_psnr = infer(epoch, model, test_loader, logger)

                #logger.add_scalar('psnr/val', valid_psnr, epoch)
                #logging.info("Epoch %d: valid_psnr %.3f"%(epoch, valid_psnr))
                
                #logger.add_scalar('flops/val', flops, epoch)
                #logging.info("Epoch %d: flops %.3f"%(epoch, flops))

                logging.info('validation | {"epoch":%d,"valid_psnr":%.3f,"flops":%.3f} '%(
                                int(epoch),float(valid_psnr),float(flops)))

            os.system("rm " + os.path.join(config.save, 'weights_*.pt'))
            #save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))
            save(model, os.path.join(config.save, 'weights_%06d.pt'%epoch))

            save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_policy': lr_policy.state_dict(),
            'save_path': config.save,
            }, False, os.path.join(config.save, "checkpoint.pt"))

            os.system("rm " + os.path.join(config.save, 'checkpoint_*.pt'))

            save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_policy': lr_policy.state_dict(),
            'save_path': config.save,
            }, False, os.path.join(config.save, "checkpoint_%06d.pt"%epoch))

    save(model, os.path.join(config.save, 'weights.pt'))


def train(train_loader_model, model, teacher_model, optimizer, lr_policy, logger, epoch):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)


    for step in pbar:
        lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        minibatch = dataloader_model.next()
        input = minibatch['A']
        input = input.cuda(non_blocking=True)
        target = teacher_model(input)

        loss = model.module._loss(input, target)

        #logger.add_scalar('loss/train', loss, epoch*len(pbar)+step)
        logging.info('train | {"epoch":%d,"step":%d,"loss":%.6f} '%(
                int(epoch),int(epoch*len(pbar)+step),float(loss)))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))

    torch.cuda.empty_cache()
    del loss


def infer(epoch, model, test_loader, logger):
    model.eval()

    for i, batch in enumerate(test_loader):
        # Set model input
        real_A = Variable(batch['A']).cuda()
        fake_B = model(real_A).data.float().clamp_(0, 1)

        #img_name = '08%02d_gen.png' % (i+1) if i < 99 else '0900_gen.png'
        img_name = "%06d"%i + "." + "png"
        logging.info('infer %s'%(img_name))

        save_image(fake_B, os.path.join(config.output_dir, img_name))

    psnr = compute_psnr(config.output_dir, config.dataset_valid_hr_path)
    print('save image')
    return psnr


if __name__ == '__main__':
    main() 
