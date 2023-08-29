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
from collections import OrderedDict
from config_search import config
from datasets import ImageDataset

from utils.init_func import init_weight
from utils.img_utils import patch_based_infer

from architect import Architect
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat, save_checkpoint
from model_search_flexible import NAS_GAN as Network
from model_infer import NAS_GAN_Infer

from util_gan.cyclegan import Generator
from util_gan.psnr import compute_psnr,calculate_psnr
from util_gan.lr import LambdaLR

from RRDBNet_arch import RRDBNet
#from decode import decoder, make_cell_index_matrix
import operations
import model_search_flexible
import model_infer
from sparsemax import Sparsemax
from sparsestmax import sparsestmax
sparsemax = Sparsemax(dim=-1)
operations.ENABLE_BN = config.ENABLE_BN
model_search_flexible.ENABLE_TANH = model_infer.ENABLE_TANH = config.ENABLE_TANH

from quantize import QConv2d, QConvTranspose2d, QuantMeasure
from thop import profile
from thop.count_hooks import count_convNd

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
    logging.info('path_generator | paths=%s'%(str(paths)))
    logging.info('path_generator | cells=%s'%(str(cells)))
    logging.info('path_generator | gamma_index=%s'%(str(gamma_index)))
    return paths[gamma_index], cells[gamma_index]


def main(pretrain=True):
    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    if pretrain == True:
        update_arch = False
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

    # Model #######################################
    if config.load_pretrain:
        state_cells = [torch.load(config.pretrain_load_path + p) for p in ['weights_cells_0.pt', 'weights_cells_1.pt', 'weights_cells_2.pt', 'weights_cells_3.pt', 'weights_cells_4.pt']]
        state_ups = [torch.load(config.pretrain_load_path + p) for p in ['weights_ups_0.pt', 'weights_ups_1.pt']]
        state_conv_first = torch.load(config.pretrain_load_path + 'weights_conv_first.pt')
        state_HRconv = torch.load(config.pretrain_load_path + 'weights_HRconv.pt')
        state_conv_last = torch.load(config.pretrain_load_path + 'weights_conv_last.pt')
        state = {'cells':state_cells, 'ups':state_ups, 'conv_first':state_conv_first, 'HRconv':state_HRconv, 'conv_last':state_conv_last}
    else:
        state = None
        
    model = Network(config.num_cell, config.op_per_cell, config.num_up, config.num_cell_path_search, config.nf, state=state, slimmable=config.slimmable, width_mult_list=config.width_mult_list, loss_weight=config.loss_weight, 
                    prun_modes=config.prun_modes, loss_func=config.loss_func, before_act=config.before_act, quantize=config.quantize, config=config)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # for k, v in state_conv_first.items():
    #     print(v)
    # for p in model.module.conv_first.parameters():
    #     print(p)
    # print(model)

    teacher_model = RRDBNet(3, 3, 64, 23, gc=32)
    teacher_model.load_state_dict(torch.load(config.generator_A2B), strict=True)
    teacher_model = torch.nn.DataParallel(teacher_model, device_ids=gpus).cuda()
    teacher_model.eval()

    for param in teacher_model.parameters():
        param.require_grads = False

    if type(pretrain) == str:
        # partial = torch.load(pretrain + "/weights.pt")
            #partial contains (pretrained) weights.
            #Can alternatively be loaded from model_state_dict in checkpoint.

        #checkpoint = torch.load(pretrain + "/checkpoint.pt")
        checkpoint = torch.load(pretrain)
        #checkpoint = torch.load(os.path.join(pretrain,config.pretrain_checkpoint_file_name))
        partial = checkpoint['model_state_dict']
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)

    architect = Architect(model, config)

    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.module.cells.parameters())
    parameters += list(model.module.conv_first.parameters())
    parameters += list(model.module.ups.parameters())
    parameters += list(model.module.conv_last.parameters())

    arch_parameters = []
    arch_parameters += [model.module.alpha]
    arch_parameters += [model.module.beta]
    arch_parameters += [model.module.ratio]
    arch_parameters += [model.module.gamma]
    arch_parameters += [model.module.alpha_sink_attn]
    arch_parameters += [model.module.alpha_levels_attn]

    print("arch_parameters: %s"%str(arch_parameters))

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
    arch_optimizer = torch.optim.Adam(arch_parameters, lr=config.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
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
        #checkpoint_file = os.path.join(config.continue_train_path, 'checkpoint.pt')
        checkpoint_file = os.path.join(config.continue_train_path, config.continue_train_checkpoint_file_name)
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        # arch_para = torch.load(arch_file)
        start_epoch = checkpoint['epoch']
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        lr_policy.load_state_dict(checkpoint['lr_policy'])
        if update_arch:
            model.module.alpha.data = checkpoint['alpha'].data
            model.module.beta.data =  checkpoint['beta'].data
            model.module.gamma.data = checkpoint['gamma'].data
            model.module.ratio.data = checkpoint['ratio'].data
            model.module.alpha_sink_attn.data = checkpoint['alpha_sink_attn'].data
            model.module.alpha_levels_attn.data = checkpoint['alpha_levels_attn'].data
        config.save = checkpoint['save_path']
    else:
        # create new log dir
        config.save = 'ckpt/{}/{}'.format(config.save, config.exp_name)

    create_exp_dir(config.save)
    logger = SummaryWriter(config.save)

    #log_format = '%(asctime)s %(message)s'
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    #log_format = '%(asctime)s %(message)s'
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    log_format = "%(asctime)s | %(message)s"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    #logging.info('loading from:%s', str(os.path.join(config.load_path, 'arch99.pt')))
    logging.info("args = %s", str(config))



    # data loader ###########################

    transforms_ = [ transforms.RandomCrop(config.image_height),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()]
    train_loader_model = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, unaligned=True, portion=config.train_portion), 
                        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    train_loader_arch = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, unaligned=True, portion=config.train_portion-1), 
                        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    transforms_ = [ transforms.ToTensor()]
    test_loader = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='val'), 
                        batch_size=1, shuffle=False, num_workers=config.num_workers)

    tbar = tqdm(range(start_epoch, config.nepochs), ncols=80)
    valid_psnr_history = []
    flops_history = []
    flops_supernet_history = []


    for epoch in tbar:
        logging.info('Epoch' + str(epoch))
        logging.info(pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        logging.info('base loss: %f, content loss: %f, tv loss: %f, order loss: %f' % (model.module.base_loss, model.module.content_loss, model.module.tv_loss, model.module.order_loss))

        logging.info("update arch: " + str(update_arch))
        if update_arch:
            alpha_new = 0.5
            Lambda_new = config.Lambda_init + epoch * config.Lambda_step
            model.module._update_lambda(Lambda_new, alpha_new)
            logging.info("alpha sparsity: " + str(model.module.alpha_sparsity))
            logging.info("lambda sparsity: " + str(model.module.Lambda))
            rad = epoch / config.nepochs
            model.module.set_rad(rad)
            if not (epoch+1) % 1:
                if config.sparse_type == 'sparsemax':
                    logging.info('gamma:')
                    logging.info(sparsemax(model.module.gamma))
                    logging.info('op param:')
                    logging.info(sparsemax(model.module.alpha))
                    logging.info('ratio param:')
                    logging.info(F.softmax(model.module.ratio, dim=-1))
                elif config.sparse_type == 'sparsestmax':
                    logging.info('gamma:')
                    logging.info(sparsestmax(model.module.gamma, model.module.rad))
                    logging.info('op param:')
                    logging.info(sparsestmax(model.module.alpha, model.module.rad))
                    logging.info('ratio param:')
                    logging.info(F.softmax(model.module.ratio, dim=-1))
                elif config.sparse_type == 'softmax':
                    logging.info('gamma:')
                    logging.info(F.softmax(model.module.gamma, dim=-1))
                    logging.info('op param:')
                    logging.info(F.softmax(model.module.alpha, dim=-1))
                    logging.info('ratio param:')
                    logging.info(F.softmax(model.module.ratio, dim=-1))
                else:
                    raise NotImplementedError('not valid sparse_type')

                if config.attn_sparse_type == 'sparsemax':
                    logging.info('alpha_sink_attn: (sparsemax)')
                    logging.info(sparsemax(model.module.alpha_sink_attn))
                    logging.info('alpha_levels_attn: (sparsemax)')
                    logging.info(sparsemax(model.module.alpha_levels_attn))
                elif config.attn_sparse_type == 'sparsestmax':
                    logging.info('alpha_sink_attn: (sparsestmax)')
                    logging.info(sparsestmax(model.module.alpha_sink_attn, model.module.rad))
                    logging.info('alpha_levels_attn: (sparsestmax)')
                    logging.info(sparsestmax(model.module.alpha_levels_attn, model.module.rad))
                elif config.attn_sparse_type == 'softmax':
                    logging.info('alpha_sink_attn: (softmax)')
                    logging.info(F.softmax(model.module.alpha_sink_attn, dim=-1))
                    logging.info('alpha_levels_attn: (softmax)')
                    logging.info(F.softmax(model.module.alpha_levels_attn, dim=-1))
                else:
                    raise NotImplementedError('not valid attn sparse_type')

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(pretrain, train_loader_model, train_loader_arch, model, architect, arch_optimizer, teacher_model, optimizer, lr_policy, logger, epoch, update_arch=update_arch)
        # torch.cuda.empty_cache()
        lr_policy.step()
        if epoch:# and not (epoch+1) % config.save_epoch:
            start = time.time()

            os.system("rm " + os.path.join(config.save, 'checkpoint_*.pt'))

            start = time.time()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'arch_optimizer': arch_optimizer.state_dict(),
                # 'arch_optimizer_gamma': arch_optimizer_gamma.state_dict(),
                'lr_policy': lr_policy.state_dict(),
                'save_path': config.save,
                'alpha': model.module.alpha,
                'beta': model.module.beta,
                'ratio': model.module.ratio,
                'gamma': model.module.gamma,
                'alpha_sink_attn': model.module.alpha_sink_attn,
                'alpha_levels_attn': model.module.alpha_levels_attn,
            }, False, os.path.join(config.save, "checkpoint_%06d.pt"%(epoch+1)))

        if config.infer:
            # validation
            #if epoch+1:# and not (epoch+1) % config.eval_epoch:
            if epoch and not (epoch+1) % config.eval_epoch:
                tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
                # print('infer phase...')

                #save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))
                os.system("rm " + os.path.join(config.save, 'weights_*.pt'))
                save(model, os.path.join(config.save, 'weights_%06d.pt'%(epoch+1)))

                with torch.no_grad():
                    if pretrain == True:
                        model.module.prun_mode = "min"
                        valid_psnr = infer(epoch, model, test_loader, logger)
                        #logger.add_scalar('psnr/val_min', valid_psnr, epoch)
                        #logging.info("Epoch %d: valid_psnr_min %.3f"%(epoch, valid_psnr))
                        logging.info('validation | {"epoch":%d,"valid_psnr_min":%.3f} '%(
                            int(epoch),float(valid_psnr)))

                        if len(model.module._width_mult_list) > 1:
                            model.module.prun_mode = "max"
                            valid_psnr = infer(epoch, model, test_loader, logger)
                            #logger.add_scalar('psnr/val_max', valid_psnr, epoch)
                            logging.info("Epoch %d: valid_psnr_max %.3f"%(epoch, valid_psnr))

                            model.module.prun_mode = "random"
                            valid_psnr = infer(epoch, model, test_loader, logger)
                            #logger.add_scalar('psnr/val_random', valid_psnr, epoch)
                            #logging.info("Epoch %d: valid_psnr_random %.3f"%(epoch, valid_psnr))
                            logging.info('validation | {"epoch":%d,"valid_psnr_random":%.3f} '%(int(epoch),float(valid_psnr)))

                    else:
                        model.module.prun_mode = None

                        valid_psnr, flops = infer(epoch, model, test_loader, logger, finalize=True)

                        #logger.add_scalar('psnr/val', valid_psnr, epoch)
                        #logging.info("Epoch %d: valid_psnr %.3f"%(epoch, valid_psnr))
                        
                        #logger.add_scalar('flops/val', flops, epoch)
                        #logging.info("Epoch %d: flops %.3f"%(epoch, flops))

                        logging.info('validation | {"epoch":%d,"valid_psnr":%.3f,"flops":%f} '%(
                            int(epoch),float(valid_psnr),float(flops)))
                        print('validation | {"epoch":%d,"valid_psnr":%.3f,"flops":%f} '%(
                            int(epoch),float(valid_psnr),float(flops)))

                        valid_psnr_history.append(valid_psnr)
                        flops_history.append(flops)
                        
                        if update_arch:
                            flops_supernet_history.append(architect.flops_supernet)

                    if update_arch:
                        state = {}
                        state['alpha'] = getattr(model.module, 'alpha')
                        state['beta'] = getattr(model.module, 'beta')
                        state['ratio'] = getattr(model.module, 'ratio')
                        state['gamma'] = getattr(model.module, 'gamma')
                        state['alpha_sink_attn'] = getattr(model.module, 'alpha_sink_attn')
                        state['alpha_levels_attn'] = getattr(model.module, 'alpha_levels_attn')
                        state["psnr"] = valid_psnr
                        state["flops"] = flops

                        os.system("rm " + os.path.join(config.save, 'arch_*.pt'))
                        torch.save(state, os.path.join(config.save, "arch_%06d.pt"%(epoch+1)))

                        if config.flops_weight > 0:
                            if flops < config.flops_min:
                                architect.flops_weight /= 2
                            elif flops > config.flops_max:
                                architect.flops_weight *= 2
                            #logger.add_scalar("arch/flops_weight", architect.flops_weight, epoch+1)
                            logging.info("arch_flops_weight = " + str(architect.flops_weight))
                            logging.info('validation | {"epoch":%d,"valid_psnr":%.3f,"flops":%f,"log10_arch_flops_weight":%f} '%(
                                int(epoch),float(valid_psnr),float(flops),float(np.log10(architect.flops_weight))))

    #save(model, os.path.join(config.save, 'weights.pt'))
    
    if update_arch:
        state = {}
        state['alpha'] = model.module.alpha
        state['beta'] = model.module.beta
        state['ratio'] = model.module.ratio
        state['gamma'] = model.module.gamma
        state['alpha_sink_attn'] = model.module.alpha_sink_attn
        state['alpha_levels_attn'] = model.module.alpha_levels_attn
        torch.save(state, os.path.join(config.save, "arch.pt"))


def train(pretrain, train_loader_model, train_loader_arch, model, architect, arch_optimizer, teacher_model, optimizer, lr_policy, logger, epoch, update_arch=True):
    logging.info('inside train, optimizer and lr info start')
    #logging.info('(take care: initial lr here?) '+str(optimizer.param_groups))
    opt_state_dict = optimizer.state_dict()
    logging.info('optimizer state_dict lr: '+str(opt_state_dict['param_groups'][0]['lr']))
    logging.info('optimizer state_dict initial_lr: '+str(opt_state_dict['param_groups'][0]['initial_lr']))
    logging.info('lr_policy last lr: '+str(lr_policy.get_last_lr()))
    #logging.info(str(lr_policy.print_lr()))
    logging.info(str(lr_policy.state_dict()))
    logging.info('inside train, optimizer and lr info end')

    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    for step in pbar:
        #minibatch = dataloader_model.next()
        try:
            minibatch = dataloader_model.next()
        except StopIteration:
            dataloader_model = iter(train_loader_model)
            minibatch = dataloader_model.next()
        
        input = minibatch['A']
        input = input.cuda(non_blocking=True)
        target = teacher_model(input)

        if update_arch:
            if config.sparse_type == 'sparsestmax' or config.attn_sparse_type == 'sparsestmax':
                rad = (epoch*config.niters_per_epoch+step)/(config.nepochs*config.niters_per_epoch)
                print('sparsestmax rad=%s'%str(rad))
                model.module.set_rad(rad)
                assert (rad>=0 and rad<=1)

            pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_arch)))

            minibatch = dataloader_arch.next()
            input_search = minibatch['A']
            input_search = input_search.cuda(non_blocking=True)
            target_search = teacher_model(input_search)
            loss_arch = architect.step(input, target, input_search, target_search, arch_optimizer, epoch)
            if (step+1) % 1000 == 0:
                logging.info('base loss: %f, content loss: %f, tv loss: %f, order loss: %f' % (
                        model.module.base_loss, model.module.content_loss, model.module.tv_loss, model.module.order_loss))
                if config.sparse_type == 'sparsemax':
                    logging.info('gamma:')
                    logging.info(sparsemax(model.module.gamma))
                    logging.info('op param:')
                    logging.info(sparsemax(model.module.alpha))
                    logging.info('ratio param:')
                    logging.info(F.softmax(model.module.ratio, dim=-1))
                elif config.sparse_type == 'sparsestmax':
                    logging.info('gamma:')
                    logging.info(sparsestmax(model.module.gamma, model.module.rad))
                    logging.info('op param:')
                    logging.info(sparsestmax(model.module.alpha, model.module.rad))
                    logging.info('ratio param:')
                    logging.info(F.softmax(model.module.ratio, dim=-1))
                elif config.sparse_type == 'softmax':
                    logging.info('gamma:')
                    logging.info(F.softmax(model.module.gamma, dim=-1))
                    logging.info('op param:')
                    logging.info(F.softmax(model.module.alpha, dim=-1))
                    logging.info('ratio param:')
                    logging.info(F.softmax(model.module.ratio, dim=-1))
                else:
                    raise NotImplementedError('not valid sparse_type')

                if config.attn_sparse_type == 'sparsemax':
                    logging.info('alpha_sink_attn: (sparsemax)')
                    logging.info(sparsemax(model.module.alpha_sink_attn))
                    logging.info('alpha_levels_attn: (sparsemax)')
                    logging.info(sparsemax(model.module.alpha_levels_attn))
                elif config.attn_sparse_type == 'sparsestmax':
                    logging.info('alpha_sink_attn: (sparsestmax)')
                    logging.info(sparsestmax(model.module.alpha_sink_attn, model.module.rad))
                    logging.info('alpha_levels_attn: (sparsestmax)')
                    logging.info(sparsestmax(model.module.alpha_levels_attn, model.module.rad))
                elif config.attn_sparse_type == 'softmax':
                    logging.info('alpha_sink_attn: (softmax)')
                    logging.info(F.softmax(model.module.alpha_sink_attn, dim=-1))
                    logging.info('alpha_levels_attn: (softmax)')
                    logging.info(F.softmax(model.module.alpha_levels_attn, dim=-1))
                else:
                    raise NotImplementedError('not valid attn sparse_type')

            #if (step+1) % 10 == 0:
            #    logger.add_scalar('loss_arch/train', loss_arch, epoch*len(pbar)+step)
            #    logger.add_scalar('arch/flops_supernet', architect.flops_supernet, epoch*len(pbar)+step)
            if (step+1) % 10 == 0:
                logging.info('train_arch | {"epoch":%d,"step":%d,"loss_arch":%.6f,"flops_supernet":%f} '%(
                    int(epoch),int(epoch*len(pbar)+step),float(loss_arch),float(architect.flops_supernet)))

        loss = model.module._loss(input, target, pretrain)


        optimizer.zero_grad()
        #logger.add_scalar('loss/train', loss, epoch*len(pbar)+step)
        logging.info('train | {"epoch":%d,"step":%d,"loss":%.6f} '%(
                int(epoch),int(epoch*len(pbar)+step),float(loss)))
        loss.backward()

        if (step+1) % 1000 == 0:
            print(model.parameters())
            #for param in model.parameters():
            #    print(type(param), param.size())
            total_nan_sum = 0
            total_num_params = 0
            for name, param in model.named_parameters():
                nan_sum = torch.sum( torch.isnan(param).type(torch.uint8) ).item()
                num_params = torch.prod(torch.tensor(param.size())).item()
                layer_max = torch.max(param).item()
                layer_min = torch.min(param).item()
                #print(name, param.size(), type(param))
                print(name, param.size(),'num_params:',num_params)
                print('nan_sum:',nan_sum,'layer_min:',layer_min,'layer_max:',layer_max)
                #print('num_params:',num_params)
                total_nan_sum += nan_sum
                total_num_params += num_params
            
            print(param)
            print('total_nan_sum:',total_nan_sum)
            print('total_num_params:',total_num_params)
            print('\n')

        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))

    torch.cuda.empty_cache()
    del loss
    if update_arch: del loss_arch


def infer(epoch, model, test_loader, logger, finalize=False):
    print("\ntrain.py infer()\n")
    model.eval()
    with torch.no_grad(): 
        for i, batch in enumerate(test_loader):
            #img_name = '08%02d_gen.png' % (i+1) if i < 99 else '0900_gen.png'
            img_name = "%06d"%i + "." + "png"
            logging.info('infer | %s'%(img_name))

            # Set model input
            try:
                real_A = Variable(batch['A']).cuda()
                if config.attention:
                    fake_B_patched = patch_based_infer(real_A,model,
                            patch_shape=(config.image_height,config.image_width)).data.float().clamp_(0, 1)
                    save_image(fake_B_patched, os.path.join(config.output_dir, "patched_"+img_name))
                    del fake_B_patched
                else:
                    fake_B = model(real_A).data.float().clamp_(0, 1)
                    fake_B_patched = patch_based_infer(real_A,model,
                            patch_shape=(config.image_height,config.image_width)).data.float().clamp_(0, 1)
                    psnr_patched_diff = calculate_psnr(
                        fake_B_patched.cpu().detach().numpy()*255.0,fake_B.cpu().detach().numpy()*255.0)
                    print("psnr_patched_diff: ",psnr_patched_diff)
                    save_image(fake_B, os.path.join(config.output_dir, img_name))
                    save_image(fake_B_patched, os.path.join(config.output_dir, "patched_"+img_name))
                    del fake_B
                    del fake_B_patched
            except RuntimeError as exception:
                print("exception: %s"%(str(exception)))
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            
            #save_image(fake_B, os.path.join('output/B_nasgan', img_name))
            #save_image(fake_B, os.path.join(config.output_dir, img_name))
            #del fake_B

        #psnr = compute_psnr('output/B_nasgan', config.dataset_path + '/val_hr')
        #psnr = compute_psnr(config.output_dir, config.dataset_path + '/val_hr')
        psnr = compute_psnr(config.output_dir, config.dataset_path+'/val_hr', test_Y=False)
        psnr_y = compute_psnr(config.output_dir, config.dataset_path+'/val_hr', test_Y=True)
        logging.info('infer | psnr=%s'%(str(psnr)))
        logging.info('infer | psnr_y=%s'%(str(psnr_y)))

        if finalize:
            alpha = getattr(model.module, 'alpha')
            beta = getattr(model.module, 'beta')
            ratio = getattr(model.module, 'ratio')
            gamma = getattr(model.module, 'gamma')
            alpha_sink_attn = getattr(model.module, 'alpha_sink_attn')
            alpha_levels_attn = getattr(model.module, 'alpha_levels_attn')

            print("infer finalize, arch_param shapes")
            print(alpha.shape)
            print(beta.shape)
            print(ratio.shape)
            print(gamma.shape)
            print(alpha_sink_attn.shape)
            print(alpha_levels_attn.shape)

            if config.sparse_type == 'sparsemax':
                op_idx_all_list = sparsemax(alpha).argmax(-1)
                quantize_all_list = sparsemax(beta).argmax(-1) == 1
                ratio_all_list = sparsemax(ratio).argmax(-1)
                path_index = torch.argmax(sparsemax(gamma))
            elif config.sparse_type == 'sparsestmax':
                op_idx_all_list = sparsestmax(alpha, 1).argmax(-1)
                quantize_all_list = sparsestmax(beta, 1).argmax(-1) == 1
                ratio_all_list = sparsestmax(ratio, 1).argmax(-1)
                path_index = torch.argmax(sparsestmax(gamma, 1))
            elif config.sparse_type == 'softmax':
                op_idx_all_list = F.softmax(alpha, dim=-1).argmax(-1)
                quantize_all_list = F.softmax(beta, dim=-1).argmax(-1) == 1
                ratio_all_list = F.softmax(ratio, dim=-1).argmax(-1)
                path_index = torch.argmax(F.softmax(gamma, dim=-1))
            else:
                raise NotImplementedError('ivalid sparse type')
            
            logging.info('infer | op_idx_all_list=%s'%(str(op_idx_all_list)))
            logging.info('infer | quantize_all_list=%s'%(str(quantize_all_list)))
            logging.info('infer | ratio_all_list=%s'%(str(ratio_all_list)))
            logging.info('infer | path_index=%s'%(str(path_index)))
            
            path, cell_index = path_generator(config.num_cell, config.num_up, config.num_cell_path_search, path_index)
            logging.info('infer | (epoch %s) path=%s'%(str(epoch),str(path)))
            logging.info('infer | cell_index=%s'%(str(cell_index)))
            num_up = sum(path)
            num_cell = len(path) - num_up
            op_idx_list = op_idx_all_list[cell_index]
            quantize_list = quantize_all_list[cell_index]
            ratio_list = ratio_all_list[cell_index]

            logging.info('infer | num_up=%s'%(str(num_up)))
            logging.info('infer | num_cell=%s'%(str(num_cell)))
            logging.info('infer | op_idx_list=%s'%(str(op_idx_list)))
            logging.info('infer | quantize_list=%s'%(str(quantize_list)))
            logging.info('infer | ratio_list=%s'%(str(ratio_list)))

            if config.attention:
                attn_sink_k = alpha_sink_attn.shape[-1]
                attn_levels_t = alpha_levels_attn.shape[-1]
            else:
                alpha_sink_attn = None
                alpha_levels_attn = None
                attn_sink_k = None
                attn_levels_t = None

            #model_infer = NAS_GAN_Infer(getattr(model.module, 'alpha'), getattr(model.module, 'beta'), getattr(model.module, 'ratio'), getattr(model.module, 'gamma'), num_cell=config.num_cell, op_per_cell=config.op_per_cell, num_up=config.num_up, num_cell_path_search=config.num_cell_path_search,
            #                            width_mult_list=config.width_mult_list, loss_weight=config.loss_weight, loss_func=config.loss_func, before_act=config.before_act, quantize=config.quantize)
            model = NAS_GAN_Infer(op_idx_list, quantize_list, ratio_list, path,
                          num_cell=num_cell, op_per_cell=config.op_per_cell, num_up=num_up, num_cell_path_search=config.num_cell_path_search, 
                          width_mult_list=config.width_mult_list, loss_weight=config.loss_weight, loss_func=config.loss_func,
                          before_act=config.before_act, quantize=config.quantize, nf=config.nf,
                          iv_mode=config.iv_mode,attention=config.attention,alpha_sink_attn=alpha_sink_attn,alpha_levels_attn=alpha_levels_attn,
                          sink_k=attn_sink_k,levels_t=attn_levels_t,primitives_attn=config.primitives_attn,trilevelnas_x=config.trilevelnas_x)

            #flops = 0
            #flops = model_infer.forward_flops((3, 510, 350))
            #flops, params = profile(model, inputs=(torch.randn(1, 3, 256, 256),), custom_ops=custom_ops)
            #logging.info("two flops:")
            #logging.info("flops:")
            #logging.info(str(flops))
            #logging.info('infer | {"epoch":%d,"flops1":%f} '%(int(epoch),float(flops)))
            #flops = model.forward_flops(size=(3, 256, 256))
            #logging.info('infer | {"epoch":%d,"flops2":%f} '%(int(epoch),float(flops)))
            #logging.info(str(flops))

            if not(config.attention):
                flops, params = profile(model, inputs=(torch.randn(1, 3, 256, 256),), custom_ops=custom_ops)
                logging.info("params = %fMB, FLOPs = %fGFlops (thop profile 3,256,256)"%(params/1e6,flops/1e9))
                print("params = %fMB, FLOPs = %fGFlops (thop profile 3,256,256)"%(params/1e6,flops/1e9))

            flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), custom_ops=custom_ops)
            logging.info("params = %fMB, FLOPs = %fGFlops (thop profile 3,32,32)"%(params/1e6,flops/1e9))
            print("params = %fMB, FLOPs = %fGFlops (thop profile 3,32,32)"%(params/1e6,flops/1e9))

            return psnr_y, flops

        else:
            return psnr_y


if __name__ == '__main__':
    main(pretrain=config.pretrain) 
