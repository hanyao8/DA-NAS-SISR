import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from operations_attn import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
# from utils.darts_utils import drop_path, compute_speed, compute_speed_tensorrt
from pdb import set_trace as bp
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from util_gan.vgg_feature import VGGFeature
from thop import profile
import time
from sparsemax import Sparsemax
from sparsestmax import sparsestmax
ENABLE_TANH = False
sparsemax = Sparsemax(dim=-1)
# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())#.to(logits.device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class SingleOp(nn.Module):
    def __init__(self, op, C_in, C_out, kernel_size=3 , stride=1, slimmable=True, width_mult_list=[1.], quantize=True):
        super(SingleOp, self).__init__()
        self._op = op(C_in, C_out, kernel_size=kernel_size, stride=stride, slimmable=slimmable, width_mult_list=width_mult_list)
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.slimmable = slimmable

    def set_prun_ratio(self, ratio):
        self._op.set_ratio(ratio)

    def forward(self, x, beta, ratio):
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        if self.quantize == 'search':
            result = (beta[0]*self._op(x, quantize=False) + beta[1]*self._op(x, quantize=True)) * r_score0 * r_score1
        elif self.quantize:
            result = self._op(x, quantize=True) * r_score0 * r_score1
        else:
            result = self._op(x, quantize=False) * r_score0 * r_score1

        return result

    def forward_flops(self, size, beta, ratio):
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        if self.quantize == 'search':
            flops_full, size_out = self._op.forward_flops(size, quantize=False)
            flops_quant, _ = op.forward_flops(size, quantize=True)
            flops = beta[0] * flops_full + beta[1] * flops_quant
        elif self.quantize:
            flops, size_out = op.forward_flops(size, quantize=True)
        else:
            flops, size_out = self._op.forward_flops(size, quantize=False)

        flops = flops * r_score0 * r_score1

        return flops, size_out


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.], quantize=True):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.slimmable = slimmable

        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, slimmable=slimmable, width_mult_list=width_mult_list)
            self._ops.append(op)

    def set_prun_ratio(self, ratio):
        for op in self._ops:
            op.set_ratio(ratio)

    def forward(self, x, alpha, beta, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        for w, op in zip(alpha, self._ops):
            if self.quantize == 'search':
                result = result + (beta[0]*op(x, quantize=False) + beta[1]*op(x, quantize=True)) * w * r_score0 * r_score1
            elif self.quantize:
                result = result + op(x, quantize=True) * w * r_score0 * r_score1
            else:
                result = result + op(x, quantize=False) * w * r_score0 * r_score1
            # print(type(op), result.shape)
        return result


    def forward_latency(self, size, alpha, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        for w, op in zip(alpha, self._ops):
            latency, size_out = op.forward_latency(size)
            result = result + latency * w * r_score0 * r_score1
        return result, size_out


    def forward_flops(self, size, alpha, beta, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        for w, op in zip(alpha, self._ops):
            if self.quantize == 'search':
                flops_full, size_out = op.forward_flops(size, quantize=False)
                flops_quant, _ = op.forward_flops(size, quantize=True)
                flops = (beta[0] * flops_full + beta[1] * flops_quant)

            elif self.quantize:
                flops, size_out = op.forward_flops(size, quantize=True)

            else:
                flops, size_out = op.forward_flops(size, quantize=False)

            result = result + flops * w * r_score0 * r_score1

        return result, size_out


class MixedOp_attn(nn.Module):
    def __init__(self,num_frames,num_channels_in,iv_mode,primitives_attn):
        super(MixedOp_attn, self).__init__()
        print('initialize MixedOp_attn')

        self.iv_mode = iv_mode
        #if iv_mode=='image':
        #    PRIMITIVES_attn = PRIMITIVES_attn_image
        #elif iv_mode=='video':
        #    PRIMITIVES_attn = PRIMITIVES_attn_video
        #else:
        #    raise Exception('invalid mode')

        self.primitives_attn = primitives_attn
        self.num_ops_attn = len(primitives_attn)
        self._ops_attn = nn.ModuleList()
        for i, op_attn_name in enumerate(primitives_attn):
            self._ops_attn.append(OPS_Attention[op_attn_name](num_channels_in,num_frames))

    def forward(self,x_source,alpha_sink,alpha_levels):
        x_sink = 0
        if self.iv_mode == 'image':
            #print('msf line 250')
            #print(alpha_sink.shape)
            #print(alpha_levels.shape)
            #assert alpha_sink.shape[-1] == 6 #temporary
            #assert alpha_levels.shape[0]*alpha_levels.shape[1]*alpha_levels.shape[2] == 9 #temporary
            B,C,H,W=x_source.shape
            x_source = x_source.view(B,1,C,H,W)

        l1_out = [0]*self.num_ops_attn
        for i in range(self.num_ops_attn):
            #print(self.primitives_attn[i])
            l1_out[i] = self._ops_attn[i](x_source)
            #x_sink += l1_out[i] * alpha_sink[0*self.num_ops_attn+i] #level 1
            x_sink += l1_out[i] * alpha_sink[i]
            #print("\n")

        l2_in = [0]*self.num_ops_attn
        for i in range(self.num_ops_attn):
            for j in range(self.num_ops_attn):
                l2_in[j] += l1_out[i] * alpha_levels[0][i][j]

        for i in range(self.num_ops_attn):
            #print(self.primitives_attn[i])
            x_sink += self._ops_attn[i](l2_in[i]) * alpha_sink[self.num_ops_attn+i] #level 2

        if self.iv_mode == 'image':
            x_sink = x_sink.view(B,C,H,W)
        return(x_sink)


class Cell(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf=64, op_per_cell=5, slimmable=True, width_mult_list=[1.], quantize=False):
        super(Cell, self).__init__()

        self.nf = nf
        self.op_per_cell = op_per_cell
        self.slimmable = slimmable
        self._width_mult_list = width_mult_list
        self.quantize = quantize

        self.ops = nn.ModuleList()

        for _ in range(op_per_cell):
            self.ops.append(MixedOp(self.nf, self.nf, slimmable=slimmable, width_mult_list=width_mult_list, quantize=quantize))

    def forward(self, x, alpha, beta, ratio):
        out = x

        for i, op in enumerate(self.ops):
            if i == 0:
                out = op(out, alpha[i], beta[i], [1, ratio[i]])
            elif i == self.op_per_cell - 1:
                out = op(out, alpha[i], beta[i], [ratio[i-1], 1])
            else:
                out = op(out, alpha[i], beta[i], [ratio[i-1], ratio[i]])

        return out*0.2 + x


    def forward_flops(self, size, alpha, beta, ratio):
        flops_total = []

        for i, op in enumerate(self.ops):
            if i == 0:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [1, ratio[i]])
                flops_total.append(flops)
            elif i == self.op_per_cell - 1:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [ratio[i-1], 1])
                flops_total.append(flops)               
            else:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [ratio[i-1], ratio[i]])
                flops_total.append(flops)

        return sum(flops_total), size


class Cell_attn(nn.Module):
    def __init__(self,num_frames,num_channels_in,op_per_cell,iv_mode,primitives_attn):
        super(Cell_attn, self).__init__()
        self.ops = nn.ModuleList()
        for i in range(op_per_cell):
            self.ops.append(
                MixedOp_attn(num_frames,num_channels_in,iv_mode,primitives_attn))

    def forward(self, x, alpha_sink, alpha_levels):
        out = x
        for i, op in enumerate(self.ops):
            out = op(out,alpha_sink[i],alpha_levels[i])
        return out + x


class UpSample(nn.Module):
    def __init__(self, nf=64, quantize=False):
        super(UpSample, self).__init__()
        self.nf = nf

        self.upconv = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
                        
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # output = self.upconv(x)
        output = self.upconv(x)
        # output = self.lrelu(self.upconv(F.interpolate(x, scale_factor=2, mode='nearest')))

        return output

    def forward_flops(self, size):
        # size = (size[0], size[1]*2, size[2]*2)
        flops, size = self.upconv.forward_flops(size)
        
        return flops, size


class NAS_GAN(nn.Module):
    def __init__(self, num_cell=5, op_per_cell=5, num_up=2, num_cell_path_search=3, nf=64,
                state=None, slimmable=True, width_mult_list=[1.,], loss_weight = [1e0, 1e5, 1e0, 1e-7],
                prun_modes='arch_ratio', loss_func='MSE', before_act=True, quantize=False, config=None):

        super(NAS_GAN, self).__init__()
        # self.device = device
        if config:
            self.config = config
        
        self.num_cell = num_cell
        self.op_per_cell = op_per_cell
        self.num_up = num_up
        self.num_cell_path_search = num_cell_path_search

        # self._layers = num_cell * op_per_cell
        self.length = self.num_cell + self.num_up
        self._layers = self.num_up + self.num_cell_path_search
        self._stem_cells = self.num_cell - self.num_cell_path_search
        # self.total_num_cell = self._stem_cells + (self.num_up+1) * self.num_cell_path_search

        self._width_mult_list = width_mult_list
        self._prun_modes = prun_modes
        self.prun_mode = None # prun_mode is higher priority than _prun_modes
        self._flops = 0
        self._params = 0

        self.base_weight = loss_weight[0]
        self.style_weight = loss_weight[1]
        self.content_weight = loss_weight[2]
        self.tv_weight = loss_weight[3]
        self.order_constraint_weight = loss_weight[4]

        # if config.distributed:
        #     self.vgg = torch.nn.parallel.DistributedDataParallel(VGGFeature(before_act=before_act).to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        # else:
        self.vgg = torch.nn.DataParallel(VGGFeature(before_act=before_act)).cuda()

        self.quantize = quantize
        self.slimmable = slimmable

        self.nf = nf

        assert config.iv_mode == 'image' #temporary: only image mode
        self.iv_mode = config.iv_mode
        if config.iv_mode == 'image':
            #self.PRIMITIVES_attention = PRIMITIVES_attn_image
            self.PRIMITIVES_attention = config.primitives_attn
            num_frames = 1
        #elif config.iv_mode == 'video':
        #    self.PRIMITIVES_attention = PRIMITIVES_attn_video
        else:
            raise(Exception)

        self.attention = config.attention
        self.num_cell_attn = config.num_cell_attn
        self.op_per_cell_attn = config.op_per_cell_attn
        self.num_levels_attn = config.num_levels_attn

        self.conv_first = Conv(3, self.nf, 3, 1, 1, bias=True)

        # self.op_idx_list = op_idx_list
        # self.quantize_list = quantize_list
        # self.ratio_list = ratio_list

        self.cells = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(self.num_cell-self.num_cell_path_search):
            cell = Cell(nf=self.nf, op_per_cell=self.op_per_cell, width_mult_list=self._width_mult_list)
            if state:
                cell.load_state_dict(state['cells'][i])
            self.cells.append(cell)

        print('msf line 441 model init %s'%(utils.profiling.memory_usage()))

        if config.attention:
            self.cells_attn = nn.ModuleList()
            for i in range(self.num_cell_attn):
                self.cells_attn.append(Cell_attn(
                    num_frames=num_frames,num_channels_in=64,
                    op_per_cell=config.op_per_cell_attn,iv_mode=config.iv_mode,primitives_attn=config.primitives_attn))

        # self.HRconv = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.pixelshuffle = nn.PixelShuffle(4)
        self.conv_last = Conv(self.nf, 3*4*4, 3, 1, 1, bias=True)
                        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.tanh = nn.Tanh()

        self.loss_func = nn.MSELoss() if loss_func == 'MSE' else nn.L1Loss()
        self.num_feature_maps, self.path_index = self.dfs(state)
        self.total_num_cell = len(self.cells)
        self.num_path = len(self.path_index)
        self.arch_params = self._build_arch_parameters()
        print('num_path',self.num_path)
        print('path index:', self.path_index)
        print('num_feature_maps:', self.num_feature_maps)
        # self._reset_arch_parameters()
        ####################################### hyperparameters for sparse group lasso
        self.alpha_sparsity = 0.5
        self.Lambda = 0.001
        self.rad = 0

        if state:
            self.conv_first.load_state_dict(state['conv_first'])
            self.HRconv.load_state_dict(state['HRconv'])
            self.conv_last.load_state_dict(state['conv_last'])
        # print('level2_cell_length:', self.level2_cell_length)
        # print('flops:', self.flops)
        self.base_loss = 0
        self.content_loss = 0
        self.tv_loss = 0
        self.order_loss = 0


    def set_rad(self, rad):
        self.rad = rad

    def get_rad(self):
        return self.rad

    def _update_lambda(self, lambda1_new, alpha_new):
      self.Lambda = lambda1_new
      self.alpha_sparsity = alpha_new


    def _get_lambda(self):
      return self.Lambda, self.alpha

    def sample_prun_ratio(self, mode="arch_ratio"):
        '''
        mode: "min"|"max"|"random"|"arch_ratio"(default)
        '''
        assert mode in ["min", "max", "random", "arch_ratio"]
        if mode == "arch_ratio":
            ratio = self.ratio
            ratio_sampled = []
            for cell_id in range(len(self.cells)):
                ratio_cell = []
                for op_id in range(self.op_per_cell-1):
                    ratio_cell.append(gumbel_softmax(F.log_softmax(ratio[cell_id][op_id], dim=-1), hard=True))
                ratio_sampled.append(ratio_cell)

            return ratio_sampled

        elif mode == "min":
            ratio_sampled = []
            for cell_id in range(len(self.cells)):
                ratio_cell = []
                for op_id in range(self.op_per_cell-1):
                    ratio_cell.append(self._width_mult_list[0])
                ratio_sampled.append(ratio_cell)

            return ratio_sampled

        elif mode == "max":
            ratio_sampled = []
            for cell_id in range(len(self.cells)):
                ratio_cell = []
                for op_id in range(self.op_per_cell-1):
                    ratio_cell.append(self._width_mult_list[-1])
                ratio_sampled.append(ratio_cell)

            return ratio_sampled

        elif mode == "random":
            ratio_sampled = []

            for cell_id in range(len(self.cells)):
                ratio_cell = []
                for op_id in range(self.op_per_cell-1):
                    ratio_cell.append(np.random.choice(self._width_mult_list))
                ratio_sampled.append(ratio_cell)
        
            return ratio_sampled


    def dfs(self, state):
        stack = [(0, self.num_cell-self.num_cell_path_search, [])]   # (level, layer, flops, size)
        num_feature_maps = 0
        path_index = []
        while stack:
            level, layer, index = stack.pop()
            #print(index)
            index.append(num_feature_maps)
            num_feature_maps += 1
            if layer < self.num_cell:
                cell = Cell(nf=self.nf, op_per_cell=self.op_per_cell, width_mult_list=self._width_mult_list)
                if state:
                    cell.load_state_dict(state['cells'][layer])
                stack.append((level, layer+1, index.copy()))
                self.cells.append(cell)
            if level < self.num_up:
                cell = UpSample(self.nf, quantize=self.quantize)
                if state:
                    cell.load_state_dict(state['ups'][level])
                stack.append((level+1, layer, index.copy()))
                self.ups.append(cell)
            if level == self.num_up and layer == self.num_cell:
                path_index.append(index.copy())
        return num_feature_maps, path_index#, flops

    def forward(self, input, Flops=False):
        if self.config.sparse_type == 'sparsemax':
            gamma = sparsemax(self.gamma)
            alpha = sparsemax(self.alpha)
            beta = sparsemax(self.beta)
        elif self.config.sparse_type == 'sparsestmax':
            gamma = sparsestmax(self.gamma, self.rad)
            alpha = sparsestmax(self.alpha, self.rad)
            beta = sparsestmax(self.beta, self.rad)
        elif self.config.sparse_type == 'softmax':
            gamma = F.softmax(self.gamma, dim=-1)
            alpha = F.softmax(self.alpha, dim=-1)
            beta = F.softmax(self.beta, dim=-1)
        else:
            raise NotImplementedError('invalid sparse type')

        if self.config.attn_sparse_type == 'sparsemax':
            alpha_sink_attn = sparsemax(self.alpha_sink_attn)
            alpha_levels_attn = sparsemax(self.alpha_levels_attn)
        elif self.config.attn_sparse_type == 'sparsestmax':
            alpha_sink_attn = sparsestmax(self.alpha_sink_attn, self.rad)
            alpha_levels_attn = sparsestmax(self.alpha_levels_attn, self.rad)
        elif self.config.attn_sparse_type == 'softmax':
            alpha_sink_attn = F.softmax(self.alpha_sink_attn, dim=-1)
            alpha_levels_attn = F.softmax(self.alpha_levels_attn, dim=-1)
            #note softmax: there is a difference between dim=-1 and dim=-2
            #depending on definition, one choice guarantees that a level 1 node to have at least one output
            #other choice guaranttes a level 2 node to have at least one input
        else:
            raise NotImplementedError('invalid attn sparse type')

        if self.prun_mode is not None:
            ratio = self.sample_prun_ratio(mode=self.prun_mode)
        else:
            ratio = self.sample_prun_ratio(mode=self._prun_modes)
        # gamma = [F.softmax(weight, dim=-1) for weight in self.gamma]

        input = self.conv_first(input)
        count_cell = 0
        count_up = 0
        for i in range(self.num_cell-self.num_cell_path_search):
            input = self.cells[count_cell](input, alpha[count_cell], beta[count_cell], ratio[count_cell])
            count_cell += 1
        
        #print('msf line 612 model forward %s'%(utils.profiling.memory_usage()))

        out = 0
        # out_path = 0
        stack = [(input, 0, self.num_cell-self.num_cell_path_search)]
        path_index = 0
        feature_map_index = 0
        i = 0
        while stack:
            #print("msf line 621, feature_map_index=%d"%feature_map_index)
            h, level, layer = stack.pop()
            out += gamma[feature_map_index] * h
            #Each path is associated with a gamma index
            feature_map_index += 1
            if layer < self.num_cell:
                stack.append((self.cells[count_cell](h, alpha[count_cell], beta[count_cell], ratio[count_cell]), level, layer+1))
                count_cell += 1
            if level < self.num_up:
                stack.append((self.ups[count_up](h), level+1, layer))
                count_up += 1
        assert len(self.cells) == count_cell
        assert len(self.ups) == count_up
        assert len(gamma) == feature_map_index

        print("msf line 636, final feature_map_index=%d"%feature_map_index)

        #print("count_cell and count_up:")
        #print(count_cell)
        #print(count_up)

        print('msf line 640 model forward profiling %s'%(utils.profiling.memory_usage()))

        if self.attention:
            supernet_out = out
            for i, cell in enumerate(self.cells_attn):
                out = cell(out, alpha_sink_attn[i], alpha_levels_attn[i])
            out = out + supernet_out

        out = self.pixelshuffle(self.conv_last(out))
        if ENABLE_TANH:
            out = (self.tanh(out)+1)/2
        return out

    
    def forward_flops(self, size, alpha=True, beta=True, ratio=True, gamma=True):
        if self.config.sparse_type == 'sparsemax':
            if gamma:
                gamma = sparsemax(self.gamma)
            else:
                gamma = F.softmax(torch.ones_like(self.gamma).cuda(), dim=-1)
            if alpha:
                alpha = sparsemax(self.alpha)
            else:
                alpha = torch.ones_like(self.alpha).cuda()* 1./len(PRIMITIVES)
            if beta:
                beta = sparsemax(self.beta)
            else:
                beta = torch.ones_like(self.beta).cuda()* 1./2
        elif self.config.sparse_type == 'sparsestmax':
            if gamma:
                gamma = sparsestmax(self.gamma, self.rad)
            else:
                gamma = F.softmax(torch.ones_like(self.gamma).cuda(), dim=-1)
            if alpha:
                alpha = sparsestmax(self.alpha, self.rad)
            else:
                alpha = torch.ones_like(self.alpha).cuda()* 1./len(PRIMITIVES)
            if beta:
                beta = sparsestmax(self.beta, self.rad)
            else:
                beta = torch.ones_like(self.beta).cuda()* 1./2
        elif self.config.sparse_type == 'softmax':
            if gamma:
                gamma = F.softmax(self.gamma, dim=-1)
            else:
                gamma = F.softmax(torch.ones_like(self.gamma).cuda(), dim=-1)
            if alpha:
                alpha = F.softmax(self.alpha, dim=-1)
            else:
                alpha = torch.ones_like(self.alpha).cuda()* 1./len(PRIMITIVES)
            if beta:
                beta = F.softmax(self.beta, dim=-1)
            else:
                beta = torch.ones_like(self.beta).cuda()* 1./2
        else:
            raise NotImplementedError('invalid sparse type')

        if ratio:
            if self.prun_mode is not None:
                ratio = self.sample_prun_ratio(mode=self.prun_mode)
            else:
                ratio = self.sample_prun_ratio(mode=self._prun_modes)
        else:
            ratio = self.sample_prun_ratio(mode='max')

        # size = (3, 256, 256)
        flops, size = self.conv_first.forward_flops(size)
        flops_total = 0
        count_cell = 0
        count_up = 0
        path_index = 0
        feature_map_index = 0
        for _ in range(self.num_cell-self.num_cell_path_search):
            temp_flop, size = self.cells[count_cell].forward_flops(size, alpha[count_cell], beta[count_cell], ratio[count_cell])
            count_cell += 1
            flops += temp_flop
        stack = [(0, self.num_cell-self.num_cell_path_search, flops, size)]
        while stack:
            level, layer, flops, size = stack.pop()
            flops_total += gamma[feature_map_index] * flops
            feature_map_index += 1
            if layer < self.num_cell:
                temp_flop, temp_size = self.cells[count_cell].forward_flops(size, alpha[count_cell], beta[count_cell], ratio[count_cell])
                stack.append((level, layer+1, temp_flop+flops, temp_size))
                count_cell += 1
            if level < self.num_up:
                temp_flop, temp_size = self.ups[count_up].forward_flops(size)
                stack.append((level+1, layer, temp_flop+flops, temp_size))
                count_up += 1
        assert feature_map_index == len(gamma), str(feature_map_index)
        return flops_total


    def _criterion(self, y_hat, x):
        self.base_loss = self.base_weight * self.loss_func(y_hat, x)

        #this is actually perceptual loss?
        y_c_features = self.vgg(x)#.to(self.device)
        y_hat_features = self.vgg(y_hat)#.to(self.device)
        self.content_loss = self.content_weight * self.loss_func(y_c_features, y_hat_features)
        
        #total variation loss (see AGD paper)
        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        self.tv_loss = self.tv_weight * (diff_i + diff_j)
        
        #################################################################### order loss
        order_loss = 0
        start = 0
        for index in self.path_index:
            weight = self.gamma[index]
            loss = torch.sum(weight[1:]-weight[0:len(weight)-1])
            order_loss += loss
        self.order_loss = order_loss * self.order_constraint_weight

        total_loss = self.base_loss + self.content_loss + self.tv_loss + self.order_loss

        return total_loss


    def _loss(self, input, target, pretrain=False):
        # return loss
        loss = 0
        if pretrain is not True:
            # "random width": sampled by gambel softmax
            self.prun_mode = None
            # start = time.time()
            logit = self(input)
            # forward_end = time.time()
            loss = loss + self._criterion(logit, target)
            # loss_end = time.time()

            # print('forward:', forward_end-start)
            # print('loss:', loss_end-forward_end)


        if len(self._width_mult_list) > 1:
            self.prun_mode = "max"
            # start = time.time()
            logit = self(input)
            # forward_end = time.time()

            # print('logits:', logit.size())
            # print('target:', target.size())
            loss = loss + self._criterion(logit, target)
            # loss_end = time.time()
            # print('forward:', forward_end-start)
            # print('loss:', loss_end-forward_end)

            self.prun_mode = "min"
            logit = self(input)
            # print('logits:', logit.size())
            # print('target:', target.size())
            loss = loss + self._criterion(logit, target)

            if pretrain == True:
                self.prun_mode = "random"
                logit = self(input)
                # print('logits:', logit.size())
                # print('target:', target.size())
                loss = loss + self._criterion(logit, target)

                self.prun_mode = "random"
                logit = self(input)
                # print('logits:', logit.size())
                # print('target:', target.size())
                loss = loss + self._criterion(logit, target)

        elif pretrain == True and len(self._width_mult_list) == 1:
            self.prun_mode = "max"
            logit = self(input)
            loss = loss + self._criterion(logit, target)

        return loss


    def _build_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        num_ops_attn = len(self.PRIMITIVES_attention)

        # setattr(self, 'alpha', nn.Parameter(Variable(1e-3*torch.ones(self.total_num_cell, self.op_per_cell, num_ops).cuda(), requires_grad=True)))
        # setattr(self, 'beta', nn.Parameter(Variable(1e-3*torch.ones(self.total_num_cell, self.op_per_cell, 2).cuda(), requires_grad=True)))
        # setattr(self, 'gamma', nn.Parameter(Variable(1e-3*torch.ones(self.num_up+1, self.num_cell_path_search+1, 2).cuda(), requires_grad=True)))
        # setattr(self, 'gamma_group', nn.Parameter(Variable(1e-3*torch.ones(self.num_paths).cuda(), requires_grad=True)))
        # setattr(self, 'gamma_element', nn.Parameter(Variable(1e-3*torch.ones(self.num_paths, self.length).cuda(), requires_grad=True)))
        # self.gamma_group = nn.Parameter(Variable(1e-3*torch.ones(self.num_paths).cuda(), requires_grad=True))

        self.gamma = Variable(0.3*torch.ones(self.num_feature_maps).cuda(), requires_grad=True)
        self.alpha_sink_attn = Variable(0.3 * torch.ones(self.num_cell_attn, self.op_per_cell_attn, self.num_levels_attn*num_ops_attn).cuda(), requires_grad=True)
        self.alpha_levels_attn = Variable(0.3 * torch.ones(self.num_cell_attn, self.op_per_cell_attn, self.num_levels_attn-1, num_ops_attn, num_ops_attn).cuda(), requires_grad=True)

        if self.config.random_arch_init:
            self.alpha = Variable(1e-3*torch.randn(self.total_num_cell, self.op_per_cell, num_ops).cuda(), requires_grad=True)
            self.beta = Variable(1e-3*torch.randn(self.total_num_cell, self.op_per_cell, 2).cuda(), requires_grad=True)
            #self.alpha_sink_attn = Variable(1e-3 * torch.randn(self.num_cell_attn, self.op_per_cell_attn, self.num_levels_attn*num_ops_attn).cuda(), requires_grad=True)
            #self.alpha_levels_attn = Variable(1e-3 * torch.randn(self.num_cell_attn, self.op_per_cell_attn, self.num_levels_attn-1, num_ops_attn, num_ops_attn).cuda(), requires_grad=True)
        else:
            self.alpha = Variable(1e-3*torch.ones(self.total_num_cell, self.op_per_cell, num_ops).cuda(), requires_grad=True)
            self.beta = Variable(1e-3*torch.ones(self.total_num_cell, self.op_per_cell, 2).cuda(), requires_grad=True)
            #self.alpha_sink_attn = Variable(1e-3 * torch.ones(self.num_cell_attn, self.op_per_cell_attn, self.num_levels_attn*num_ops_attn).cuda(), requires_grad=True)
            #self.alpha_levels_attn = Variable(1e-3 * torch.ones(self.num_cell_attn, self.op_per_cell_attn, self.num_levels_attn-1, num_ops_attn, num_ops_attn).cuda(), requires_grad=True)

        if self._prun_modes == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1

        self.ratio = Variable(1e-3*torch.randn(self.total_num_cell, self.op_per_cell-1, num_widths).cuda(), requires_grad=True)

        return {'alpha': self.alpha, 'beta':self.beta, 'ratio':self.ratio, 'gamma':self.gamma,
                'alpha_sink_attn': self.alpha_sink_attn, 'alpha_levels_attn': self.alpha_levels_attn}

        # def _reset_arch_parameters(self):

        # num_ops = len(PRIMITIVES)
        # if self._prun_modes == 'arch_ratio':
        #     # prunning ratio
        #     num_widths = len(self._width_mult_list)
        # else:
        #     num_widths = 1

        # getattr(self, "gamma_group").data = Variable(1e-3*torch.ones(self.num_paths).cuda(), requires_grad=True)
        # getattr(self, "gamma_element").data = Variable(1e-3*torch.ones(self.num_paths, self.length).cuda(), requires_grad=True)


        # getattr(self, "alpha").data = Variable(1e-3*torch.ones(self.total_num_cell, self.op_per_cell, num_ops).cuda(), requires_grad=True)
        # getattr(self, "beta").data = Variable(1e-3*torch.ones(self.total_num_cell, self.op_per_cell, 2).cuda(), requires_grad=True)
        # getattr(self, "gamma").data = Variable(1e-3*torch.ones(self.num_up+1, self.num_cell_path_search+1, 2).cuda(), requires_grad=True)
        # getattr(self, "ratio").data = Variable(1e-3*torch.ones(self.total_num_cell, self.op_per_cell-1, num_widths).cuda(), requires_grad=True)

        
