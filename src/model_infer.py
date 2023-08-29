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
ENABLE_TANH = False

def make_divisible(v, divisor=8, min_value=3):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
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


def update_mask_levels(mask_levels,alpha_levels,level_idx,op_idx,levels_t):
    assert level_idx>0
    dependecy_op_idx_list = torch.topk(alpha_levels[level_idx-1,:,op_idx], k=levels_t)[1]
    for dep_idx in dependecy_op_idx_list:
        mask_levels[level_idx-1][dep_idx][op_idx] = 1
        if level_idx-1>0: #will not be called for typical 2-level design as in AttnAGD
            mask_levels = update_mask_levels(mask_levels,alpha_levels,level_idx-1,dep_idx,levels_t)
    return mask_levels


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, op_idx, quantize, stride=1):
        super(MixedOp, self).__init__()
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, stride, slimmable=False, width_mult_list=[1.])
        self.quantize = quantize

    def forward(self, x):
        return self._op(x, quantize=self.quantize)

    def forward_latency(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        latency, size_out = self._op.forward_latency(size)
        return latency, size_out

    def forward_flops(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        flops, size_out = self._op.forward_flops(size, quantize=self.quantize)
        return flops, size_out


class MixedOp_attn(nn.Module):
    def __init__(self,alpha_sink,alpha_levels,sink_k,levels_t,primitives_attn,iv_mode,num_frames,num_channels_in):
        super(MixedOp_attn, self).__init__()
        print('initialize MixedOp_attn')

        self.alpha_sink = alpha_sink
        self.alpha_levels = alpha_levels

        self.iv_mode = iv_mode
        #if iv_mode=='image':
        #    PRIMITIVES_attn = PRIMITIVES_attn_image
        #elif iv_mode=='video':
        #    PRIMITIVES_attn = PRIMITIVES_attn_video
        #else:
        #    raise Exception('invalid mode')
        PRIMITIVES_attn = primitives_attn
        print(PRIMITIVES_attn)

        num_levels = 2 #temporary hard coded
        self.num_levels = num_levels
        num_ops_attn = len(PRIMITIVES_attn)
        self.num_ops_attn = len(PRIMITIVES_attn)
        assert int(self.num_levels*self.num_ops_attn) == alpha_sink.shape[-1]
        assert int((self.num_levels-1)*(self.num_ops_attn**2)) == alpha_levels.shape[0]*alpha_levels.shape[1]*alpha_levels.shape[2]

        self._ops_attn = nn.ModuleList()
        for i, op_attn_name in enumerate(PRIMITIVES_attn):
            self._ops_attn.append(OPS_Attention[op_attn_name](num_channels_in,num_frames))

        self.mask_sink = torch.zeros(self.num_levels,self.num_ops_attn,dtype=torch.int32)
        self.mask_levels = torch.zeros(self.num_levels-1,self.num_ops_attn,self.num_ops_attn,dtype=torch.int32)

        assert int(num_levels*num_ops_attn) == alpha_sink.shape[-1]
        assert int((num_levels-1)*(num_ops_attn**2)) == alpha_levels.shape[0]*alpha_levels.shape[1]*alpha_levels.shape[2]

        print("sink selected")
        sink_selected= torch.topk(alpha_sink, k=sink_k, dim=-1)[1]
        print(sink_selected.shape)
        print(sink_selected)

        for i in range(len(sink_selected)):
            level_idx = int(sink_selected[i]/self.num_ops_attn)
            op_idx = int(sink_selected[i]%self.num_ops_attn)
            self.mask_sink[level_idx][op_idx] = 1
            if level_idx>0:
                self.mask_levels = update_mask_levels(self.mask_levels,alpha_levels,level_idx,op_idx,levels_t)

        print("self.mask_levels: %s"%str(self.mask_levels))
        print("self.mask_sink: %s"%str(self.mask_sink))

        mask_source_levels_contrib = torch.clamp(torch.sum(self.mask_levels[0],-1),min=0,max=1)
        mask_source_sink_contrib = self.mask_sink[0]
        self.mask_source = torch.clamp(mask_source_levels_contrib+mask_source_sink_contrib,min=0,max=1)
        print("self.mask_source: %s"%str(self.mask_source))

    def forward(self,x_source):
        x_sink = 0
        if self.iv_mode == 'image':
            B,C,H,W=x_source.shape
            x_source = x_source.view(B,1,C,H,W)

        l1_out = [0]*self.num_ops_attn
        for i in range(self.num_ops_attn):
            if self.mask_source[i]==1:
                l1_out[i] = self._ops_attn[i](x_source)
            if self.mask_sink[0][i]==1:
                x_sink += l1_out[i]

        l2_in = [0]*self.num_ops_attn
        for i in range(self.num_ops_attn):
            for j in range(self.num_ops_attn):
                if self.mask_levels[0][i][j]==1:
                    l2_in[j] += l1_out[i]

        for i in range(self.num_ops_attn):
            if self.mask_sink[1][i]==1:
                x_sink += self._ops_attn[i](l2_in[i])

        if self.iv_mode == 'image':
            x_sink = x_sink.view(B,C,H,W)
        return(x_sink)


class Cell(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, op_idx_list, quantize_list, ratio_list, nf=64, op_per_cell=5, width_mult_list=[1.]):
        super(Cell, self).__init__()
        self.nf = nf
        self.op_per_cell = op_per_cell
        self._width_mult_list = width_mult_list
        self.ops = nn.ModuleList()
        for i in range(op_per_cell):
            if i == 0:
                #print(ratio_list[i])
                #print(op_idx_list[i])
                #print(quantize_list[i])
                #print(width_mult_list[0])
                #print(width_mult_list[ratio_list[i]])
                self.ops.append(MixedOp(self.nf, make_divisible(self.nf * width_mult_list[ratio_list[i]]), op_idx_list[i], quantize_list[i]))
            elif i == op_per_cell - 1:
                self.ops.append(MixedOp(make_divisible(self.nf * width_mult_list[ratio_list[i-1]]), self.nf, op_idx_list[i], quantize_list[i]))
            else:
                self.ops.append(MixedOp(make_divisible(self.nf * width_mult_list[ratio_list[i-1]]), make_divisible(self.nf * width_mult_list[ratio_list[i]]), op_idx_list[i], quantize_list[i]))

    def forward(self, x):
        out = x
        for op in self.ops:
            out = op(out)
        return out*0.2 + x

    def forward_flops(self, size):
        flops_total = []
        for i, op in enumerate(self.ops):
            flops, size = op.forward_flops(size)
            flops_total.append(flops)
        return sum(flops_total), size


class Cell_attn(nn.Module):
    def __init__(self, alpha_sink,alpha_levels,sink_k,levels_t,primitives_attn,
            iv_mode,num_frames,num_channels_in):
        super(Cell_attn, self).__init__()
        assert alpha_sink.shape[0] == alpha_levels.shape[0]
        op_per_cell = alpha_sink.shape[0]
        self.ops = nn.ModuleList()
        for i in range(op_per_cell):
            self.ops.append(
                MixedOp_attn(alpha_sink[i],alpha_levels[i],sink_k,levels_t,primitives_attn,
                    iv_mode,num_frames,num_channels_in))

    def forward(self, x):
        out = x
        for op in self.ops:
            out = op(out)
        return out + x


class UpSample_A(nn.Module):
    def __init__(self, nf=64, quantize=False):
        super(UpSample_A, self).__init__()
        self.nf = nf
        self.upconv = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        output = self.lrelu(self.upconv(F.interpolate(x, scale_factor=2, mode='nearest')))
        return output
        
    def forward_flops(self, size):
        size = (size[0], size[1]*2, size[2]*2)
        flops, size = self.upconv.forward_flops(size)
        return flops, size
        

class UpSample_B(nn.Module):
    def __init__(self, nf=64, quantize=False):
        super(UpSample_B, self).__init__()
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


class NAS_GAN_Infer(nn.Module):
    def __init__(self, op_idx_list, quantize_list, ratio_list, path,
                    num_cell=5, op_per_cell=5, num_up=2, num_cell_path_search=3, width_mult_list=[1.,], loss_weight = [1e0, 1e5, 1e0, 1e-7], 
                    loss_func='MSE', before_act=True, quantize=False, nf=64,
                    iv_mode='image',attention=False,alpha_sink_attn=None,alpha_levels_attn=None,sink_k=None,levels_t=None,
                    primitives_attn=None,trilevelnas_x="B"):

        super(NAS_GAN_Infer, self).__init__()

        self.trilevelnas_x = trilevelnas_x

        self.num_cell = num_cell
        self.op_per_cell = op_per_cell
        self.num_up = num_up
        self.num_cell_path_search = num_cell_path_search

        self._layers = num_cell * op_per_cell

        self._width_mult_list = width_mult_list
        self._flops = 0
        self._params = 0

        self.base_weight = loss_weight[0]
        self.style_weight = loss_weight[1]
        self.content_weight = loss_weight[2]
        self.tv_weight = loss_weight[3]

        self.vgg = torch.nn.DataParallel(VGGFeature(before_act=before_act)).cuda()
        self.nf = nf
        self.conv_first = Conv(3, self.nf, 3, 1, 1, bias=True)
        self.ups = []
        for _ in range(self.num_up):
            if self.trilevelnas_x == "A":
                self.ups.append(UpSample_A(self.nf, quantize=quantize))
            elif self.trilevelnas_x == "B":
                self.ups.append(UpSample_B(self.nf, quantize=quantize))

        if quantize == 'search':
            quantize_list = quantize_list
        elif quantize:
            quantize_list = [ [True for m in range(op_per_cell)] for n in range(num_cell)]      
        else:
            quantize_list = [ [False for m in range(op_per_cell)] for n in range(num_cell)]  

        # ratio_list = F.softmax(ratio_actual, dim=-1).argmax(-1)
        
        self.cells = nn.ModuleList()
        count_cell = 0
        count_up = 0
        for action in path:
            if action == 0:
                self.cells.append(Cell(op_idx_list[count_cell], quantize_list[count_cell], ratio_list[count_cell], nf=self.nf, op_per_cell=op_per_cell, width_mult_list=width_mult_list))
                count_cell += 1
                print('action:', 0)
            else:
                self.cells.append(self.ups[count_up])
                count_up += 1
                print('action:', 1)

        assert iv_mode == 'image' #temporary: only image mode
        self.iv_mode = iv_mode
        if iv_mode == 'image':
            num_frames = 1

        self.attention = attention
        if attention:
            alpha_attn_sink = alpha_sink_attn #temporary
            alpha_attn_levels = alpha_levels_attn #temporary
            self.alpha_attn_sink = alpha_attn_sink
            self.alpha_attn_levels = alpha_attn_levels
            self.cells_attn = nn.ModuleList()
            assert alpha_attn_sink.shape[0] == alpha_attn_levels.shape[0]
            for i in range(alpha_attn_sink.shape[0]):
                self.cells_attn.append(Cell_attn(
                    alpha_sink=alpha_attn_sink[i],alpha_levels=alpha_attn_levels[i],
                    sink_k=sink_k,levels_t=levels_t,primitives_attn=primitives_attn,
                    iv_mode=iv_mode,num_frames=num_frames,num_channels_in=64))

        # self.trunk_conv = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        # self.upconv1 = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        # self.upconv2 = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.pixelshuffle = nn.PixelShuffle(4)
        
        if self.trilevelnas_x=="A":
            self.conv_last = Conv(self.nf, 3, 3, 1, 1, bias=True)
        elif self.trilevelnas_x=="B":
            self.conv_last = Conv(self.nf, 3*4*4, 3, 1, 1, bias=True)
                        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.tanh = nn.Tanh()

        self.loss_func = nn.MSELoss() if loss_func == 'MSE' else nn.L1Loss()


    def forward(self, input):
        out = orig = self.conv_first(input)

        for i, cell in enumerate(self.cells):
            out = cell(out)

        if self.attention:
            supernet_out = out
            for i, cell in enumerate(self.cells_attn):
                    out = cell(out)
            out = out + supernet_out

        if self.trilevelnas_x=="A":
            out = self.conv_last(out)
        elif self.trilevelnas_x=="B":
            out = self.pixelshuffle(self.conv_last(out))

        if ENABLE_TANH:
            out = (self.tanh(out)+1)/2

        return out
        ###################################

    #def forward_flops(self, size):
    #    flops_total = []

    #    flops, size = self.conv_first.forward_flops(size)
    #    flops_total.append(flops)

    #    for i, cell in enumerate(self.cells):
    #        flops, size = cell.forward_flops(size)
    #        flops_total.append(flops)

    #    flops, size = self.conv_last.forward_flops(size)
    #    flops_total.append(flops)

    #    return sum(flops_total)

        ###################################

    def gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    # def _criterion(self, y_hat, x):
    #     base_loss = self.base_weight * nn.L1Loss()(y_hat, x)

    #     y_c_features = self.vgg(x)
    #     y_hat_features = self.vgg(y_hat)

    #     y_hat_gram = [self.gram(fmap) for fmap in y_hat_features]
    #     x_gram = [self.gram(fmap) for fmap in y_c_features]

    #     style_loss = 0
    #     for j in range(4):
    #         style_loss += self.style_weight * nn.functional.mse_loss(y_hat_gram[j], x_gram[j])

    #     recon = y_c_features[1]
    #     recon_hat = y_hat_features[1]
    #     content_loss = self.content_weight * nn.L1Loss()(recon_hat, recon)

    #     diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
    #     diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
    #     tv_loss = self.tv_weight * (diff_i + diff_j)

    #     total_loss = base_loss + style_loss + content_loss + tv_loss

    #     # print(style_loss.data, content_loss.data, tv_loss.data)

    #     return total_loss

    def _criterion(self, y_hat, x):
        base_loss = self.base_weight * self.loss_func(y_hat, x)

        y_c_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)

        content_loss = self.content_weight * self.loss_func(y_c_features, y_hat_features)

        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        tv_loss = self.tv_weight * (diff_i + diff_j)

        total_loss = base_loss + content_loss + tv_loss

        return total_loss


    def _loss(self, input, target):
        logit = self(input)
        loss = self._criterion(logit, target)

        return loss
