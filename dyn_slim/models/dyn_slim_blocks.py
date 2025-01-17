import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyn_slim.models.dyn_slim_ops import DSpwConv2d, DSdwConv2d, DSBatchNorm2d, DSAvgPool2d, DSAdaptiveAvgPool2d
from timm.models.layers import sigmoid
from math import ceil

import time

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DSInvertedResidual(nn.Module):

    def __init__(self, in_channels_list, out_channels_list, kernel_size, layer_num, stride=1, dilation=1,
                 act_layer=nn.ReLU, noskip=False, exp_ratio=6.0, se_ratio=0.25,
                 norm_layer=DSBatchNorm2d, norm_kwargs=None, conv_kwargs=None,
                 drop_path_rate=0., bias=False, has_gate=False):
        super(DSInvertedResidual, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.layer_num = layer_num
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_channels_list = [make_divisible(inc * exp_ratio) for inc in in_channels_list]
        self.has_residual = not noskip
        self.drop_path_rate = drop_path_rate
        self.has_gate = has_gate
        self.downsample = None
        if self.has_residual:
            if in_channels_list[-1] != out_channels_list[-1] or stride == 2:
                downsample_layers = []
                if stride == 2:
                    downsample_layers += [DSAvgPool2d(2, 2, ceil_mode=True,
                                                       count_include_pad=False, channel_list=in_channels_list)]
                if in_channels_list[-1] != out_channels_list[-1]:
                    downsample_layers += [DSpwConv2d(in_channels_list,
                                                     out_channels_list,
                                                     bias=bias)]
                self.downsample = nn.Sequential(*downsample_layers)
        # Point-wise expansion
        self.conv_pw = DSpwConv2d(in_channels_list, mid_channels_list, bias=bias)
        self.bn1 = norm_layer(mid_channels_list, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = DSdwConv2d(mid_channels_list,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  dilation=dilation,
                                  bias=bias)
        self.bn2 = norm_layer(mid_channels_list, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Channel attention and gating
        self.gate = MultiHeadGate(mid_channels_list,
                                se_ratio=se_ratio,
                                channel_gate_num=18 if has_gate else 0)

        # Point-wise linear projection
        self.conv_pwl = DSpwConv2d(mid_channels_list, out_channels_list, bias=bias)
        self.bn3 = norm_layer(out_channels_list, **norm_kwargs)
        self.channel_choice = -1
        self.mode = 'largest'
        self.next_channel_choice = None
        self.last_feature = None
        self.random_choice = 0
        self.init_residual_norm()
        self.latency =None

    def init_residual_norm(self, level='block'):
        if self.has_residual:
            if level == 'block':
                self.bn3.set_zero_weight()
            elif level == 'channel':
                self.bn1.set_zero_weight()
                self.bn3.set_zero_weight()

    def feature_module(self, location):
        if location == 'post_exp':
            return 'act1'
        return 'conv_pwl'

    def feature_channels(self, location):
        if location == 'post_exp':
            return self.conv_pw.out_channels
        # location == 'pre_pw'
        return self.conv_pwl.in_channels

    def get_last_stage_distill_feature(self):
        return self.last_feature

    def forward(self, x):
        START_TIME=time.time()
        self._set_gate()

        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Channel attention and gating
        x = self.gate(x)
        if self.has_gate:
            self.prev_channel_choice = self.channel_choice
            self.channel_choice = self._new_gate()
            self._set_gate(set_pwl=True)
        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            if self.drop_path_rate > 0. and self.mode == 'largest':
                # Only apply drop_path on largest model
                x = drop_path(x, self.training, self.drop_path_rate)
            x += residual
        self.latency=(time.time()-START_TIME)
        return x

    def _set_gate(self, set_pwl=False):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', self.channel_choice)
        if set_pwl:
            self.conv_pwl.prev_channel_choice = self.prev_channel_choice
            if self.downsample is not None:
                for n, m in self.downsample.named_modules():
                    set_exist_attr(m, 'prev_channel_choice', self.prev_channel_choice)

    def _new_gate(self):
        if self.mode == 'largest':
            return -1
        elif self.mode == 'smallest':
            return 0
        elif self.mode == 'uniform' or self.mode == "choice" or self.mode == "multi-choice":
            return self.random_choice
        elif self.mode == 'random':
            return random.randint(0, len(self.out_channels_list) - 1)
        elif self.mode == 'dynamic':
            if self.gate.has_gate:
                return self.gate.get_gate()
            else:
                return 0

    def get_gate(self):
        return self.channel_choice

    def get_latency(self):
        return self.latency

class DSDepthwiseSeparable(nn.Module):

    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, layer_num, stride=1, dilation=1,
                 act_layer=nn.ReLU, noskip=False, se_ratio=0.25,
                 norm_layer=DSBatchNorm2d, norm_kwargs=None, conv_kwargs=None,
                 drop_path_rate=0., bias=False, has_gate=False):
        super(DSDepthwiseSeparable, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.layer_num = layer_num
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        self.has_residual = not noskip
        self.drop_path_rate = drop_path_rate
        self.has_gate = has_gate
        self.downsample = None
        if self.has_residual:
            if in_channels_list[-1] != out_channels_list[-1] or stride == 2:
                downsample_layers = []
                if stride == 2:
                    downsample_layers += [DSAvgPool2d(2, 2, ceil_mode=True,
                                                       count_include_pad=False, channel_list=in_channels_list)]
                if in_channels_list[-1] != out_channels_list[-1]:
                    downsample_layers += [DSpwConv2d(in_channels_list,
                                                     out_channels_list,
                                                     bias=bias)]
                self.downsample = nn.Sequential(*downsample_layers)
        # Depth-wise convolution
        self.conv_dw = DSdwConv2d(in_channels_list,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  dilation=dilation,
                                  bias=bias)
        self.bn1 = norm_layer(in_channels_list, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Channel attention and gating
        self.gate = MultiHeadGate(in_channels_list,
                                se_ratio=se_ratio,
                                channel_gate_num=18 if has_gate else 0)

        # Point-wise convolution
        self.conv_pw = DSpwConv2d(in_channels_list, out_channels_list, bias=bias)
        self.bn2 = norm_layer(out_channels_list, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.channel_choice = -1
        self.mode = 'largest'
        self.next_channel_choice = None
        self.random_choice = 0
        self.init_residual_norm()
        self.latency =None

    def init_residual_norm(self, level='block'):
        if self.has_residual:
            if level == 'block':
                self.bn2.set_zero_weight()
            elif level == 'channel':
                self.bn1.set_zero_weight()
                self.bn2.set_zero_weight()

    def forward(self, x):
        START_TIME=time.time()
        self._set_gate()
        residual = x

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Channel attention and gating
        x = self.gate(x)
        if self.has_gate:
            self.prev_channel_choice = self.channel_choice
            self.channel_choice = self._new_gate()
            self._set_gate(set_pw=True)
        # Point-wise convolution
        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            if self.drop_path_rate > 0. and self.mode == 'largest':
                # Only apply drop_path on largest model
                x = drop_path(x, self.training, self.drop_path_rate)
            x += residual
        self.latency=(time.time()-START_TIME)
        return x

    def _set_gate(self, set_pw=False):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', self.channel_choice)
        if set_pw:
            self.conv_pw.prev_channel_choice = self.prev_channel_choice
            if self.downsample is not None:
                for n, m in self.downsample.named_modules():
                    set_exist_attr(m, 'prev_channel_choice', self.prev_channel_choice)

    def _new_gate(self):
        if self.mode == 'largest':
            return -1
        elif self.mode == 'smallest':
            return 0
        elif self.mode == 'uniform' or self.mode == 'choice' or self.mode == "multi-choice":
            return self.random_choice
        elif self.mode == 'random':
            return random.randint(0, len(self.out_channels_list) - 1)
        elif self.mode == 'dynamic':
            if self.gate.has_gate:
                return self.gate.get_gate()
            else:
                return 0

    def get_gate(self):
        return self.channel_choice

    def get_latency(self):
        return self.latency


class MultiHeadGate(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU,
                 attn_act_fn=sigmoid, divisor=1, channel_gate_num=None, gate_num_features=1024):
        super(MultiHeadGate, self).__init__()
        self.attn_act_fn = attn_act_fn
        self.channel_gate_num = channel_gate_num
        reduced_chs = make_divisible((reduced_base_chs or in_chs[-1]) * se_ratio, divisor)
        self.avg_pool = DSAdaptiveAvgPool2d(1, channel_list=in_chs)
        self.conv_reduce = DSpwConv2d(in_chs, [reduced_chs], bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = DSpwConv2d([reduced_chs], in_chs, bias=True)

        self.has_gate = False
        if channel_gate_num > 1:
            self.has_gate = True
            # self.gate = nn.Sequential(DSpwConv2d([reduced_chs], [gate_num_features], bias=True),
            #                           act_layer(inplace=True),
            #                           nn.Dropout2d(p=0.2),
            #                           DSpwConv2d([gate_num_features], [channel_gate_num], bias=True))
            self.gate = nn.Sequential(DSpwConv2d([reduced_chs], [channel_gate_num], bias=False))

        self.mode = 'largest'
        self.keep_gate, self.print_gate, self.print_idx = None, None, None
        self.channel_choice = None
        self.initialized = False
        if self.attn_act_fn == 'tanh':
            nn.init.zeros_(self.conv_expand.weight)
            nn.init.zeros_(self.conv_expand.bias)

        #addition for system based slimming
        if self.has_gate:
            #will manually set to true/false
            self.control = False
            self.control_output = None #will use this for control loss

            self.system_vars = 9
            self.control_in_complexity = channel_gate_num + (3*self.system_vars)
            self.control_out_complexity = 256
            self.control_input = 1 + (3 * ceil(float(self.system_vars)/float(self.channel_gate_num)))

            #just initializing these
            self.system_vector = torch.zeros(1, self.system_vars, 1) 
            self.control_vector = torch.zeros(1, self.system_vars, 1)
            self.target_vector = torch.zeros(1, self.system_vars, 1)
            
            self.control_layer1 = torch.nn.Linear(self.control_in_complexity, self.control_out_complexity)
            self.control_layer2 = torch.nn.Linear(self.control_out_complexity, channel_gate_num)

    def forward(self, x):
        x_pool = self.avg_pool(x)
        x_reduced = self.conv_reduce(x_pool)
        x_reduced = self.act1(x_reduced)
        attn = self.conv_expand(x_reduced)
        if self.attn_act_fn == 'tanh':
            attn = (1 + attn.tanh())
        else:
            attn = self.attn_act_fn(attn)

        x = x * attn

        # print(x.shape)
        # print(x_pool.shape)
        # print(x_reduced.shape)

        if self.mode == 'dynamic' and self.has_gate:
            channel_choice = self.gate(x_reduced).squeeze(-1).squeeze(-1)
            #print(channel_choice.shape)
 
            if self.control:
                channel_choice = self.control_pass(channel_choice)

            self.keep_gate, self.print_gate, self.print_idx = better_gumbel(channel_choice) 
                                                              #gumbel_softmax(channel_choice, dim=1, training=self.training)
            
            self.print_gate = torch.tensor([[1,0,0,0,
                                             0,0,0,0,
                                             0,0,0,0,
                                             0,0,0,0,
                                             0,0]]).float().cuda()
            self.print_idx = torch.tensor([[0]]).cuda()

            self.channel_choice = self.print_gate, self.print_idx

        else:
            self.channel_choice = None

        # if self.channel_choice == None:
        #     print('None', x.shape)
        # else:
        #     print(self.channel_choice[0].shape, self.channel_choice[1].shape, x.shape)

        return x

    def get_gate(self):
        return self.channel_choice

    def control_pass(self, pre_gumbel):
        #need to change this so we concat on the second version of feature_vector
        #pre_gumbel.shape = (batch, feature vector, 1)

        #need to actually grab system variables, this is temporary
        self.system_vector = torch.ones(1, self.system_vars, 1) # these 3 vectors need to change shape
        self.control_vector = torch.ones(1, self.system_vars, 1)
        self.target_vector = torch.ones(1, self.system_vars, 1)

        #prep info for concat
        pre_gumbel = pre_gumbel.view(-1,self.channel_gate_num,1)
        self.system_vector = self.system_vector.repeat(pre_gumbel.shape[0],1,1)
        self.control_vector = self.control_vector.repeat(pre_gumbel.shape[0],1,1)
        self.target_vector = self.target_vector.repeat(pre_gumbel.shape[0],1,1)

        #combine data and pass through layers
        pre_gumbel = torch.cat((pre_gumbel, self.system_vars, self.control_vector, self.target_vector), 1)

        #TODO:assert pre_gumble shape

        pre_gumbel = self.layer1(pre_gumbel)
        pre_gumbel = torch.nn.LeakyReLU(pre_gumbel) #simple relu for now
        pre_gumbel = self.layer2(pre_gumbel)
        #pre_gumbel = torch.nn.LeakyReLU(pre_gumbel)

        #pre_gumbel = pre_gumbel[0:reset_shape].view(-1, self.channel_gate_num)

        return pre_gumbel


def better_gumbel(logits, lam = 1, dim=1):
    noise = torch.rand(logits.shape).cuda()
    noise = -torch.log(-torch.log(noise + 1e-20)+1e-20)

    out = logits + noise
    out_soft = torch.nn.functional.softmax((out/lam), dim)

    shape = out_soft.size()
    _, index = out_soft.max(dim=-1)
    out_hard = torch.zeros_like(out_soft).view(-1, shape[-1]).cuda()
    out_hard.scatter_(1, index.view(-1, 1), 1)
    out_hard = out_hard.view(*shape)

    out_hard = (out_hard - out_soft).detach() + out_soft
    return out_soft, out_hard, index 

def gumbel_softmax(logits, tau=1, hard=False, dim=1, training=True):
    """ See `torch.nn.functional.gumbel_softmax()` """
    # if training:
    # gumbels = -torch.empty_like(logits,
    #                             memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    # gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # # else:
    # #     gumbels = logits
    # y_soft = gumbels.softmax(dim)

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    with torch.no_grad():
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        #  **test**
        # index = 0
        # y_hard = torch.Tensor([1, 0, 0, 0]).repeat(logits.shape[0], 1).cuda()
    ret = y_hard - y_soft.detach() + y_soft
    return y_soft, ret, index


def drop_path(inputs, training=False, drop_path_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_path_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)
