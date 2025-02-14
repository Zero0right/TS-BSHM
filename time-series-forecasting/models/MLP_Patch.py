import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import math
import numpy as np

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class MonteCarloDropout(nn.Dropout):
    # We need to init it to False as some models may start by
    # a validation round, in which case MC dropout is disabled.
    mc_dropout_enabled: bool = False

    def train(self, mode: bool = True):
        # NOTE: we could use the line below if self.mc_dropout_rate represented
        # a rate to be applied at inference time, and self.applied_rate the
        # actual rate to be used in self.forward(). However, the original paper
        # considers the same rate for training and inference; we also stick to this.

        # self.applied_rate = self.p if mode else self.mc_dropout_rate

        if mode:  # in train mode, keep dropout as is
            self.mc_dropout_enabled = True
        # in eval mode, bank on the mc_dropout_enabled flag
        # mc_dropout_enabled is set equal to "mc_dropout" param given to predict()

    def forward(self, input):
        # NOTE: we could use the following line in case a different rate
        # is used for inference:
        # return F.dropout(input, self.applied_rate, True, self.inplace)

        return F.dropout(input, self.p, self.mc_dropout_enabled, self.inplace)

class TCN_ResidualBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_fn,
        weight_norm: bool,
        nr_blocks_below: int,
        num_layers: int,
        input_size: int,
        target_size: int,
    ):
        super().__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        # input_dim = input_size if nr_blocks_below == 0 else num_filters
        # output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        input_dim = input_size
        output_dim = target_size
        self.conv1 = nn.Conv2d(
            input_dim,
            input_dim,
            kernel_size,
            stride=1,
            padding=1,
            dilation=(dilation_base**nr_blocks_below),
        ).cuda()
        self.conv2 = nn.Conv2d(
            input_dim,
            input_dim,
            kernel_size,
            stride=1,
            padding=1,
            dilation=(dilation_base**nr_blocks_below),
        ).cuda()
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(
                self.conv1
            ), nn.utils.weight_norm(self.conv2)

        if input_dim != output_dim:
            self.conv3 = nn.Conv2d(input_dim, output_dim, 1).cuda()

    def forward(self, x):
        # print("-----self.nr_blocks_below:",self.nr_blocks_below)
        residual = x

        # first step
        # left_padding = (self.dilation_base**self.nr_blocks_below) * (
        #     self.kernel_size - 1
        # )
        # x = F.pad(x, (left_padding, 0)) #对输入张量进行填充
        # print("---first step")
        if x.shape[2] < self.kernel_size or x.shape[3] < self.kernel_size:
            # 如果卷积核大于输入大小，则不进行卷积操作，直接返回输入张量
            return x
        else:
            # 否则，进行卷积操作
            x = self.dropout_fn(F.relu(self.conv1(x))) #卷积 激活函数 随机置零

        # second step
        # print("---second step")
        # x = F.pad(x, (left_padding, 0)) #填充
        # print("left_padding:",x.shape)
        if x.shape[2] < self.kernel_size or x.shape[3] < self.kernel_size:
            # 如果卷积核大于输入大小，则不进行卷积操作，直接返回输入张量
            return x
        else:
            x = self.conv2(x) #卷积

        # print("conv:",x.shape)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x) #激活函数
        x = self.dropout_fn(x) #随机置零
        # print("second step relu dropout:",x.shape)

        # add residual
        if x.shape[2] < self.kernel_size or x.shape[3] < self.kernel_size:
            # 如果卷积核大于输入大小，则不进行卷积操作，直接返回输入张量
            return x
        else:
            if self.conv1.in_channels != self.conv2.out_channels:
                residual = self.conv3(residual)

        x = x + residual[:, :, :x.shape[2], :x.shape[3]]


        return x

class MLP_ResidualBlock(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_size: int,
            dropout= 0.,
            # use_layer_norm: bool,
    ):
        super().__init__()
        # dense layer with ReLU activation with dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )
        # linear skip connection from input to output of self.dense
        self.skip = nn.Linear(input_dim, output_dim)
        # layer normalization as output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual connection
        x = self.dense(x) + self.skip(x)
        return x

#PatchMixer
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMixer(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = seq_len = configs.seq_len
        self.pred_len = pred_len = configs.pred_len

        # TCN
        self.n_filters = 3
        self.kernel_size = 3
        self.dilation_base = 2
        self.dropout = MonteCarloDropout(p=0.2)
        self.num_layers = None
        self.weight_norm = False
        self.input_size = configs.seq_len
        self.target_length = configs.pred_len


        # Patching
        self.batch_size=configs.batch_size
        self.enc_in=configs.enc_in
        self.patch_len = patch_len = configs.patch_len # 16
        self.stride = stride = configs.stride  # 8
        self.patch_num = patch_num = int((seq_len - patch_len) // stride )+1 #patch数量
        self.padding_patch = configs.padding_patch
        if configs.padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num = patch_num = patch_num + 1
        # 使用TCN来获取patching每个片段的时序信息
        # num_layers related to TCN
        if self.num_layers is None and self.dilation_base > 1:
            num_layers = math.ceil(
                math.log(
                    (self.input_size - 1)
                    * (self.dilation_base - 1)
                    / (self.kernel_size - 1)
                    / 2
                    + 1,
                    self.dilation_base,
                )
            )
        elif self.num_layers is None:
            num_layers = math.ceil(
                (self.input_size - 1) / (self.kernel_size - 1) / 2
            )
        self.num_layers = 1

        # Building TCN module
        self.res_blocks_list = []
        for i in range(self.num_layers):  # num_layers:4 384_288:7
            res_block = TCN_ResidualBlock(
                self.n_filters,
                self.kernel_size,
                self.dilation_base,
                self.dropout,
                self.weight_norm,
                i,
                num_layers,
                configs.enc_in,
                self.target_length // 2,
            )
            self.res_blocks_list.append(res_block)
        # self.res_blocks = nn.ModuleList(self.res_blocks_list)
        # 4
        self.mlp1 = Mlp(patch_len * patch_num, patch_len, patch_len * patch_num)
        self.mlp2 = Mlp(patch_len * patch_num, pred_len * 2, pred_len)

    def forward(self, x): # B, L, D -> B, H, D
        B, _, D = x.shape
        L = self.patch_num
        P = self.patch_len

        # z_res = self.lin_res(x.permute(0, 2, 1)) # B, L, D -> B, H, D
        # z_res = self.dropout_res(z_res)

        # patching

        # 初始化一个张量来存储所有 patch
        patched_tensor = torch.zeros(self.batch_size, self.patch_num, self.patch_len, self.enc_in).cuda()

        # 进行 patching 操作
        for i in range(self.patch_num):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, start_idx:end_idx, :]
            if patch.shape[1]!=0:
                patched_tensor[:, i, :, :] = patch
        z=patched_tensor.permute(0, 3, 1, 2)
        # for res_block in self.res_blocks_list:
        #     z = res_block(z)
        z=z.reshape(self.batch_size,self.enc_in,-1)
        z = self.mlp1(z)

        # 4
        z_mlp = self.mlp2(z) # B, D, L * P -> B, D, H
        return z_mlp.permute(0,2,1)

class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()
        # Defining parameters


        self.target_size = configs.enc_in
        self.nr_params = 1
        # RevIN 可逆实例归一化
        self.rev = RevIN(configs.enc_in)

        # self.Linear = nn.Linear(self.target_length//2, self.target_length)
        # self.MLP=MLP_ResidualBlock(input_dim=self.target_length//2,output_dim=self.target_length,hidden_size=32,dropout=0.1)
        self.PatchMixer = PatchMixer(configs)

    def forward(self, x, x_mark, y_true, y_mark):
        # z1 = self.rev(x, 'norm')
        y=self.PatchMixer(x)
        # z2 = self.rev(y, 'denorm')
        return y

    def first_prediction_index(self) -> int:
        return -self.output_chunk_length