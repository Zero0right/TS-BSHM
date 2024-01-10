import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import math

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

class _ResidualBlock(nn.Module):
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

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim,
            num_filters,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        self.conv2 = nn.Conv1d(
            num_filters,
            output_dim,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(
                self.conv1
            ), nn.utils.weight_norm(self.conv2)

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        # print("-----self.nr_blocks_below:",self.nr_blocks_below)
        residual = x
        # print("residual:",residual.shape)

        # first step
        left_padding = (self.dilation_base**self.nr_blocks_below) * (
            self.kernel_size - 1
        )
        x = F.pad(x, (left_padding, 0)) #对输入张量进行填充
        # print("---first step")
        # print("left_padding:",x.shape)
        x = self.dropout_fn(F.relu(self.conv1(x))) #卷积 激活函数 随机置零
        # print("conv relu dropout:",x.shape)

        # second step
        # print("---second step")
        x = F.pad(x, (left_padding, 0)) #填充
        # print("left_padding:",x.shape)
        x = self.conv2(x) #卷积
        # print("conv:",x.shape)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x) #激活函数
        x = self.dropout_fn(x) #随机置零
        # print("second step relu dropout:",x.shape)

        # add residual
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual
        # print("residual :",residual.shape)
        # print("x:",x.shape)
        # print("-----end---")

        return x


class Model(nn.Module):
    # def __init__(
    #     self,
    #     input_size: int,
    #     kernel_size: int,
    #     num_filters: int,
    #     num_layers: Optional[int],
    #     dilation_base: int,
    #     weight_norm: bool,
    #     target_size: int,
    #     nr_params: int,
    #     target_length: int,
    #     dropout: float,
    #     **kwargs
    # ):
    def __init__(self,configs):
        super().__init__()
        # Defining parameters
        self.input_size = configs.seq_len
        self.n_filters = 3
        self.kernel_size = 3
        self.target_length = configs.pred_len
        self.target_size = configs.enc_in
        self.nr_params = 1
        self.dilation_base = 2
        self.dropout = MonteCarloDropout(p=0.2)
        self.num_layers=None
        self.weight_norm=False

        # If num_layers is not passed, compute number of layers needed for full history coverage
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
        self.num_layers = num_layers
        # print("self.num_layers：",self.num_layers)

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers): #num_layers:4 384_288:7
            res_block = _ResidualBlock(
                self.n_filters,
                self.kernel_size,
                self.dilation_base,
                self.dropout,
                self.weight_norm,
                i,
                num_layers,
                self.input_size,
                self.target_length//2,
            )
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)
        self.Linear = nn.Linear(self.target_length//2, self.target_length)

    def forward(self, x_in, x_mark, y_true, y_mark):
        x = x_in

        for res_block in self.res_blocks_list:
            x = res_block(x)
        y = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return y

    def first_prediction_index(self) -> int:
        return -self.output_chunk_length