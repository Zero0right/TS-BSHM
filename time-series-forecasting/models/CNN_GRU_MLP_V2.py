import torch
import torch.nn as nn

# 定义一个裁剪模块，用于去除多余的padding
class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# 定义一个TCN块，包含因果卷积，批标准化，激活函数，dropout和残差连接
#init初始化神经网络结构，初始化参数
#forward正向传播，输入数据经过各种操作传播至输出层的过程
#forward在训练过程中会被调用多次，具体取决于训练集大小、训练迭代次数epoch、批量大小batch_size
class TCNBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 dilation,
                 padding,
                 dropout):
        super(TCNBlock, self).__init__()
        # 一维卷积，提取时序特征
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
        # 自定义截断层，去除因果卷积产生的右侧多余部分
        self.chomp1 = Chomp(padding)
        # 批归一化层，加速收敛和提高泛化能力
        self.bn1 = nn.BatchNorm1d(output_dim)
        # 增加非线性
        self.relu1 = nn.ReLU()
        # 失活层，防止过拟合
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp(padding)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 顺序容器，将上述子模块按照顺序组合成完成的TCN块
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        # 一维卷积层，输入和输出维度不一致时进行降采样，以便残差连接
        self.downsample = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else None
        self.relu = nn.ReLU()
        # 初始化权重的函数，它将卷积层的权重服从均值为0，标准差为0.01的正态分布
        self.init_weights()

    # 初始化权重
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    # 前向传播 x:输入数据 (batch_size,input_dim,seq_len)
    def forward(self, x: torch.Tensor):
        # 将输入数据通过self.net得到输出数据out，一个三维张量，形状为(batch_size, output_dim, seq_len)
        out = self.net(x)
        # 判断是否需要进行降采样 降采样减少样本特征维度
        # res = x if self.downsample is None else self.downsample(x)
        # res = x
        return self.relu(out)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout,seq_len):
        super(TCN, self).__init__()
        self.tcn_network = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(num_channels * (seq_len-1), output_size)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将特征维度移动到第二个维度上
        out = self.tcn_network(x)
        return out.squeeze(1)

class ResidualBlock(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_size: int,
            dropout: float,
            use_layer_norm: bool,
    ):
        super().__init__()
        # dense layer with ReLU activation with dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )
        # linear skip connection from input to output of self.dense
        self.skip = nn.Linear(input_dim, output_dim)
        # layer normalization as output
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual connection
        x = self.dense(x) + self.skip(x)
        # layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class Model(nn.Module):
    def __init__(
            self,configs
    ):
        super(Model,self).__init__()
        self.input_chunk_length=configs.seq_len
        self.output_chunk_length=configs.pred_len
        self.output_dim=configs.enc_in
        self.hidden_size=128
        self.use_layer_norm=False
        self.dropout=0.1
        self.num_encoder_layers=1
        self.num_decoder_layers=1
        self.temporal_decoder_hidden=32
        self.decoder_input_dim=16
        self.tcn_output_dim=4
        self.d_model=configs.d_model
        encoder_dim = (
                self.input_chunk_length * self.output_dim
        )

        # zyc
        # self.tcn = TCNBlock(
        #     input_dim=configs.seq_len,
        #     output_dim=self.tcn_output_dim,
        #     kernel_size=2,
        #     dilation=1,
        #     padding=0,
        #     dropout=0.1).to("cuda")
        self.tcn = TCN(
            input_size=configs.enc_in,
            output_size=self.tcn_output_dim,
            num_channels=configs.seq_len,
            kernel_size=4,
            dropout=0.1,
            seq_len=configs.seq_len).to("cuda")

        # self.encoders = nn.Sequential(
        #     ResidualBlock(
        #         input_dim=configs.seq_len*configs.enc_in+self.tcn_output_dim,
        #         output_dim=self.hidden_size,
        #         hidden_size=self.hidden_size,
        #         use_layer_norm=self.use_layer_norm,
        #         dropout=self.dropout,
        #     ),
        # )

        self.gru = nn.GRU(
            input_size=configs.seq_len*configs.enc_in+self.tcn_output_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.decoders = nn.Sequential(
            # add decoder output layer
            ResidualBlock(
                input_dim=self.hidden_size,
                output_dim=self.decoder_input_dim * self.output_chunk_length,
                hidden_size=self.hidden_size,
                use_layer_norm=self.use_layer_norm,
                dropout=self.dropout,
            ),
        )
        self.temporal_decoder = ResidualBlock(
            input_dim=self.decoder_input_dim,
            output_dim=configs.enc_in,
            hidden_size=self.temporal_decoder_hidden,
            use_layer_norm=self.use_layer_norm,
            dropout=self.dropout,
        )
        self.lookback_skip = nn.Linear(
            self.input_chunk_length, self.output_chunk_length
        )

    def forward(
            self, x, x_mark, y_true, y_mark
    ) -> torch.Tensor:
        x_lookback = x
        # zyc tcn
        x_lookback_cnn_features = self.tcn(x_lookback)
        # print("x_lookback_cnn_features:",x_lookback_cnn_features.shape)

        # setup input to encoder
        encoded = [
            x_lookback,
            x_lookback_cnn_features
        ]
        # print("lookback:", x_lookback.shape)
        # print("x_dynamic_past_covariates:",x_dynamic_past_covariates.shape)
        # print("x_dynamic_future_covariates:",x_dynamic_future_covariates.shape)
        # print("x_static_covariates:",x_static_covariates.shape)
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)
        # print("encoded:", encoded.shape)

        # # 计算模型占用的空间大小
        # total_params = sum(p.numel() for p in self.encoders.parameters())
        # total_size_MB = total_params * 4 / (1024 ** 2)  # 将参数数量乘以每个浮点数所占的字节数（通常是4字节），然后转换为MB
        # print(f"self.encoders Total parameters: {total_params}; Total size: {total_size_MB:.2f} MB")
        # # 计算模型占用的空间大小
        # total_params = sum(p.numel() for p in self.decoders.parameters())
        # total_size_MB = total_params * 4 / (1024 ** 2)  # 将参数数量乘以每个浮点数所占的字节数（通常是4字节），然后转换为MB
        # print(f"self.decoders Total parameters: {total_params}; Total size: {total_size_MB:.2f} MB")
        # # 计算模型占用的空间大小
        # total_params = sum(p.numel() for p in self.temporal_decoder.parameters())
        # total_size_MB = total_params * 4 / (1024 ** 2)  # 将参数数量乘以每个浮点数所占的字节数（通常是4字节），然后转换为MB
        # print(f"self.temporal_decoder Total parameters: {total_params}; Total size: {total_size_MB:.2f} MB")
        # # 计算模型占用的空间大小
        # total_params = sum(p.numel() for p in self.lookback_skip.parameters())
        # total_size_MB = total_params * 4 / (1024 ** 2)  # 将参数数量乘以每个浮点数所占的字节数（通常是4字节），然后转换为MB
        # print(f"self.lookback_skip Total parameters: {total_params}; Total size: {total_size_MB:.2f} MB")

        # encoder, decode, reshape
        # encoded = self.encoders(encoded)
        encoded=self.gru(encoded)[0] #获取整个序列的隐藏状态
        decoded = self.decoders(encoded)
        # print("encoded_output:", encoded.shape)
        # print("decoded_outpur:", decoded.shape)

        # get view that is batch size x output chunk length x self.decoder_output_dim x nr params
        decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)

        # stack and temporally decode with future covariate last output steps
        temporal_decoder_input = [
            decoded,
        ]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]

        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
        print("temporal_decoder_input:",temporal_decoder_input.shape)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)
        # print("temporal_decoder_input:", temporal_decoder_input.shape)
        # print("temporal_decoder_output:", temporal_decoded.shape)

        # pass x_lookback through self.lookback_skip but swap the last two dimensions
        # this is needed because the skip connection is applied across the input time steps
        # and not across the output time steps
        skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)
        # print("skip:",skip.shape)
        # lookback_features_skip=self.lookback_skip(x_lookback_cnn_features.transpose(1, 2)).transpose(1, 2)

        # add skip connection
        y = temporal_decoded + skip.reshape_as(
            temporal_decoded
        ) \
            # +lookback_features_skip.reshape_as(temporal_decoded) # skip.view(temporal_decoded.shape)

        y = y.view(-1, self.output_chunk_length, self.output_dim)
        seq_last = x[:, -1:, :].detach()
        y = y + seq_last
        # print("y:", y.shape)
        return y
