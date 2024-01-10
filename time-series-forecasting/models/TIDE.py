import torch
import torch.nn as nn
# from torch.nn import Module

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
        encoder_dim = (
                self.input_chunk_length * self.output_dim
        )
        print("encoder_dim:", encoder_dim)

        self.encoders = nn.Sequential(
            ResidualBlock(
                input_dim=configs.seq_len*configs.enc_in,
                output_dim=self.hidden_size,
                hidden_size=self.hidden_size,
                use_layer_norm=self.use_layer_norm,
                dropout=self.dropout,
            ),
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
            input_dim=16,
            output_dim=configs.enc_in,
            hidden_size=32,
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

        # setup input to encoder
        encoded = [
            x_lookback,
        ]
        # print("lookback:", x_lookback.shape)
        # print("x_dynamic_past_covariates:",x_dynamic_past_covariates.shape)
        # print("x_dynamic_future_covariates:",x_dynamic_future_covariates.shape)
        # print("x_static_covariates:",x_static_covariates.shape)
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)

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
        encoded = self.encoders(encoded)
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
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)
        # print("temporal_decoder_input:", temporal_decoder_input.shape)
        # print("temporal_decoder_output:", temporal_decoded.shape)

        # pass x_lookback through self.lookback_skip but swap the last two dimensions
        # this is needed because the skip connection is applied across the input time steps
        # and not across the output time steps
        skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)

        # add skip connection
        y = temporal_decoded + skip.reshape_as(
            temporal_decoded
        )  # skip.view(temporal_decoded.shape)

        y = y.view(-1, self.output_chunk_length, self.output_dim)
        # print("y:", y.shape)
        return y
