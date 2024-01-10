import torch
import torch.nn as nn

class CNNModule(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout,seq_len):
        super(CNNModule, self).__init__()
        self.cnn_network = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(num_channels * (seq_len-1), output_size)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将特征维度移动到第二个维度上
        out = self.cnn_network(x)
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
        self.seq_len=configs.seq_len
        self.pred_len=configs.pred_len
        self.enc_in=configs.enc_in
        self.hidden_size=128
        self.use_layer_norm=False
        self.dropout=0.1
        self.num_encoder_layers=1
        self.num_decoder_layers=1
        self.temporal_decoder_hidden=32
        self.decoder_input_dim=16
        self.cnn_output_dim=4
        self.d_model=configs.d_model
        # zyc
        self.cnn = CNNModule(
            input_size=self.enc_in,
            output_size=self.cnn_output_dim,
            num_channels=self.seq_len,
            kernel_size=4,
            dropout=0.1,
            seq_len=configs.seq_len).to("cuda")

        self.encoders= nn.Sequential(
            nn.GRU(
                input_size=self.seq_len * self.enc_in + self.cnn_output_dim,
                hidden_size=self.hidden_size,
                num_layers=1,
                bias=True,
                batch_first=True,
            )
        )

        self.decoders = nn.Sequential(
            # add decoder output layer
            ResidualBlock(
                input_dim=self.hidden_size,
                output_dim=self.decoder_input_dim * self.pred_len,
                hidden_size=self.hidden_size,
                use_layer_norm=self.use_layer_norm,
                dropout=self.dropout,
            ),
        )
        self.temporal_decoder = ResidualBlock(
            input_dim=self.decoder_input_dim,
            output_dim=self.enc_in,
            hidden_size=self.temporal_decoder_hidden,
            use_layer_norm=self.use_layer_norm,
            dropout=self.dropout,
        )
        self.lookback_skip = nn.Linear(
            self.seq_len, self.pred_len
        )

    def forward(
            self, x, x_mark, y_true, y_mark
    ) -> torch.Tensor:
        # zyc cnns
        x_cnn_features = self.cnn(x)

        encoded = [
            x,
            x_cnn_features
        ]
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)

        # encoder, decode, reshape
        encoded = self.encoders(encoded)[0]
        decoded = self.decoders(encoded)

        # get view that is batch size x output chunk length x self.decoder_output_dim x nr params
        decoded = decoded.view(x.shape[0], self.pred_len, -1)

        # stack and temporally decode with future covariate last output steps
        temporal_decoder_input = [
            decoded,
        ]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]
        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)

        skip = self.lookback_skip(x.transpose(1, 2)).transpose(1, 2)

        # add skip connection
        y = temporal_decoded + skip.reshape_as(
            temporal_decoded
        )
        y = y.view(-1, self.pred_len, self.enc_in)
        seq_last = x[:, -1:, :].detach()
        y = y + seq_last
        return y
