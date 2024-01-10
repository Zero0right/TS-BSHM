import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()
        # Defining parameters
        self.hidden_dim = 25
        self.n_layers = 1
        self.target_size = configs.enc_in
        self.nr_params = 1
        # num_layers_out_fc = [] if num_layers_out_fc is None else num_layers_out_fc
        num_layers_out_fc = []
        self.out_len = configs.pred_len
        self.name = "RNN"

        # Defining the RNN module
        self.rnn = getattr(nn, self.name)(
            configs.enc_in, self.hidden_dim, self.n_layers, batch_first=True, dropout=configs.dropout
        )

        # The RNN module is followed by a fully connected layer, which maps the last hidden layer
        # to the output of desired length
        last = self.hidden_dim
        feats = []
        for feature in num_layers_out_fc + [self.out_len * self.target_size * self.nr_params]:
            feats.append(nn.Linear(last, feature))
            last = feature
        self.fc = nn.Sequential(*feats) #多层全连接层

    def forward(self, x_in, x_mark, y_true, y_mark):
        x= x_in
        # data is of size (batch_size, input_chunk_length, input_size)
        batch_size = x.size(0)
        out, hidden = self.rnn(x)
        # print("out:",out.shape)

        """ Here, we apply the FC network only on the last output point (at the last time step)
        """
        if self.name == "LSTM":
            hidden = hidden[0]
        predictions = hidden[-1, :, :]
        predictions = self.fc(predictions)
        predictions = predictions.view(
            batch_size, self.out_len, self.target_size
        )
        # print("prediction:",predictions.shape)
        # predictions is of size (batch_size, output_chunk_length, 1)
        return predictions