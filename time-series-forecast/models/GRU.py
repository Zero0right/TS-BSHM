import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.gru=nn.GRU(configs.seq_len, configs.pred_len, configs.e_layers)

    def forward(self, x, enc_mark, dec, dec_mark):
        x = torch.transpose(x, 1, 2)
        output = self.gru(x)
        output = torch.transpose(output[0], 1, 2)
        return output