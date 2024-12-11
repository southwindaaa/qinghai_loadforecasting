import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in

        self.mlp = nn.Linear(self.seq_len*self.channels, 512)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(512, 256)
        self.mlp3 = nn.Linear(256, self.pred_len)
        self.nn_seq = nn.Sequential(self.mlp, 
                                    self.mlp2, 
                                    self.mlp3)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = self.nn_seq(x_enc.permute(0,2,1))
        return x.permute(0,2,1)
