import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Gate(nn.Module):
    def __init__(self, d_part1, d_part2, d_target, d_hidden):
        super().__init__()
        self.d_part1 = d_part1
        self.d_part2 = d_part2
        self.d_hid = d_target
        self.p1_tar_w = nn.Linear(d_part1, d_hidden)
        self.p1_tar_u = nn.Linear(d_target, d_hidden)
        self.p2_tar_w = nn.Linear(d_part2, d_hidden)
        self.p2_tar_u = nn.Linear(d_target, d_hidden)
        self.layer_norm = nn.LayerNorm(d_hidden)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input1_seq, input2_seq, target):
        p1_1 = self.p1_tar_w(input1_seq)
        p1_2 = self.p1_tar_u(target)
        p2_1 = self.p2_tar_w(input2_seq)
        p2_2 = self.p2_tar_u(target)

        z_l = F.tanh(p1_1 + p1_2)
        z_r = F.tanh(p2_1 + p2_2)

        z_w = torch.cat([z_l, z_r], dim=1)
        z_w = self.softmax(z_w)

        z_l_w = z_w[:, 0, :].unsqueeze(1)
        z_r_w = z_w[:, 1, :].unsqueeze(1)

        out = z_l_w * input1_seq + z_r_w * p2_1
        # out = self.layer_norm(out)
        return out





