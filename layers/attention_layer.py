import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from layers.scale_dot_prod_attn import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_input, d_k, d_v, n_head):
        super().__init__()
        self.d_input = d_input
        self.n_head = n_head
        self.w_qs = nn.Linear(d_input, n_head * d_k)
        self.w_ks = nn.Linear(d_input, n_head * d_k)
        self.w_vs = nn.Linear(d_input, n_head * d_v)

        self._score = nn.Linear(d_input * 2, 1)
        # self.mlp = nn.Linear(d_input * 2, d_input * self.n_head)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(n_head * d_input, d_input)
        self.layer_norm = nn.LayerNorm(d_input)
        self.tanh = nn.Tanh()

    def forward(self, input):
        batch_size = len(input)
        len_input = len(input[0])
        q = self.w_qs(input).view(batch_size, len_input, self.n_head, self.d_input)
        k = self.w_ks(input).view(batch_size, len_input, self.n_head, self.d_input)
        v = self.w_vs(input).view(batch_size, len_input, self.n_head, self.d_input)

        residual = input

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_input, self.d_input)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_input, self.d_input)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_input, self.d_input)  # (n*b) x lv x dv

        q = torch.unsqueeze(q, dim=1).expand(-1, len_input, -1, -1)
        k = torch.unsqueeze(k, dim=2).expand(-1, -1, len_input, -1)
        kq = self.tanh(self._score(torch.cat((k, q), dim=-1)))

        score = F.softmax(kq, dim=-1).squeeze(3)
        output = torch.bmm(score, v)

        output = output.view(self.n_head, batch_size, len_input, self.d_input)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_input, -1)  # b x lq x (n*dv)

        output = self.fc(output)
        output = self.layer_norm(output + residual)

        return output


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(config.dropout)
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(self.tanh(self.w_1(output)))
        output = output.transpose(1, 2)
        # output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class NormalAttention(nn.Module):
    def __init__(self, d_input, d_target, d_hidden):
        super(NormalAttention, self).__init__()
        self.attn = nn.Linear(d_input, d_hidden)
        self.attn_target = nn.Linear(d_target, d_hidden)
        # self.combine = nn.Linear(d_input + d_target, 1)
        self.attn_target_1 = nn.Linear(d_hidden + d_hidden, d_hidden)
        self.combine = nn.Linear(d_hidden, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(d_input)
        self.tanh = nn.Tanh()

    def forward(self, input_seq, target_seq):
        combine_input = self.attn(input_seq)
        tar = self.attn_target(target_seq)
        tar = tar.unsqueeze(1)
        combine_tar = tar.view(len(input_seq), 1, -1)
        _combine_input = torch.unsqueeze(combine_input, dim=1).expand(-1, 1, -1, -1)
        _combine_tar = torch.unsqueeze(combine_tar, dim=2).expand(-1, -1, len(input_seq[0]), -1)

        # _combine_input = torch.unsqueeze(input_seq, dim=1).expand(-1, 1, -1, -1)
        # _combine_tar = torch.unsqueeze(tar, dim=2).expand(-1, -1, len(input_seq[0]), -1)

        # _combine_tar = combine_tar.view(1, 1, 1, 50).expand(-1, -1, len(input_seq[1]), -1)

        # attn_out = nn.Tanh(_combine_tar + _combine_input)
        attn_out = self.tanh(self.attn_target_1(torch.cat((_combine_input, _combine_tar), dim=-1)))
        attn_out = self.dropout(self.combine(attn_out))
        attn_score = self.softmax(attn_out.squeeze(3))
        # attn_out = input_seq * attn
        # attn_out = attn_out.sum(dim=1)
        out = torch.bmm(attn_score, input_seq)
        # out = self.layer_norm(out)

        return out


class MultiHeadAttentionDotProduct(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.fc(output)
        output = self.layer_norm(output + residual)

        return output, attn
