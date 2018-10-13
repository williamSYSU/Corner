import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class DotProductAttention(nn.Module):

    # Scaled-dot-product Attention layer

    def __init__(self, d_query, d_key, d_value, mapping_on="query"):

        # mapping_on: whether linear transformation is required, mapping query or key into a new space
        # mapping_on: "query" || "key" || "both" || "none"

        super(DotProductAttention, self).__init__()

        self.d_query = d_query
        self.d_key = d_key
        self.d_value = d_value
        self.mapping_on = mapping_on

        if mapping_on == "query":
            # mapping query to key's space
            self.q_h = nn.Linear(d_query, d_key)
        elif mapping_on == "key":
            # mapping key to query's space
            self.k_h = nn.Linear(d_key, d_query)
        elif mapping_on == "both":
            # mapping query and key into the same space
            self.q_h = nn.Linear(d_query, d_value)
            self.k_h = nn.Linear(d_key, d_value)

        self.temper = np.power(d_value, 0.5)
        # self.weight = nn.Parameter(torch.Tensor(d_query, d_query))
        # uniform = 1. / math.sqrt(self.d_query)
        # self.weight.data.uniform_(-uniform, uniform)

    def forward(self, q, k, v):

        # query: [s_batch, 1, d_query]
        # key: [*, l_key, d_key] # usually d_key = d_query
        # value: [*, l_value, d_value] # usually l_value = l_key
        # if len(key.shape) == 3, then "*" must equal to s_batch

        if self.mapping_on == "query":
            q = self.q_h(q)
        elif self.mapping_on == "key":
            k = self.k_h(k)
        elif self.mapping_on == "both":
            q = self.q_h(q)
            k = self.k_h(k)
        # print("11", k[0])
        # [s_b, 1, d_q] * [*, d_k, l_k] = [s_b, 1, l_k]
        if len(k.shape) == 3:
            # similarity = torch.matmul(q, k.permute(0, 2, 1)) / self.temper
            # similarity = torch.matmul(q, k.permute(0, 2, 1))
            similarity = torch.matmul(q, k.permute(0, 2, 1))
        else:
            # len(k.shape) == 2
            similarity = torch.matmul(q, k.transpose(0, 1)) / self.temper

        # print("22", similarity[0])
        attn = f.softmax(similarity, dim=-1)
        # print("attn : ", attn[1])
        # [s_b, 1, l_k] * [*, l_v, d_v] = [s_b, 1, d_v]
        output = torch.matmul(attn, v)
        # print("44", output[0])

        return output, attn
