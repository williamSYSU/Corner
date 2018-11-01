# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : aspect_sum.py
# @Time         : Created at 2018/9/20
# @Blog         : http://zhiweil.ml/
# @Description  : Averaging the aspect embedding
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn


class AspectMean(nn.Module):
    def __init__(self, max_sen_len, if_expand=False):
        """
        :param max_sen_len: maximum length of sentence
        """
        super(AspectMean, self).__init__()
        self.max_sen_len = max_sen_len
        self.if_expand = if_expand

    def forward(self, aspect):
        """

        :param aspect: size: [batch_size, max_asp_len, embed_size]
        :return: aspect mean embedding, size: [batch_size, (max_sen_len,) embed_size]
        """
        len_tmp = torch.sum(aspect != 0, dim=2)
        aspect_len = torch.sum(len_tmp != 0, dim=1).unsqueeze(dim=1).float()
        out = aspect.sum(dim=1)
        out = out.div(aspect_len)
        # 求均值后，匹配句子长度，复制多个aspect embedding
        if self.if_expand:
            out = out.unsqueeze(dim=1).expand(-1, self.max_sen_len, -1)
        return out
