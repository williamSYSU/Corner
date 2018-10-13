# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : Corner-william
# @FileName     : Datapro.py
# @Time         : Created at 2018/10/11
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as f


class NormalAttention(nn.Module):
    def __init__(self, d_input, d_target, d_hidden, dropout=0.1):
        super(NormalAttention, self).__init__()
        self.d_input = d_input
        self.d_target = d_target
        self.d_hid = d_hidden
        self.attn = nn.Linear(d_input, d_hidden)
        self.attn_target = nn.Linear(d_target, d_hidden)
        # self.combine = nn.Linear(d_input + d_target, 1)
        self.attn_target_1 = nn.Linear(d_hidden + d_hidden, d_hidden)
        self.combine = nn.Linear(d_hidden, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
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

        # attn_out = self.tanh(_combine_tar + _combine_input)
        attn_out = self.tanh(self.attn_target_1(torch.cat((_combine_input, _combine_tar), dim=-1)))
        attn_out = self.dropout(self.combine(attn_out))
        attn_score = self.softmax(attn_out.squeeze(3))
        # attn_out = input_seq * attn
        # attn_out = attn_out.sum(dim=1)
        out = torch.bmm(attn_score, input_seq)

        return out


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

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input1_seq, input2_seq, target):
        p1_1 = self.p1_tar_w(input1_seq)
        p1_2 = self.p1_tar_u(target)
        p2_1 = self.p2_tar_w(input2_seq)
        p2_2 = self.p2_tar_u(target)

        z_l = self.tanh(p1_1 + p1_2)
        z_r = self.tanh(p2_1 + p2_2)

        z_w = torch.cat([z_l, z_r], dim=1)
        z_w = self.softmax(z_w)

        z_l_w = z_w[:, 0, :].unsqueeze(1)
        z_r_w = z_w[:, 1, :].unsqueeze(1)

        out = z_l_w * input1_seq + z_r_w * p2_1

        return out


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


# import Datapro
MAX_LENGTH = 300
# BATCH_SIZE = 20
# instructor = Datapro.Instructor()
# embed = instructor.embedding
device = torch.device("cuda")
final_embedding = np.array(np.load("embed/Vector_word_embedding_all.npy"))
# add = -1 + 2*np.random.random(300)
add = np.zeros(300)
final_embedding = np.row_stack((final_embedding, add))
embed = torch.from_numpy(final_embedding)


class WdeRnnEncoderFix(nn.Module):
    def __init__(self, hidden_size, output_size, context_dim, trained_aspect, dropout=0.1):
        super(WdeRnnEncoderFix, self).__init__()
        #         self.opt = opt
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(hidden_size, 300, bidirectional=True, batch_first=True)
        self.embedded = nn.Embedding.from_pretrained(embed)
        self.aspect_embed = nn.Embedding.from_pretrained(trained_aspect)
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.context_input_ = nn.Linear(600, 50)
        self.embedding_layers = nn.Linear(0 + hidden_size, output_size)
        # self.slf_attention = attention.MultiHeadAttention(600, 3)
        # self.slf_attention = attention.MultiHeadAttentionDotProduct(3, 600, 300, 300, 0.01)
        # self.Position_wise = attention.PositionwiseFeedForward(600, 600, 0.01)
        self.attention = NormalAttention(600, 50, 50)
        self.gate = Gate(300, 50, 50, 300)

        self.min_context = nn.Linear(300, 50)

    def forward(self, input, hidden):
        BATCH_SIZE = len(input)
        batch_len = input[:, 0]
        batch_context = input[:, 1]
        input_index = input[:, 2:]
        input_index = input_index.long()
        # seq_len = batch_len.item()
        # input_index = input_index[0][0:seq_len]
        # print('input_index',input_index)
        # print(hidden.size())
        sorted_seq_lengths, indices = torch.sort(batch_len, descending=True)
        # TODO: change NO.1 -> switch order of following two lines
        _, desorted_indices = torch.sort(indices, descending=False)
        input_index = input_index[:, 0: sorted_seq_lengths[0]]

        input_index = input_index[indices]
        input_value = self.embedded(input_index)
        input_value = input_value.float()
        packed_inputs = nn.utils.rnn.pack_padded_sequence(input_value, sorted_seq_lengths.cpu().data.numpy()
                                                          , batch_first=True)

        # print(sorted_seq_lengths, indices)
        output, hidden = self.blstm(packed_inputs, hidden)
        padded_res, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        desorted_output = padded_res[desorted_indices]

        '''
        self attention module add or not?
        point wise product add or not?
        '''
        # desorted_output = self.slf_attention(desorted_output, context_input)
        # desorted_output, _ = self.slf_attention(desorted_output, desorted_output, desorted_output)
        # desorted_output = self.Position_wise(desorted_output)

        '''
        Normal attention module add or not?
        '''

        context_input = self.aspect_embed(batch_context).float()
        context_input = self.min_context(context_input)

        attn_target = self.attention(desorted_output, context_input)

        desorted_output = F.max_pool2d(desorted_output, (desorted_output.size(1), 1))

        # output.view(self.hidden_size * 2, -1)
        # output = torch.max(output)
        desorted_output = self.tanh(self.hidden_layer(desorted_output))

        context_input = context_input.view(BATCH_SIZE, 1, 50)
        _context_input = self.tanh(self.context_input_(attn_target))

        gate_out = self.gate(desorted_output, _context_input, context_input)

        embedding_input = torch.cat((desorted_output, _context_input), dim=2)
        desorted_output = self.tanh(self.embedding_layers(gate_out))
        return desorted_output

    def initHidden(self, BATCH_SIZE):
        return (torch.zeros(2, BATCH_SIZE, self.hidden_size, device=device),
                torch.zeros(2, BATCH_SIZE, self.hidden_size, device=device))


class PreTrainABAE_fix(nn.Module):
    def __init__(self, embed_dim, n_aspect, aspect_embedding):
        super(PreTrainABAE_fix, self).__init__()
        # self.opt = opt
        self.embed_dim = embed_dim
        self.n_aspect = n_aspect
        self.embedded = nn.Embedding.from_pretrained(embed)

        # query: global_content_embeding: [batch_size, embed_dim]
        # key: inputs: [batch_size, doc_size, embed_dim]
        # value: inputs
        # mapping the input word embedding to global_content_embedding space
        self.sentence_embedding_attn = DotProductAttention(
            d_query=embed_dim,
            d_key=embed_dim,
            d_value=embed_dim,
            mapping_on="key"
        )

        # embed_dim => n_aspect
        self.aspect_linear = nn.Linear(embed_dim, n_aspect)

        # initialized with the centroids of clusters resulting from running k-means on word embeddings in corpus
        self.aspect_lookup_mat = nn.Parameter(data=aspect_embedding, requires_grad=True)
        # self.aspect_lookup_mat = nn.Parameter(torch.Tensor(n_aspect, embed_dim).double())
        # self.aspect_lookup_mat.data.uniform_(-1, 1)

    def forward(self, inputs, eps=config.epsilon):
        input_lengths = inputs[:, 0]
        inputs = inputs[:, 2:]
        input_index = inputs.long()
        sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        input_index = input_index[:, 0: sorted_seq_lengths[0]]
        # input_index = input_index[indices]
        inputs = self.embedded(input_index).double()

        # inputs: [batch_size, doc_size, embed_dim]
        # input_lengths: [batch_size]
        # averaging embeddings in a document: [batch_size, 1, embed_dim]
        avg_denominator = input_lengths.repeat(self.embed_dim).view(self.embed_dim, -1).transpose(0, 1).float()
        global_content_embed = torch.sum(inputs.double(), dim=1).div(avg_denominator.double())
        global_content_embed = global_content_embed.unsqueeze(dim=1)

        # construct sentence embedding, with attention(query: global_content_embed, keys: inputs, value: inputs)
        # [batch_size, embed_dim]
        sentence_embedding, _ = self.sentence_embedding_attn(
            global_content_embed.float(), inputs.float(), inputs.float()
        )
        # print("attn : ", sentence_embedding)
        sentence_embedding = sentence_embedding.squeeze(dim=1)

        # [batch_size, n_aspect]
        aspect_weight = F.softmax(self.aspect_linear(sentence_embedding), dim=1)

        _, predicted = torch.max(aspect_weight.data, 1)

        return predicted

    def regular(self, eps=config.epsilon):
        div = eps + torch.norm(self.aspect_lookup_mat, 2, -1)
        div = div.view(-1, 1)
        self.aspect_lookup_mat.data = self.aspect_lookup_mat / div


from io import open
import unicodedata
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time

batch_size = 128
sample_size = 100000


class DataProcess(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Instructor:
    def __init__(self):
        all_data, embedding, self.pos_test, self.neg_test = push()
        self.embedding = embedding
        self.train_data_loader = DataLoader(
            dataset=all_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8
        )

    def beginTrain_lstm(self):

        init_aspect = np.array(np.load("initAspect.npy"))
        # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
        init_aspect = torch.from_numpy(init_aspect)
        PreTrainABAE = PreTrainABAE_fix(300, 24, init_aspect).cuda()

        pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
        aspect_dict = PreTrainABAE.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
        aspect_dict.update(pre_trained_dict)
        PreTrainABAE.load_state_dict(aspect_dict)
        PreTrainABAE = PreTrainABAE.eval()

        trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data
        run = WdeRnnEncoderFix(300, 300, 50, trained_aspect)
        run = run.cuda()
        # TODO: change NO.2 -> chagne optimizer initialize
        # params = []
        # for param in run.parameters():
        #     if param.requires_grad:
        #         params.append(param)
        # optimizer = optim.SGD(params, lr=0.0001)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, run.parameters()), lr=0.0001)
        loss_func = nn.TripletMarginLoss(margin=4.0, p=2)

        for epoch in range(200):
            run_hidden = run.initHidden(batch_size)
            loss_last = torch.tensor([0], dtype=torch.float)

            for idx, sample_batch in enumerate(self.train_data_loader):
                now = time.time()
                run = run.train()
                input1 = sample_batch['input1'].cuda()
                input2 = sample_batch['input2'].cuda()
                input3 = sample_batch['input3'].cuda()
                # if input1[:,0].item() < 3 or input2[:,0].item() < 3 or input3[:,0].item() < 3:
                #     continue
                aspect_info = PreTrainABAE(input1)
                input1[:, 1] = aspect_info
                aspect_info = PreTrainABAE(input2)
                input2[:, 1] = aspect_info
                aspect_info = PreTrainABAE(input3)
                input3[:, 1] = aspect_info
                out1 = run(input1.cuda(), run_hidden).view(batch_size, 300)
                out2 = run(input2.cuda(), run_hidden).view(batch_size, 300)
                out3 = run(input3.cuda(), run_hidden).view(batch_size, 300)

                loss_last = loss_func(out1, out2, out3)
                loss_last.backward()
                optimizer.step()
            # TODO: remove valid
            # if epoch % 2 == 0:
            #     run.zero_grad()
            #     run = run.eval()
            #     valid_now = self.valid(PreTrainABAE, run)
            #     a = round((loss_last).item(), 5)
            #     b = round(valid_now, 5)
            #     if valid_now > 1.13:
            #         file_name = "pretrainmodel/" + "every2_loss_" + str(a) + "valid_" + str(
            #             b) + ".pkl"
            #         torch.save(run.state_dict(), file_name)
            #     valid_compare = valid_now
            #
            #     print('epoch {} of {}: TEST : {}'.format(epoch, 200, valid_now))
            print('epoch {} of {}: loss : {}'.format(epoch, 200, (loss_last).item()))


def push():
    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?\(\)\"])", r"", s)
        s = re.sub(r"[^0-9a-zA-Z]+", r" ", s)
        return s

    def normalize(s):
        # s = unicodeToAscii(s.strip())
        s = re.sub(r"([\[\]\"\n])", r"", s)
        return s

    def unicodeToAscii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn'
                       )

    lines_pos1 = open(
        'data/Weakly_labeled_data_1.1M/camera_positive.csv'
        , encoding='utf-8').read().strip().split('\n')
    lines_neg1 = open(
        'data/Weakly_labeled_data_1.1M/camera_negative.csv'
        , encoding='utf-8').read().strip().split('\n')
    lines_pos2 = open(
        'data/Weakly_labeled_data_1.1M/cellphone_positive.csv'
        , encoding='utf-8').read().strip().split('\n')
    lines_neg2 = open(
        'data/Weakly_labeled_data_1.1M/cellphone_negative.csv'
        , encoding='utf-8').read().strip().split('\n')
    lines_pos3 = open(
        'data/Weakly_labeled_data_1.1M/laptop_positive.csv'
        , encoding='utf-8').read().strip().split('\n')
    lines_neg3 = open(
        'data/Weakly_labeled_data_1.1M/laptop_negative.csv'
        , encoding='utf-8').read().strip().split('\n')

    lines_pos = lines_pos1 + lines_pos2 + lines_pos3
    lines_neg = lines_neg1 + lines_neg2 + lines_neg3

    lines = open(
        'data/Labeled_data_11754/new_11754.csv'
        , encoding='gbk').read().strip().split('\n')

    pairs_classify = [normalizeString(s) for s in lines]

    pairs_pos = [normalizeString(s) for s in lines_pos]
    pairs_neg = [normalizeString(s) for s in lines_neg]

    vocab = {}
    print("=" * 100)
    print("Take Word To Vec")

    final_embedding = np.array(np.load("embed/Vector_word_embedding_all.npy"))
    # final_embedding = np.delete(final_embedding, 60905, 0)
    # print(final_embedding[60905])

    maxlen = 0
    bb = []

    def word2idx(sentence, vocab, maxlen, bb):
        items = sentence.strip().split()
        if len(items) > maxlen:
            maxlen = len(items)
            bb = items
        for word in items:
            if word not in vocab:
                vocab[word] = len(vocab)
        return maxlen, bb

    for line in pairs_classify:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    for line in pairs_pos:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    for line in pairs_neg:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    input_sentence_1 = 108947 + np.zeros((len(pairs_pos), 302))
    input_sentence_1 = input_sentence_1.astype(np.int)
    input_sentence_2 = 108947 + np.zeros((len(pairs_neg), 302))
    input_sentence_2 = input_sentence_2.astype(np.int)

    def sentence2vec(sentence, vocab, wordindex):
        items = sentence.strip().split()
        length = len(items)
        for word in items:
            wordindex.append(vocab[word])
        return length, wordindex

    def cal_sentence_index():
        for line in range(len(pairs_pos)):
            wordindex = []
            length, wordindex = sentence2vec(pairs_pos[line], vocab, wordindex)
            input_sentence_1[line][0] = length
            input_sentence_1[line][1] = 10
            input_sentence_1[line][2:length + 2] = np.array(wordindex)

        for line in range(len(pairs_neg)):
            wordindex = []
            length, wordindex = sentence2vec(pairs_neg[line], vocab, wordindex)
            input_sentence_2[line][0] = length
            input_sentence_2[line][1] = 10
            input_sentence_2[line][2:length + 2] = np.array(wordindex)
        return input_sentence_1, input_sentence_2

    cal_sentence_index()

    #     add = -1 + 2*np.random.random(300)
    add = np.zeros(300)
    final_embedding = np.row_stack((final_embedding, add))

    np.random.shuffle(input_sentence_1)
    np.random.shuffle(input_sentence_2)

    input_pos_train = input_sentence_1[:int(len(input_sentence_1) * 0.7), :]
    input_neg_train = input_sentence_2[:int(len(input_sentence_2) * 0.7), :]

    input_pos_test = input_sentence_1[int(len(input_sentence_1) * 0.7):, :]
    input_neg_test = input_sentence_2[int(len(input_sentence_2) * 0.7):, :]

    def random_sample(matrix, sample_size):
        matrix_after = []
        sample_index = np.random.randint(0, len(matrix), sample_size)
        for i in sample_index:
            # np.row_stack((matrix_after, matrix[i]))
            matrix_after.append(matrix[i])
        return np.array(matrix_after)

    train_pos_1 = random_sample(input_pos_train, sample_size)
    train_pos_2 = random_sample(input_pos_train, sample_size)
    train_pos_neg = random_sample(input_neg_train, sample_size)
    train_neg_1 = random_sample(input_neg_train, sample_size)
    train_neg_2 = random_sample(input_neg_train, sample_size)
    train_neg_pos = random_sample(input_pos_train, sample_size)

    train_dim1 = np.vstack((train_pos_1, train_neg_1))
    train_dim2 = np.vstack((train_pos_2, train_neg_2))
    train_dim3 = np.vstack((train_pos_neg, train_neg_pos))

    def read_data(dim_1, dim_2, dim_3):
        all_data = []
        for idx in range(len(dim_1)):
            items = torch.from_numpy(dim_1[idx])
            items1 = torch.from_numpy(dim_2[idx])
            items2 = torch.from_numpy(dim_3[idx])
            data = {
                'input1': items,
                'input2': items1,
                'input3': items2
            }
            all_data.append(data)
        return all_data

    all_data = DataProcess(read_data(train_dim1, train_dim2, train_dim3))
    return all_data, final_embedding, np.array(input_pos_test[0:8000, :]), np.array(input_neg_test[0:8000, :])


if __name__ == '__main__':
    instructor = Instructor()
    # instructor.beginTrain()
    instructor.beginTrain_lstm()
