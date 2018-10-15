import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import layers.Gate as Gate
import layers.attention_layer as attention
from layers.CRF import LinearCRF
from layers.attention import DotProductAttention


class WdeRnnEncoder(nn.Module):
    def __init__(self, hidden_size, output_size, context_dim, embed, trained_aspect):
        super(WdeRnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(hidden_size, 300, bidirectional=True, batch_first=True)
        self.embedded = nn.Embedding.from_pretrained(embed)
        self.aspect_embed = nn.Embedding.from_pretrained(trained_aspect)
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.context_input_ = nn.Linear(600, 50)
        self.embedding_layers = nn.Linear(0 + hidden_size, output_size)
        self.embedding_layer_s = nn.Linear(50 + hidden_size, output_size)
        self.classifier_layer = nn.Linear(output_size, 2)
        self.softmax = nn.Softmax(dim=2)
        # self.slf_attention = attention.MultiHeadAttention(600, 3)
        # self.slf_attention = attention.MultiHeadAttentionDotProduct(3, 600, 300, 300, 0.01)
        # self.Position_wise = attention.PositionwiseFeedForward(600, 600, 0.01)
        self.min_context = nn.Linear(300, 50)
        self.attention = attention.NormalAttention(600, 50, 50)
        self.gate = Gate.Gate(300, 50, 50, 300)
        self.dropout = nn.Dropout(config.dropout)

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
        desorted_output = self.tanh(self.dropout(self.embedding_layers(gate_out)))
        # desorted_output = self.tanh(self.dropout(self.embedding_layer_s(embedding_input)))

        # result = self.softmax(self.classifier_layer(desorted_output))
        result = self.classifier_layer(desorted_output)
        return result

    def initHidden(self, BATCH_SIZE):
        return (torch.zeros(2, BATCH_SIZE, self.hidden_size, device=config.device),
                torch.zeros(2, BATCH_SIZE, self.hidden_size, device=config.device))


class AttentionEncoder(nn.Module):
    def __init__(self, hidden_size, output_size, context_dim, embed):
        super(AttentionEncoder, self).__init__()
        self.slf_attention = attention.MultiHeadAttentionDotProduct(3, 300, 300, 300)
        # self.slf_attention = attention.MultiHeadAttention(300, 300, 300, 3)
        self.hidden_size = hidden_size
        self.embedded = nn.Embedding.from_pretrained(embed)
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.context_input_ = nn.Linear(300, 50)
        self.embedding_layers = nn.Linear(0 + hidden_size, output_size)
        self.classifier_layer = nn.Linear(output_size, 2)
        self.softmax = nn.Softmax(dim=2)
        # self.slf_attention = attention.MultiHeadAttention(600, 3)
        self.Position_wise = attention.PositionwiseFeedForward(300, 300)
        self.attention = attention.NormalAttention(300, 50, 50)
        self.gate = Gate.Gate(300, 50, 50, 300)
        self.dropout = nn.Dropout(config.dropout)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(301, 300, padding_idx=200),
            freeze=True)

    def forward(self, input, context_input):
        BATCH_SIZE = len(input)
        batch_len = input[:, 0]
        batch_context = input[:, 1]
        input_index = input[:, 2: 202]
        input_pos = input[:, 202:]
        input_index = input_index.long()
        input_pos = input_pos.long()
        # seq_len = batch_len.item()
        # input_index = input_index[0][0:seq_len]
        # print('input_index',input_index)
        # print(hidden.size())
        sorted_seq_lengths, indices = torch.sort(batch_len, descending=True)
        input_index = input_index[:, 0: sorted_seq_lengths[0]]
        input_pos = input_pos[:, 0: sorted_seq_lengths[0]]
        pos_value = self.position_enc(input_pos)
        input_value = self.embedded(input_index)
        # input_value = input_value + pos_value.double()
        input_value = input_value.float()

        # packed_inputs = nn.utils.rnn.pack_padded_sequence(input_value, sorted_seq_lengths.cpu().data.numpy()
        #                                                   , batch_first=True)

        # print(sorted_seq_lengths, indices)
        # output, hidden = self.blstm(packed_inputs, hidden)
        # padded_res, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # desorted_output = padded_res[desorted_indices]

        desorted_output, _ = self.slf_attention(input_value, input_value, input_value)
        # desorted_output = self.slf_attention(input_value)
        '''
        self attention module add or not?
        point wise product add or not?
        '''
        # desorted_output = self.slf_attention(desorted_output, context_input)
        # desorted_output, _ = self.slf_attention(desorted_output, desorted_output, desorted_output)
        desorted_output = self.Position_wise(desorted_output)

        '''
        Normal attention module add or not?
        '''
        attn_target = self.attention(desorted_output, context_input)

        desorted_output = F.max_pool2d(desorted_output, (desorted_output.size(1), 1))

        # output.view(self.hidden_size * 2, -1)
        # output = torch.max(output)
        desorted_output = self.tanh(self.hidden_layer(desorted_output))

        context_input = context_input.view(BATCH_SIZE, 1, 50)
        _context_input = self.tanh(self.context_input_(attn_target))

        gate_out = self.gate(desorted_output, _context_input, context_input)

        embedding_input = torch.cat((desorted_output, _context_input), dim=2)
        desorted_output = self.tanh(self.dropout(self.embedding_layers(gate_out)))

        result = self.softmax(self.classifier_layer(desorted_output))
        return result


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class PreTrainABAE(nn.Module):
    def __init__(self, aspect_embedding, embed):
        super(PreTrainABAE, self).__init__()
        self.embed_dim = config.embed_dim
        self.n_aspect = config.n_aspect
        self.embedded = nn.Embedding.from_pretrained(embed)

        # query: global_content_embeding: [batch_size, embed_dim]
        # key: inputs: [batch_size, doc_size, embed_dim]
        # value: inputs
        # mapping the input word embedding to global_content_embedding space
        self.sentence_embedding_attn = DotProductAttention(
            d_query=self.embed_dim,
            d_key=self.embed_dim,
            d_value=self.embed_dim,
            mapping_on="key"
        )

        # embed_dim => n_aspect
        self.aspect_linear = nn.Linear(self.embed_dim, self.n_aspect)

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

        div = eps + torch.norm(self.aspect_lookup_mat, 2, -1)
        div = div.view(-1, 1)

        aspect_matrix = self.aspect_lookup_mat / div
        reg = torch.sum(torch.matmul(aspect_matrix, aspect_matrix.permute(1, 0)) ** 2 -
                        torch.eye(24).double().to(config.device))

        return predicted, self.aspect_lookup_mat.data, reg

    def regular(self, eps=config.epsilon):
        div = eps + torch.norm(self.aspect_lookup_mat, 2, -1)
        div = div.view(-1, 1)
        self.aspect_lookup_mat.data = self.aspect_lookup_mat / div


class align_WdeRnnEncoder(nn.Module):
    def __init__(self, hidden_size, output_size, context_dim, embed, trained_aspect):
        super(align_WdeRnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(hidden_size, 300, bidirectional=True, batch_first=True)
        self.embedded = nn.Embedding.from_pretrained(embed)
        self.aspect_embed = nn.Embedding.from_pretrained(trained_aspect)
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.context_input_ = nn.Linear(600, 50)
        self.embedding_layers = nn.Linear(0 + hidden_size, output_size)
        self.classifier_layer = nn.Linear(output_size, 2)
        self.softmax = nn.Softmax(dim=2)
        # self.slf_attention = attention.MultiHeadAttention(600, 3)
        # self.slf_attention = attention.MultiHeadAttentionDotProduct(3, 600, 300, 300, 0.01)
        # self.Position_wise = attention.PositionwiseFeedForward(600, 600, 0.01)
        self.min_context = nn.Linear(300, 50)
        self.attention = attention.NormalAttention(600, 50, 50)
        self.gate = Gate.Gate(300, 50, 50, 300)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input, hidden, update_aspect):
        BATCH_SIZE = len(input)
        self.aspect_embed.weight.data.copy_(update_aspect)
        batch_len = input[:, 0]
        batch_context = input[:, 1]
        input_index = input[:, 2:]
        input_index = input_index.long()
        # seq_len = batch_len.item()
        # input_index = input_index[0][0:seq_len]
        # print('input_index',input_index)
        # print(hidden.size())
        sorted_seq_lengths, indices = torch.sort(batch_len, descending=True)
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
        desorted_output = self.tanh(self.dropout(self.embedding_layers(gate_out)))

        # result = self.softmax(self.classifier_layer(desorted_output))
        result = self.classifier_layer(desorted_output)
        return result

    def initHidden(self, BATCH_SIZE):
        return (torch.zeros(2, BATCH_SIZE, self.hidden_size, device=config.device),
                torch.zeros(2, BATCH_SIZE, self.hidden_size, device=config.device))


class EncoderCRF(nn.Module):
    def __init__(self, hidden_size, output_size, context_dim, embed, trained_aspect):
        super(EncoderCRF, self).__init__()
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(hidden_size, 300, bidirectional=True, batch_first=True)
        self.embedded = nn.Embedding.from_pretrained(embed)
        self.aspect_embed = nn.Embedding.from_pretrained(trained_aspect)
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.context_input_ = nn.Linear(600, 50)
        self.embedding_layers = nn.Linear(0 + hidden_size, output_size)
        self.embedding_layer_s = nn.Linear(50 + hidden_size, output_size)
        self.classifier_layer = nn.Linear(output_size, 2)
        self.softmax = nn.Softmax(dim=2)
        # self.slf_attention = attention.MultiHeadAttention(600, 3)
        # self.slf_attention = attention.MultiHeadAttentionDotProduct(3, 600, 300, 300, 0.01)
        # self.Position_wise = attention.PositionwiseFeedForward(600, 600, 0.01)
        self.min_context = nn.Linear(300, 50)
        self.attention = attention.NormalAttention(600, 50, 50)
        self.gate = Gate.Gate(300, 50, 50, 300)
        self.dropout = nn.Dropout(config.dropout)
        self.crf = LinearCRF()
        self.feat2tri = nn.Linear(600, 2)

    def forward(self, input, hidden, style):
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

        # feats = self.feat2tri(desorted_output)

        if style == "train":
            sent, bestseqs = self.compute_scores(desorted_output)
        else:
            sent, bestseqs = self.compute_predict_scores(desorted_output)

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

        # desorted_output = F.max_pool2d(desorted_output, (desorted_output.size(1), 1))
        desorted_output = sent

        # output.view(self.hidden_size * 2, -1)
        # output = torch.max(output)
        # desorted_output = self.tanh(self.hidden_layer(desorted_output))
        desorted_output = self.hidden_layer(desorted_output)

        context_input = context_input.view(BATCH_SIZE, 1, 50)
        _context_input = self.tanh(self.context_input_(attn_target))

        gate_out = self.gate(desorted_output, _context_input, context_input)

        embedding_input = torch.cat((desorted_output, _context_input), dim=2)
        # desorted_output = self.tanh(self.dropout(self.embedding_layers(gate_out)))
        desorted_output = self.tanh(self.dropout(self.embedding_layer_s(embedding_input)))

        # result = self.softmax(self.classifier_layer(desorted_output))
        result = self.classifier_layer(desorted_output)
        return result

    def initHidden(self, BATCH_SIZE):
        return (torch.zeros(2, BATCH_SIZE, self.hidden_size, device=config.device),
                torch.zeros(2, BATCH_SIZE, self.hidden_size, device=config.device))

    def compute_scores(self, context):
        # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
        feat_context = context  # sent_len * dim_sum
        tri_scores = self.feat2tri(feat_context)
        marginals = self.crf(tri_scores)
        select_polarity = marginals[:, :, 1].unsqueeze(1)

        marginals = marginals.transpose(1, 2)  # 2 * sent_len
        sent_v = torch.bmm(select_polarity, context)  # 1 * feat_dim
        # label_scores = self.feat2label(sent_v).squeeze(0)

        return sent_v, marginals

    def compute_predict_scores(self, context):
        # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
        feat_context = context  # sent_len * dim_sum
        tri_scores = self.feat2tri(feat_context)
        marginals = self.crf(tri_scores)
        select_polarity = marginals[:, :, 1].unsqueeze(1)

        best_seqs = self.crf.predict(tri_scores)
        # best_seqs = 1

        sent_v = torch.bmm(select_polarity, context)  # 1 * feat_dim
        # print(best_seqs)

        return sent_v, best_seqs


class RealAspectExtract(nn.Module):
    def __init__(self, hidden_size, output_size, context_dim, embed, trained_aspect):
        super(RealAspectExtract, self).__init__()
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(hidden_size, 300, bidirectional=True, batch_first=True)
        self.embedded = nn.Embedding.from_pretrained(embed)
        self.aspect_embed = nn.Embedding.from_pretrained(trained_aspect)
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.context_input_ = nn.Linear(600, 50)
        self.embedding_layers = nn.Linear(0 + hidden_size, output_size)
        self.embedding_layer_s = nn.Linear(50 + hidden_size, output_size)
        self.classifier_layer = nn.Linear(output_size, 2)
        self.softmax = nn.Softmax(dim=2)
        # self.slf_attention = attention.MultiHeadAttention(600, 3)
        # self.slf_attention = attention.MultiHeadAttentionDotProduct(3, 600, 300, 300, 0.01)
        # self.Position_wise = attention.PositionwiseFeedForward(600, 600, 0.01)
        self.min_context = nn.Linear(300, 50)
        self.attention = attention.NormalAttention(600, 50, 50)
        self.gate = Gate.Gate(300, 50, 50, 300)
        self.dropout = nn.Dropout(config.dropout)
        self.crf = LinearCRF()
        self.feat2tri = nn.Linear(600, 2)
        self.crf2aspect = nn.Linear(600, 300)

    def forward(self, input, hidden, style):
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
        _, desorted_indices = torch.sort(indices, descending=False)
        input_index = input_index[:, 0: sorted_seq_lengths[0]]
        input_index = input_index[indices]
        input_value = self.embedded(input_index)

        """
        aspect context from pre train
        """
        context_input = self.aspect_embed(batch_context).float()

        context_input_norm = self.norm(context_input)
        input_value_norm = self.input_norm(input_value)

        sims = torch.matmul(input_value_norm.float(), context_input_norm.view(input_value.size()[0], -1, 1))

        context_cat = context_input_norm.unsqueeze(1).expand(-1, sorted_seq_lengths[0], -1)

        input_value = (context_cat * sims) + input_value.float()

        input_value = input_value.float()

        packed_inputs = nn.utils.rnn.pack_padded_sequence(input_value, sorted_seq_lengths.cpu().data.numpy()
                                                          , batch_first=True)

        # print(sorted_seq_lengths, indices)
        output, hidden = self.blstm(packed_inputs, hidden)
        padded_res, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        desorted_output = padded_res[desorted_indices]

        # feats = self.feat2tri(desorted_output)

        if style == "train":
            sent, bestseqs = self.compute_scores(desorted_output)
        else:
            sent, bestseqs = self.compute_predict_scores(desorted_output)

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
        sent_300d = self.crf2aspect(sent)

        context_input = self.min_context(sent_300d)

        attn_target = self.attention(desorted_output, context_input)

        desorted_output = F.max_pool2d(desorted_output, (desorted_output.size(1), 1))
        # desorted_output = sent

        # output.view(self.hidden_size * 2, -1)
        # output = torch.max(output)
        desorted_output = self.tanh(self.hidden_layer(desorted_output))
        # desorted_output = self.hidden_layer(desorted_output)

        context_input = context_input.view(BATCH_SIZE, 1, 50)
        _context_input = self.tanh(self.context_input_(attn_target))

        gate_out = self.gate(desorted_output, _context_input, context_input)

        # embedding_input = torch.cat((desorted_output, _context_input), dim=2)
        desorted_output = self.tanh(self.dropout(self.embedding_layers(gate_out)))
        # desorted_output = self.tanh(self.dropout(self.embedding_layer_s(embedding_input)))

        # result = self.softmax(self.classifier_layer(desorted_output))
        result = self.classifier_layer(desorted_output)
        return result

    def initHidden(self, BATCH_SIZE):
        return (torch.zeros(2, BATCH_SIZE, self.hidden_size, device=config.device),
                torch.zeros(2, BATCH_SIZE, self.hidden_size, device=config.device))

    def compute_scores(self, context):
        # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
        feat_context = context  # sent_len * dim_sum
        tri_scores = self.feat2tri(feat_context)
        marginals = self.crf(tri_scores)
        select_polarity = marginals[:, :, 1].unsqueeze(1)

        marginals = marginals.transpose(1, 2)  # 2 * sent_len
        sent_v = torch.bmm(select_polarity, context)  # 1 * feat_dim
        # label_scores = self.feat2label(sent_v).squeeze(0)

        return sent_v, marginals

    def compute_predict_scores(self, context):
        # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
        feat_context = context  # sent_len * dim_sum
        tri_scores = self.feat2tri(feat_context)
        marginals = self.crf(tri_scores)
        select_polarity = marginals[:, :, 1].unsqueeze(1)

        best_seqs = self.crf.predict(tri_scores)
        # best_seqs = 1

        sent_v = torch.bmm(select_polarity, context)  # 1 * feat_dim
        # print(best_seqs)

        return sent_v, best_seqs

    def norm(self, aspect_context, eps=config.epsilon):
        div = eps + torch.norm(aspect_context, 2, -1)
        div = div.view(-1, 1)
        aspect_context = aspect_context / div
        return aspect_context

    def input_norm(self, context, eps=config.epsilon):
        div = eps + torch.norm(context, 2, -1)
        div = div.view(context.size()[0], -1, 1)
        context = context / div
        return context


class CrfWdeRnnEncoder(nn.Module):
    def __init__(self, hidden_size, output_size, context_dim, embed, trained_aspect):
        super(CrfWdeRnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(hidden_size, 300, bidirectional=True, batch_first=True)
        self.embedded = nn.Embedding.from_pretrained(embed)
        self.aspect_embed = nn.Embedding.from_pretrained(trained_aspect)
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.context_input_ = nn.Linear(600, 50)
        self.embedding_layers = nn.Linear(0 + hidden_size, output_size)
        self.classifier_layer = nn.Linear(output_size, 2)
        self.softmax = nn.Softmax(dim=2)
        # self.slf_attention = attention.MultiHeadAttention(600, 3)
        # self.slf_attention = attention.MultiHeadAttentionDotProduct(3, 600, 300, 300, 0.01)
        # self.Position_wise = attention.PositionwiseFeedForward(600, 600, 0.01)
        self.min_context = nn.Linear(300, 50)
        self.attention = attention.NormalAttention(600, 50, 50)
        self.gate = Gate.Gate(300, 50, 50, 300)
        self.dropout = nn.Dropout(config.dropout)
        self.crf = LinearCRF()
        self.feat2tri = nn.Linear(600, 2)
        self.crf2aspect = nn.Linear(600, 300)

    def forward(self, input, hidden, update_aspect, style):
        BATCH_SIZE = len(input)
        self.aspect_embed.weight.data.copy_(update_aspect)
        batch_len = input[:, 0]
        batch_context = input[:, 1]
        input_index = input[:, 2:]
        input_index = input_index.long()
        # seq_len = batch_len.item()
        # input_index = input_index[0][0:seq_len]
        # print('input_index',input_index)
        # print(hidden.size())
        sorted_seq_lengths, indices = torch.sort(batch_len, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        input_index = input_index[:, 0: sorted_seq_lengths[0]]
        input_index = input_index[indices]
        input_value = self.embedded(input_index)

        context_input = self.aspect_embed(batch_context).float()

        context_input_norm = self.norm(context_input)
        input_value_norm = self.input_norm(input_value)

        sims = torch.matmul(input_value_norm.float(), context_input_norm.view(input_value.size()[0], -1, 1))

        context_cat = context_input_norm.unsqueeze(1).expand(-1, sorted_seq_lengths[0], -1)

        input_value = (context_cat * sims) + input_value.float()

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

        if style == "train":
            sent, bestseqs = self.compute_scores(desorted_output)
        else:
            sent, bestseqs = self.compute_predict_scores(desorted_output)

        '''
        Normal attention module add or not?
        '''
        sent_300d = self.crf2aspect(sent)

        context_input = self.min_context(sent_300d)

        attn_target = self.attention(desorted_output, context_input)

        desorted_output = F.max_pool2d(desorted_output, (desorted_output.size(1), 1))

        # output.view(self.hidden_size * 2, -1)
        # output = torch.max(output)
        desorted_output = self.tanh(self.hidden_layer(desorted_output))

        context_input = context_input.view(BATCH_SIZE, 1, 50)
        _context_input = self.tanh(self.context_input_(attn_target))

        gate_out = self.gate(desorted_output, _context_input, context_input)

        embedding_input = torch.cat((desorted_output, _context_input), dim=2)
        desorted_output = self.tanh(self.dropout(self.embedding_layers(gate_out)))

        # result = self.softmax(self.classifier_layer(desorted_output))
        result = self.classifier_layer(desorted_output)
        return result

    def initHidden(self, BATCH_SIZE):
        return (torch.zeros(2, BATCH_SIZE, self.hidden_size, device=config.device),
                torch.zeros(2, BATCH_SIZE, self.hidden_size, device=config.device))

    def compute_scores(self, context):
        # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
        feat_context = context  # sent_len * dim_sum
        tri_scores = self.feat2tri(feat_context)
        marginals = self.crf(tri_scores)
        select_polarity = marginals[:, :, 1].unsqueeze(1)

        marginals = marginals.transpose(1, 2)  # 2 * sent_len
        sent_v = torch.bmm(select_polarity, context)  # 1 * feat_dim
        # label_scores = self.feat2label(sent_v).squeeze(0)

        return sent_v, marginals

    def compute_predict_scores(self, context):
        # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
        feat_context = context  # sent_len * dim_sum
        tri_scores = self.feat2tri(feat_context)
        marginals = self.crf(tri_scores)
        select_polarity = marginals[:, :, 1].unsqueeze(1)

        best_seqs = self.crf.predict(tri_scores)
        # best_seqs = 1

        sent_v = torch.bmm(select_polarity, context)  # 1 * feat_dim
        # print(best_seqs)

        return sent_v, best_seqs

    def norm(self, aspect_context, eps=config.epsilon):
        div = eps + torch.norm(aspect_context, 2, -1)
        div = div.view(-1, 1)
        aspect_context = aspect_context / div
        return aspect_context

    def input_norm(self, context, eps=config.epsilon):
        div = eps + torch.norm(context, 2, -1)
        div = div.view(context.size()[0], -1, 1)
        context = context / div
        return context
