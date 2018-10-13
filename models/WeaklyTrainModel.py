import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import layers.Gate as Gate
import layers.attention_layer as attention
from layers.attention import DotProductAttention

# MAX_LENGTH = 170
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# final_embedding = np.array(np.load("embed/Vector1.npy"))
# embed = torch.from_numpy(final_embedding)


class WdeCnn(nn.Module):
    def __init__(self, vector_size, hidden_dim, context_dim, dropout_p=0.1):
        super(WdeCnn, self).__init__()
        self.conv2d_h1 = nn.Conv2d(1, 200, (1, vector_size))
        self.conv2d_h2 = nn.Conv2d(1, 200, (2, vector_size))
        self.conv2d_h3 = nn.Conv2d(1, 200, (3, vector_size))
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding.from_pretrained(embed)
        self.hidden_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.context_input = nn.Linear(context_dim, 100)
        self.embedding_layer = nn.Linear(100 + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_dim + 100, hidden_dim)

    def forward(self, input, context_input):
        batch_len = input[:, 0]
        batch_context = input[:, 1]
        input_index = input[:, 2:]
        input_index = input_index.long()
        # seq_len = batch_len.item()
        # input_index = input_index[0][0:seq_len]
        # print('input_index',input_index)
        input_value = self.embedding(input_index).view(BATCH_SIZE, 1, MAX_LENGTH, 300).float()
        # print(input_value.size())

        input_h1 = self.tanh(self.conv2d_h1(input_value))
        input_h2 = self.tanh(self.conv2d_h2(input_value))
        input_h3 = self.tanh(self.conv2d_h3(input_value))

        input_h1 = F.max_pool2d(input_h1, (MAX_LENGTH, 1))
        input_h2 = F.max_pool2d(input_h2, (MAX_LENGTH - 2 + 1, 1))
        input_h3 = F.max_pool2d(input_h3, (MAX_LENGTH - 3 + 1, 1))

        output_h1 = input_h1.view(BATCH_SIZE, 1, -1)
        output_h2 = input_h2.view(BATCH_SIZE, 1, -1)
        output_h3 = input_h3.view(BATCH_SIZE, 1, -1)

        output = torch.cat((output_h1, output_h2), dim=2)
        output = torch.cat((output, output_h3), dim=2)
        output = self.tanh(self.hidden_layer(output))
        context_input = context_input.view(BATCH_SIZE, 1, 50)
        context_input = self.tanh(self.context_input(context_input))
        embedding_input = torch.cat((output, context_input), dim=2)
        output = self.out(embedding_input)

        return output


class WdeRnnEncoder(nn.Module):
    def __init__(self, hidden_size, output_size, context_dim, embed):
        super(WdeRnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(hidden_size, 300, bidirectional=True, batch_first=True)
        self.embedding = nn.Embedding.from_pretrained(embed)
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.context_input = nn.Linear(context_dim, 50)
        self.embedding_layer = nn.Linear(50 + hidden_size, output_size)

    def forward(self, input, hidden, context_input):
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
        input_index = input_index[:, 0: sorted_seq_lengths[0]]
        _, desorted_indices = torch.sort(indices, descending=False)
        input_index = input_index[indices]
        input_value = self.embedding(input_index).float()
        packed_inputs = nn.utils.rnn.pack_padded_sequence(input_value, sorted_seq_lengths.cpu().data.numpy(),
                                                          batch_first=True)

        # print(sorted_seq_lengths, indices)
        output, hidden = self.blstm(packed_inputs, hidden)
        padded_res, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        desorted_output = padded_res[desorted_indices]
        desorted_output = F.max_pool2d(desorted_output, (desorted_output.size(1), 1))

        # output.view(self.hidden_size * 2, -1)
        # output = torch.max(output)
        desorted_output = self.tanh(self.hidden_layer(desorted_output))

        context_input = context_input.view(BATCH_SIZE, 1, 50)
        context_input = self.tanh(self.context_input(context_input))

        embedding_input = torch.cat((desorted_output, context_input), dim=2)
        desorted_output = self.tanh(self.embedding_layer(embedding_input))
        return desorted_output

    def initHidden(self, BATCH_SIZE):
        return (torch.zeros(2, BATCH_SIZE, self.hidden_size, device=device),
                torch.zeros(2, BATCH_SIZE, self.hidden_size, device=device))


class WdeRnnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, context_dim):
        super(WdeRnnDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.context_input = nn.Linear(context_dim, 100)
        self.embedding_layer = nn.Linear(100 + hidden_size, output_size)

    def forward(self, input, context):
        input.view(self.hidden_size * 2, -1)
        output = torch.max(input)
        output.view(1, -1)
        output = self.tanh(self.hidden_layer(input))
        output = self.tanh(self.context_input(output))
        output = self.tanh(self.embedding_layer(output))

        return output

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)


class SoftMaxOutput(nn.Module):
    def __init__(self, hidden_size):
        super(SoftMaxOutput, self).__init__()

        self.embedding_layer = nn.Linear(hidden_size, 150)
        self.Classification_layer = nn.Linear(150, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, model):
        output = model(input)
        output = self.embedding_layer(output)
        output = self.Classification_layer(output)
        output = self.softmax(output)
        return output


class myloss(nn.Module):
    def __init__(self, batch_size):
        super(myloss, self).__init__()
        self.batch = batch_size

    def forward(self, out1, out2, out3):
        for idx, (out_1, out_2, out_3) in enumerate(zip(out1, out2, out3)):
            if idx is 0:
                loss_part1 = torch.dist(out_1, out_3, 2).view(-1)
                loss_part2 = torch.dist(out_1, out_2, 2).view(-1)
            else:
                loss_part1 = torch.cat((loss_part1, torch.dist(out_1, out_3, 2).view(-1)), dim=0)
                loss_part2 = torch.cat((loss_part2, torch.dist(out_1, out_2, 2).view(-1)), dim=0)

        compare = 2 - loss_part1 + loss_part2
        loss_last = torch.sum(torch.max(torch.zeros(self.batch, 1), compare))
        return loss_last


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
        # self.classifier_layer = nn.Linear(output_size, 2)
        # self.softmax = nn.Softmax(dim=2)
        # self.slf_attention = attention.MultiHeadAttention(600, 3)
        self.Position_wise = attention.PositionwiseFeedForward(300, 300)
        self.attention = attention.NormalAttention(300, 50, 50)
        self.gate = Gate.Gate(300, 50, 50, 300)
        # self.dropout = nn.Dropout(config.dropout)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(301, 300, padding_idx=300),
            freeze=True)

    def forward(self, input, context_input):
        BATCH_SIZE = len(input)
        batch_len = input[:, 0]
        batch_context = input[:, 1]
        input_index = input[:, 2: config.maxlen]
        input_pos = input[:, config.maxlen:]
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
        desorted_output = self.tanh((self.embedding_layers(gate_out)))
        return desorted_output


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


class WdeRnnEncoderFix(nn.Module):
    def __init__(self, hidden_dim, output_size, context_dim, embed, trained_aspect):
        super(WdeRnnEncoderFix, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = config.embed_dim
        self.blstm = nn.LSTM(self.embed_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.embedded = nn.Embedding.from_pretrained(embed)
        self.aspect_embed = nn.Embedding.from_pretrained(trained_aspect)
        self.tanh = nn.Tanh()
        self.hidden_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.context_input_ = nn.Linear(600, 50)
        self.embedding_layers = nn.Linear(0 + hidden_dim, output_size)
        # self.slf_attention = attention.MultiHeadAttention(600, 3)
        # self.slf_attention = attention.MultiHeadAttentionDotProduct(3, 600, 300, 300, 0.01)
        # self.Position_wise = attention.PositionwiseFeedForward(600, 600, 0.01)
        self.attention = attention.NormalAttention(600, 50, 50)
        self.gate = Gate.Gate(300, 50, 50, 300)
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
        return (torch.zeros(2, BATCH_SIZE, self.hidden_dim, device=config.device),
                torch.zeros(2, BATCH_SIZE, self.hidden_dim, device=config.device))


class PreTrainABAE(nn.Module):
    def __init__(self, aspect_embedding, embed):
        super(PreTrainABAE, self).__init__()
        self.embedded = nn.Embedding.from_pretrained(embed)
        self.embed_dim = config.embed_dim
        self.n_aspect = config.n_aspect
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

        return predicted

    def regular(self, eps=config.epsilon):
        div = eps + torch.norm(self.aspect_lookup_mat, 2, -1)
        div = div.view(-1, 1)
        self.aspect_lookup_mat.data = self.aspect_lookup_mat / div
