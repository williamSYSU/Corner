import torch
import torch.nn as nn
import torch.nn.functional as f

import config
from layers.attention import DotProductAttention


class ABAE(nn.Module):
    def __init__(self, aspect_embedding, embed):
        super(ABAE, self).__init__()
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

    def forward(self, inputs, neg_samples, eps=config.epsilon):
        input_lengths = inputs[:, 0]
        inputs = inputs[:, 2:]
        input_index = inputs.long()
        sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        input_index = input_index[:, 0: sorted_seq_lengths[0]]
        input_index = input_index[indices]
        inputs = self.embedded(input_index).double()

        neg_lengths = neg_samples[:, :, 0].view(config.batch_size * config.neg_size)
        neg_inputs = neg_samples[:, :, 2:]
        neg_inputs = neg_inputs.view(config.batch_size * config.neg_size, 300)
        neg_inputs = self.embedded(neg_inputs).double()

        neg_avg_denominator = neg_lengths.repeat(self.embed_dim).view(self.embed_dim, -1).transpose(0, 1).float()
        neg_global_content_embed = torch.sum(neg_inputs.double(), dim=1).div(neg_avg_denominator.double())
        neg_inputs = neg_global_content_embed.view(config.batch_size, config.neg_size, 300)

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
        aspect_weight = f.softmax(self.aspect_linear(sentence_embedding), dim=1)

        # [batch_size, n_aspect] * [n_aspect, embed_dim] = [batch_size, embed_dim]
        reconstruct_embedding = torch.matmul(aspect_weight.double(), self.aspect_lookup_mat)

        # div = eps + torch.sqrt(torch.sum(self.aspect_lookup_mat ** 2, dim=-1))

        # self.regular()

        '''regularization'''
        div = eps + torch.norm(self.aspect_lookup_mat, 2, -1)
        div = div.view(-1, 1)

        aspect_matrix = self.aspect_lookup_mat / div
        reg = torch.sum(torch.matmul(aspect_matrix, aspect_matrix.permute(1, 0)) ** 2 -
                        torch.eye(24).double().to(config.device))

        return reconstruct_embedding, sentence_embedding, neg_inputs, reg.float()

    def regular(self, eps=config.epsilon):
        div = eps + torch.norm(self.aspect_lookup_mat, 2, -1)
        div = div.view(-1, 1)
        self.aspect_lookup_mat.data = self.aspect_lookup_mat / div
