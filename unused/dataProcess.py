# from __future__ import unicode_literals, print_function, division
# from io import open
# import unicodedata
# import re
# # import gensim
# import string
# import random
# import numpy as np
# import torch
# # from spellchecker import SpellChecker
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import torch.nn as nn
# import models
# import time
# import re
# import numpy as np
# import dataprepare
# import argparse
#
#
# class DataProcess(Dataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def __len__(self):
#         return len(self.data)
#
#
# def weakly_data_process():
#     print("=" * 100)
#     print("Weakly data Process...")
#
#     vocab, pairs_pos, pairs_neg = dataprepare.weakly_data(opt)
#
#     final_embedding = np.array(np.load("embed/Vector_word_embedding_all.npy"))
#
#     maxlen = 0
#     bb = []
#     #
#     # def word2idx(sentence, vocab, maxlen, bb):
#     #     items = sentence.strip().split()
#     #     if len(items) > maxlen:
#     #         maxlen = len(items)
#     #         bb = items
#     #     for word in items:
#     #         if word not in vocab:
#     #             vocab[word] = len(vocab)
#     #     return maxlen, bb
#     #
#     # for line in pairs_pos:
#     #     maxlen, bb = word2idx(line, vocab, maxlen, bb)
#     #
#     # for line in pairs_neg:
#     #     maxlen, bb = word2idx(line, vocab, maxlen, bb)
#
#     input_sentence_1 = config.pad_idx + np.zeros((len(pairs_pos), config.maxlen))
#     input_sentence_1 = input_sentence_1.astype(np.int)
#     input_sentence_2 = config.pad_idx + np.zeros((len(pairs_neg), config.maxlen))
#     input_sentence_2 = input_sentence_2.astype(np.int)
#
#     if opt.need_pos is True:
#         input_sentence_1 = config.pad_idx + np.zeros((len(pairs_pos), 2 * config.maxlen))
#         input_sentence_1[:, config.maxlen:] = 300 + np.zeros((len(pairs_pos), config.maxlen))
#         input_sentence_1 = input_sentence_1.astype(np.int)
#         input_sentence_2 = config.pad_idx + np.zeros((len(pairs_neg), 2 * config.maxlen))
#         input_sentence_2[:, config.maxlen:] = 300 + np.zeros((len(pairs_neg), config.maxlen))
#         input_sentence_2 = input_sentence_2.astype(np.int)
#
#     def sentence2vec(sentence, vocab, wordindex):
#         items = sentence.strip().split()
#         length = len(items)
#         for word in items:
#             wordindex.append(vocab[word])
#         return length
#
#     def cal_sentence_index():
#         for line in range(len(pairs_pos)):
#             wordindex = []
#             length = sentence2vec(pairs_pos[line], vocab, wordindex)
#             input_sentence_1[line][0] = length
#             input_sentence_1[line][1] = 10
#             input_sentence_1[line][2:length + 2] = np.array(wordindex)
#             if opt.need_pos is True:
#                 input_sentence_1[line][config.maxlen:length + config.maxlen] = [x for x in range(length)]
#
#         for line in range(len(pairs_neg)):
#             wordindex = []
#             length = sentence2vec(pairs_neg[line], vocab, wordindex)
#             input_sentence_2[line][0] = length
#             input_sentence_2[line][1] = 10
#             input_sentence_2[line][2:length + 2] = np.array(wordindex)
#             if opt.need_pos is True:
#                 input_sentence_2[line][config.maxlen:length + config.maxlen] = [x for x in range(length)]
#         return input_sentence_1, input_sentence_2
#
#     cal_sentence_index()
#
#     # add = -1 + 2*np.random.random(300)
#     add = np.zeros(300)
#     final_embedding = np.row_stack((final_embedding, add))
#
#     np.random.shuffle(input_sentence_1)
#     np.random.shuffle(input_sentence_2)
#
#     input_pos_train = input_sentence_1[:int(len(input_sentence_1) * opt.weak_sr), :]
#     input_neg_train = input_sentence_2[:int(len(input_sentence_2) * opt.weak_sr), :]
#     # print(input_sentence_1[0])
#
#     input_pos_test = input_sentence_1[int(len(input_sentence_1) * opt.weak_sr):, :]
#     input_neg_test = input_sentence_2[int(len(input_sentence_2) * opt.weak_sr):, :]
#
#     def random_sample(matrix, sample_size):
#         matrix_after = []
#         sample_index = np.random.randint(0, len(matrix), sample_size)
#         for i in sample_index:
#             # np.row_stack((matrix_after, matrix[i]))
#             matrix_after.append(matrix[i])
#         return np.array(matrix_after)
#
#     train_pos_1 = random_sample(input_pos_train, opt.sample_size)
#     train_pos_2 = random_sample(input_pos_train, opt.sample_size)
#     train_pos_neg = random_sample(input_neg_train, opt.sample_size)
#     train_neg_1 = random_sample(input_neg_train, opt.sample_size)
#     train_neg_2 = random_sample(input_neg_train, opt.sample_size)
#     train_neg_pos = random_sample(input_pos_train, opt.sample_size)
#
#     train_dim1 = np.vstack((train_pos_1, train_neg_1))
#     train_dim2 = np.vstack((train_pos_2, train_neg_2))
#     train_dim3 = np.vstack((train_pos_neg, train_neg_pos))
#
#     def read_data(dim_1, dim_2, dim_3):
#         all_data = []
#         for idx in range(len(dim_1)):
#             items = torch.from_numpy(dim_1[idx])
#             items1 = torch.from_numpy(dim_2[idx])
#             items2 = torch.from_numpy(dim_3[idx])
#             data = {
#                 'input1': items,
#                 'input2': items1,
#                 'input3': items2
#             }
#             all_data.append(data)
#         return all_data
#
#     all_data = DataProcess(read_data(train_dim1, train_dim2, train_dim3))
#     return all_data, final_embedding, np.array(input_pos_test[0:opt.weak_test_samples, :]), np.array(
#         input_neg_test[0:opt.weak_test_samples, :])
#
#
# def classification_process(opt):
#     print("=" * 100)
#     print("Classification data Process...")
#
#     vocab, pairs_classifier = dataprepare.classification_data(opt)
#     final_embedding = np.array(np.load("embed/Vector_word_embedding_all.npy"))
#     # final_embedding = np.delete(final_embedding, 60905, 0)
#     # print(final_embedding[1708])
#
#     maxlen = 0
#     bb = []
#
#     # models = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin', binary=True)
#
#     def word2idx(sentence, vocab, maxlen, bb):
#         items = sentence.strip().split()
#         if len(items) > maxlen:
#             maxlen = len(items)
#             bb = items
#         for word in items[2:]:
#             if word not in vocab:
#                 vocab[word] = len(vocab)
#         return maxlen, bb
#
#     for line in pairs_classifier:
#         maxlen, bb = word2idx(line, vocab, maxlen, bb)
#
#     # print(vocab)
#
#     input_sentence = config.pad_idx + np.zeros((len(pairs_classifier), 204))
#     input_sentence = input_sentence.astype(np.int)
#
#     if opt.need_pos is True:
#         input_sentence = config.pad_idx + np.zeros((len(pairs_classifier), 408))
#         input_sentence[:, 204:] = 200 + np.zeros((len(pairs_classifier), 204))
#         input_sentence = input_sentence.astype(np.int)
#
#     def sentence2vec(sentence, vocab, wordindex):
#         items = sentence.strip().split()
#         label = 0
#         obj = 0
#         if items[0] == "positive":
#             label = 1
#         if items[1] == "objective":
#             obj = 1
#         length = len(items) - 2
#
#         for word in items[2:]:
#             wordindex.append(vocab[word])
#         return length, label, obj
#
#     pos_sentence = []
#     neg_sentence = []
#
#     def cal_sentence_index():
#         for line in range(len(pairs_classifier)):
#             wordindex = []
#             length, label, obj = sentence2vec(pairs_classifier[line], vocab, wordindex)
#             input_sentence[line][2] = length
#             input_sentence[line][3] = 10
#             input_sentence[line][0] = label
#             input_sentence[line][1] = obj
#             input_sentence[line][4:length + 4] = np.array(wordindex)
#             if opt.need_pos is True:
#                 input_sentence[line][204:length + 204] = [x for x in range(length)]
#         for sentence in input_sentence:
#             if sentence[0] == 1:
#                 pos_sentence.append(sentence)
#             else:
#                 neg_sentence.append(sentence)
#         return input_sentence, pos_sentence, neg_sentence
#
#     cal_sentence_index()
#     # print(input_sentence[0])
#     pos_sentence = np.array(pos_sentence)
#     neg_sentence = np.array(neg_sentence)
#     np.random.seed(1)
#     np.random.shuffle(pos_sentence)
#     np.random.seed(1)
#     np.random.shuffle(neg_sentence)
#
#     pos_sentence_train = pos_sentence[:int(len(pos_sentence) * opt.clas_sr), :]
#     pos_sentence_valid = pos_sentence[int(len(pos_sentence) * opt.clas_sr): int(
#         len(pos_sentence) * (opt.clas_sr + 1) / 2), :]
#     pos_sentence_test = pos_sentence[int(len(pos_sentence) * (opt.clas_sr + 1) / 2):, :]
#
#     neg_sentence_train = neg_sentence[:int(len(neg_sentence) * opt.clas_sr), :]
#     neg_sentence_valid = neg_sentence[int(len(neg_sentence) * opt.clas_sr): int(
#         len(neg_sentence) * (opt.clas_sr + 1) / 2), :]
#     neg_sentence_test = neg_sentence[int(len(neg_sentence) * (opt.clas_sr + 1) / 2):, :]
#
#     # input_train = np.array(input_sentence[:int(len(input_sentence) * 0.7), :])
#     # input_test = np.array(input_sentence[int(len(input_sentence) * 0.7):, :])
#     input_train = np.vstack((pos_sentence_train, neg_sentence_train))
#     input_valid = np.vstack((pos_sentence_valid, neg_sentence_valid))
#     input_test = np.vstack((pos_sentence_test, neg_sentence_test))
#
#     np.random.shuffle(input_train)
#     np.random.shuffle(input_valid)
#     np.random.shuffle(input_test)
#
#     # add = -1 + 2 * np.random.random(300)
#
#     add = np.zeros(300)
#     final_embedding = np.row_stack((final_embedding, add))
#
#     def read_data_train(input_train, label):
#         all_data = []
#         for idx in range(len(input_train)):
#             items = torch.from_numpy(input_train[idx])
#             items1 = torch.tensor(int(label[idx]))
#             data = {
#                 'input': items,
#                 'label': items1,
#             }
#             all_data.append(data)
#         return all_data
#
#     def read_data_valid(input_test, label):
#         all_data = []
#         for idx in range(len(input_test)):
#             items = torch.from_numpy(input_test[idx])
#             items1 = torch.tensor(int(label[idx]))
#             data = {
#                 'input': items,
#                 'label': items1,
#             }
#             all_data.append(data)
#         return all_data
#
#     def read_data_test(input_test, label):
#         all_data = []
#         for idx in range(len(input_test)):
#             items = torch.from_numpy(input_test[idx])
#             items1 = torch.tensor(int(label[idx]))
#             data = {
#                 'input': items,
#                 'label': items1,
#             }
#             all_data.append(data)
#         return all_data
#
#     train_data = DataProcess(read_data_train(input_train[:, 2:], input_train[:, 0]))
#     valid_data = DataProcess(read_data_valid(input_valid[:, 2:], input_valid[:, 0]))
#     test_data = DataProcess(read_data_test(input_test[:, 2:], input_test[:, 0]))
#     return train_data, valid_data, test_data, final_embedding
#
#
# def save_vector(opt):
#
#     lines = open('embed/word_embedding_classifier.txt',
#                  encoding='utf-8').read().strip().split('\n')
#
#     a = np.zeros((8986, 300))
#
#     def cut_space(sentence):
#         out = re.sub(r"\s{2,}", " ", sentence)
#         return out
#
#     pairs = [[s for s in l.split(',')] for l in lines]
#     for i in range(len(pairs)):
#         if len(pairs[i][1]) > 300:
#             a[i] = np.array(list(map(eval, cut_space(pairs[i][1]).strip().split())))
#             # a[i] = map(eval, a[i])
#         else:
#             a[i] = -1 + 2 * np.random.random(300)
#
#     t = -1 + 2 * np.random.random(300)
#     # c = np.insert(a, 12586, values=t, axis=0)
#     np.save("embed/Vector_classifier.npy", a)
#
#     # b = [[s for s in l.split()] for l in a]
#
#     # print(b[0])
#
#
# def aspect_extract_data_process(opt):
#     print("=" * 100)
#     print("Aspect Extract data Process...")
#
#     vocab, pairs_all = dataprepare.aspect_extract_data(opt)
#     final_embedding = np.array(np.load("embed/Vector_word_embedding_all.npy"))
#     # final_embedding = np.delete(final_embedding, 60905, 0)
#     # print(final_embedding[1708])
#
#     maxlen = 0
#     bb = []
#
#     # models = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin', binary=True)
#
#     def word2idx(sentence, vocab, maxlen, bb):
#         items = sentence.strip().split()
#         if len(items) > maxlen:
#             maxlen = len(items)
#             bb = items
#         for word in items[2:]:
#             if word not in vocab:
#                 vocab[word] = len(vocab)
#         return maxlen, bb
#
#     # for line in pairs_all:
#     #     maxlen, bb = word2idx(line, vocab, maxlen, bb)
#
#     # print(vocab)
#
#     input_sentence = config.pad_idx + np.zeros((len(pairs_all), config.maxlen))
#     input_sentence = input_sentence.astype(np.int)
#
#     if opt.need_pos is True:
#         input_sentence = config.pad_idx + np.zeros((len(pairs_all), 2 * config.maxlen))
#         input_sentence[:, config.maxlen:] = 300 + np.zeros((len(pairs_all), config.maxlen))
#         input_sentence = input_sentence.astype(np.int)
#
#     def sentence2vec(sentence, vocab, wordindex):
#         items = sentence.strip().split()
#         length = len(items)
#         for word in items:
#             wordindex.append(vocab[word])
#         return length
#
#     def cal_sentence_index():
#         for line in range(len(pairs_all)):
#             wordindex = []
#             length = sentence2vec(pairs_all[line], vocab, wordindex)
#             input_sentence[line][0] = length
#             input_sentence[line][1] = 10
#             input_sentence[line][2:length + 2] = np.array(wordindex)
#             # if input_sentence[line][2] == config.pad_idx:
#             #     print("there is a 0")
#             #     print(input_sentence[line][0])
#
#             if opt.need_pos is True:
#                 input_sentence[line][config.maxlen:length + config.maxlen] = [x for x in range(length)]
#         return input_sentence
#
#     cal_sentence_index()
#     # print(input_sentence[0])
#     input_sentence = np.array(input_sentence)
#
#     np.random.shuffle(input_sentence)
#
#     # add = -1 + 2 * np.random.random(300)
#     add = np.zeros(300)
#     final_embedding = np.row_stack((final_embedding, add))
#
#     def read_data_train(input_train):
#         all_data = []
#         for idx in range(len(input_train)):
#             items = torch.from_numpy(input_train[idx])
#             data = {
#                 'input': items
#             }
#             all_data.append(data)
#         return all_data
#
#     train_data = DataProcess(read_data_train(input_sentence))
#     return train_data, final_embedding, input_sentence
