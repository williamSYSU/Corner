# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : Corner-william
# @FileName     : data_util.py
# @Time         : Created at 2018/10/11
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

from __future__ import unicode_literals, print_function, division

import time
import unicodedata
from tqdm import tqdm

import numpy as np
import re
import torch
from io import open
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordParser
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree

import config


# from spellchecker import SpellChecker

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DataPrepare:
    def __init__(self):
        """load data from files"""
        print("=" * 100)
        print("Prepare training data...")
        # now1 = time.time()
        # lines_pos1 = open('data/Weakly_labeled_data_1.1M/camera_positive.csv',
        #                   encoding='utf-8').read().strip().split('\n')
        # lines_pos2 = open('data/Weakly_labeled_data_1.1M/cellphone_positive.csv',
        #                   encoding='utf-8').read().strip().split('\n')
        # lines_pos3 = open('data/Weakly_labeled_data_1.1M/laptop_positive.csv',
        #                   encoding='utf-8').read().strip().split('\n')
        # lines_neg1 = open('data/Weakly_labeled_data_1.1M/camera_negative.csv',
        #                   encoding='utf-8').read().strip().split('\n')
        # lines_neg2 = open('data/Weakly_labeled_data_1.1M/cellphone_negative.csv',
        #                   encoding='utf-8').read().strip().split('\n')
        # lines_neg3 = open('data/Weakly_labeled_data_1.1M/laptop_negative.csv',
        #                   encoding='utf-8').read().strip().split('\n')
        #
        # lines = open('data/Labeled_data_11754/new_11754.csv',
        #              encoding='gbk').read().strip().split('\n')
        # '''merge data'''
        # lines_pos = lines_pos1 + lines_pos2 + lines_pos3
        # lines_neg = lines_neg1 + lines_neg2 + lines_neg3
        # lines_all = lines_pos + lines_neg

        '''normalize sentences'''
        # print('costs {}s'.format(int(time.time() - now1)))

        '''load normalize sentences'''
        # self.pairs_all = [self.normalizeString(s) for s in tqdm(lines_all)]
        # self.pairs_pos = [self.normalizeString(s) for s in tqdm(lines_pos)]
        # self.pairs_neg = [self.normalizeString(s) for s in tqdm(lines_neg)]
        # self.pairs_classifier = [self.normalizeString(s) for s in tqdm(lines)]
        # self.pairs_all = open('data/nor_data/nor_all.csv', 'r').read().strip().split('\n')
        tmp = open('data/weak_comb_data/comb_pos{}.csv'.format('_retain' if config.if_retain else ''),
                   'r').read().strip().split('\n')
        self.pairs_pos = [tmp[i] for i in range(len(tmp)) if i % 2 == 0]
        self.asp_pos = [tmp[i] for i in range(len(tmp)) if i % 2 == 1]

        tmp = open('data/weak_comb_data/comb_neg{}.csv'.format('_retain' if config.if_retain else ''),
                   'r').read().strip().split('\n')
        self.pairs_neg = [tmp[i] for i in range(len(tmp)) if i % 2 == 0]
        self.asp_neg = [tmp[i] for i in range(len(tmp)) if i % 2 == 1]

        # tmp = open('data/clas_aspect.csv', 'r').read().strip().split('\n')
        tmp = open('data/weak_comb_data/comb_clas.csv'.format('_retain' if config.if_retain else ''),
                   'r').read().strip().split('\n')
        self.pairs_clas = [tmp[i] for i in range(len(tmp)) if i % 2 == 0]
        self.asp_clas = [tmp[i] for i in range(len(tmp)) if i % 2 == 1]

        # self.vocab = {}
        self.vocab = self.load_word_vocab()  # load word vocab from file

        '''count maxlen and obtain bb, build vocab'''
        # self.maxlen = 0
        # self.max_items = []
        # for line in self.pairs_clas:
        #     self.maxlen, self.max_items = self.word2idx(line, self.maxlen, self.max_items)
        #
        # for line in self.pairs_pos:
        #     self.maxlen, self.max_items = self.word2idx(line, self.maxlen, self.max_items)
        #
        # for line in self.pairs_neg:
        #     self.maxlen, self.max_items = self.word2idx(line, self.maxlen, self.max_items)

        # if save word vocab
        # self.save_word_vocab()
        # print('saved vocab done!')

    @property
    def weakly_data(self):
        """get weakly train data"""
        '''if pre-save vectors from Google News'''
        if config.pp_data_weak:
            save_name = 'embed\embedding\word_embedding_classifier.txt'
            self.saveVocab(save_name)

        return self.vocab, self.pairs_pos, self.pairs_neg, self.asp_pos, self.asp_neg

    @property
    def classification_data(self):
        """get classification train data"""
        '''if pre-save vectors from Google News'''
        if config.pp_data_clas:
            save_name = 'embed\embedding\word_embedding_classifier.txt'
            self.saveVocab(save_name)

        return self.vocab, self.pairs_clas, self.asp_clas

    @property
    def weakly_data_test(self):
        """get weakly test data"""
        count = 0

        def word2idx(sentence, vocab, maxlen, max_items, count):
            items = sentence.strip().split()
            if len(items) > maxlen:
                maxlen = len(items)
                max_items = items
            for word in items:
                if word not in vocab:
                    vocab[word] = len(vocab)
            if len(items) > 200:
                count = 1 + count
            return count, maxlen, max_items

        for line in self.pairs_clas:
            count, self.maxlen, self.max_items = word2idx(line, self.vocab, self.maxlen, self.max_items, count)

        for line in self.pairs_pos:
            count, self.maxlen, self.max_items = word2idx(line, self.vocab, self.maxlen, self.max_items, count)

        for line in self.pairs_neg:
            count, self.maxlen, self.max_items = word2idx(line, self.vocab, self.maxlen, self.max_items, count)

        '''if pre-save vectors from Google News'''
        if config.pp_data_weak:
            save_name = 'embed\embedding\word_embedding_all_include_classification8.txt'
            self.saveVocab(save_name)

        print("max_len: ", self.maxlen)
        print("count len > 200 :", count)
        return self.vocab, self.pairs_pos, self.pairs_neg

    @property
    def aspect_extract_data(self):
        """get aspect extracting train data"""
        '''if pre-save vectors from Google News'''
        if config.pp_data_clas:
            save_name = 'embed\embedding\word_embedding_classifier.txt'
            self.saveVocab(save_name)

        return self.vocab, self.pairs_all

    @property
    def weakly_data_process(self):
        """process weakly train data"""
        print("=" * 100)
        print("Weakly data Process...")

        vocab, pairs_pos, pairs_neg, asp_pos, asp_neg = self.weakly_data

        final_embedding = np.array(np.load("embed/Vector_word_embedding_all_update.npy"))

        '''initialize sentence'''
        input_sen_1 = config.pad_idx + np.zeros((len(pairs_pos), config.maxlen))
        input_sen_1 = input_sen_1.astype(np.int)
        input_sen_2 = config.pad_idx + np.zeros((len(pairs_neg), config.maxlen))
        input_sen_2 = input_sen_2.astype(np.int)

        '''initialize aspects of sentence'''
        input_asp_1 = config.pad_idx + np.zeros((len(pairs_pos), config.maxlen_asp))
        input_asp_1 = input_asp_1.astype(np.int)
        input_asp_2 = config.pad_idx + np.zeros((len(pairs_neg), config.maxlen_asp))
        input_asp_2 = input_asp_2.astype(np.int)

        '''a list of aspects of all sentences'''
        asp_list = []

        if config.need_pos is True:
            input_sen_1 = config.pad_idx + np.zeros((len(pairs_pos), 2 * config.maxlen))
            input_sen_1[:, config.maxlen:] = config.maxlen_ + np.zeros((len(pairs_pos), config.maxlen))
            input_sen_1 = input_sen_1.astype(np.int)
            input_sen_2 = config.pad_idx + np.zeros((len(pairs_neg), 2 * config.maxlen))
            input_sen_2[:, config.maxlen:] = config.maxlen_ + np.zeros((len(pairs_neg), config.maxlen))
            input_sen_2 = input_sen_2.astype(np.int)

        def sentence2vec(sentence, aspect):
            wordindex = []
            aspindex = []
            sent_items = sentence.strip().split()
            asp_items = aspect.strip().split()
            length = len(sent_items)
            for word in sent_items:
                # if word in self.vocab:
                wordindex.append(self.vocab[word])
                # else:
                #     wordindex.append(np.random.randint(len(self.vocab) - 1))  # if not found
            for asp in asp_items:
                # if asp in self.vocab:
                aspindex.append(self.vocab[asp])
                # else:
                #     aspindex.append(np.random.randint(len(self.vocab) - 1))  # if not found
            return length, wordindex, aspindex

        def cal_sentence_index():
            asp_idx = 0
            for idx, (_, asp) in enumerate(zip(pairs_pos, input_asp_1)):
                length, wordindex, aspindex = sentence2vec(pairs_pos[idx], asp_pos[idx])
                for i, a_i in enumerate(aspindex):
                    asp[i] = a_i
                asp_list.append(asp)
                input_sen_1[idx][0] = length
                input_sen_1[idx][1] = asp_idx  # idx for looking up aspect
                input_sen_1[idx][2:length + 2] = np.array(wordindex)
                asp_idx += 1

            for idx, (_, asp) in enumerate(zip(pairs_neg, input_asp_2)):
                length, wordindex, aspindex = sentence2vec(pairs_neg[idx], asp_neg[idx])
                for i, a_i in enumerate(aspindex):
                    asp[i] = a_i
                asp_list.append(asp)
                input_sen_2[idx][0] = length
                input_sen_2[idx][1] = asp_idx  # idx for looking up aspect
                input_sen_2[idx][2:length + 2] = np.array(wordindex)
                asp_idx += 1

            # return input_sen_1, input_sen_2

        '''serialize sentence and add extra info'''
        # input_sen_1, input_sen_2 = self.week_cal_sentence_index(
        #     input_sen_1, input_sen_2, pairs_pos, pairs_neg)
        cal_sentence_index()

        '''initialize unknown word embedding'''
        add = np.zeros(config.embed_dim)
        final_embedding = np.row_stack((final_embedding, add))

        '''randomly choose train and test data'''
        np.random.shuffle(input_sen_1)
        np.random.shuffle(input_sen_2)

        input_pos_train = input_sen_1[:int(len(input_sen_1) * config.weak_sr), :]
        input_neg_train = input_sen_2[:int(len(input_sen_2) * config.weak_sr), :]

        input_pos_test = input_sen_1[int(len(input_sen_1) * config.weak_sr):, :]
        input_neg_test = input_sen_2[int(len(input_sen_2) * config.weak_sr):, :]

        train_pos_1 = self.random_sample(input_pos_train, config.sample_size)
        train_pos_2 = self.random_sample(input_pos_train, config.sample_size)
        train_pos_neg = self.random_sample(input_neg_train, config.sample_size)
        train_neg_1 = self.random_sample(input_neg_train, config.sample_size)
        train_neg_2 = self.random_sample(input_neg_train, config.sample_size)
        train_neg_pos = self.random_sample(input_pos_train, config.sample_size)

        train_dim1 = np.vstack((train_pos_1, train_neg_1))
        train_dim2 = np.vstack((train_pos_2, train_neg_2))
        train_dim3 = np.vstack((train_pos_neg, train_neg_pos))

        all_data = MyDataset(self.read_weak_data(train_dim1, train_dim2, train_dim3, asp_list))

        return all_data, final_embedding, asp_list, \
               np.array(input_pos_test[0:config.weak_test_samples, :]), \
               np.array(input_neg_test[0:config.weak_test_samples, :])

    @property
    def clas_data_process(self):
        """process classification train data"""
        print("=" * 100)
        print("Classification data Process...")

        vocab, pairs_clas, asp_clas = self.classification_data
        final_embedding = np.array(np.load("embed/Vector_word_embedding_all_update.npy"))

        # maxlen = 0
        # max_items = []
        #
        # for line in pairs_clas:
        #     maxlen, max_items = self.word2idx(line, maxlen, max_items)

        # initialize with pad, 4 for extra info
        input_sent = config.pad_idx + np.zeros(
            (len(pairs_clas), 204))  # max length of sentence: 200, 4 for extra info
        input_sent = input_sent.astype(np.int)
        input_asp = config.pad_idx + np.zeros((len(pairs_clas), config.maxlen_asp))  # max length of aspect: 3
        input_asp = input_asp.astype(np.int)

        if config.need_pos is True:
            input_sent = config.pad_idx + np.zeros((len(pairs_clas), 408))
            input_sent[:, 204:] = 200 + np.zeros((len(pairs_clas), 204))
            input_sent = input_sent.astype(np.int)

        pos_sent = []
        neg_sent = []

        '''serialize sentence and add extra info'''
        pos_sent, neg_sent, asp_list = self.clas_cal_sentence_index(input_sent, input_asp, pairs_clas, asp_clas)

        '''obtain train, valid, test data'''
        pos_sent = np.array(pos_sent)
        neg_sent = np.array(neg_sent)
        # np.random.seed(1)
        np.random.shuffle(pos_sent)
        # np.random.seed(1)
        np.random.shuffle(neg_sent)

        pos_sentence_train = pos_sent[:int(len(pos_sent) * config.clas_sr), :]
        pos_sentence_valid = pos_sent[int(len(pos_sent) * config.clas_sr): int(
            len(pos_sent) * (config.clas_sr + 1) / 2), :]
        pos_sentence_test = pos_sent[int(len(pos_sent) * (config.clas_sr + 1) / 2):, :]

        neg_sentence_train = neg_sent[:int(len(neg_sent) * config.clas_sr), :]
        neg_sentence_valid = neg_sent[int(len(neg_sent) * config.clas_sr): int(
            len(neg_sent) * (config.clas_sr + 1) / 2), :]
        neg_sentence_test = neg_sent[int(len(neg_sent) * (config.clas_sr + 1) / 2):, :]

        input_train = np.vstack((pos_sentence_train, neg_sentence_train))
        input_valid = np.vstack((pos_sentence_valid, neg_sentence_valid))
        input_test = np.vstack((pos_sentence_test, neg_sentence_test))

        np.random.shuffle(input_train)
        np.random.shuffle(input_valid)
        np.random.shuffle(input_test)

        '''initialize pad word embedding'''
        add = np.zeros(config.embed_dim)  # pad embedding
        final_embedding = np.row_stack((final_embedding, add))

        # [:, 2:]: ignore first two element
        train_data = MyDataset(self.read_clas_data(input_train[:, 2:], input_train[:, 0], asp_list))
        valid_data = MyDataset(self.read_clas_data(input_valid[:, 2:], input_valid[:, 0], asp_list))
        test_data = MyDataset(self.read_clas_data(input_test[:, 2:], input_test[:, 0], asp_list))
        return train_data, valid_data, test_data, final_embedding

    @property
    def aspect_extract_data_process(self):
        """process aspect extracting train data"""
        print("=" * 100)
        print("Aspect Extract data Process...")

        vocab, pairs_all = self.aspect_extract_data
        final_embedding = np.array(np.load("embed/Vector_word_embedding_all.npy"))

        maxlen = 0
        bb = []

        def word2idx(sentence, vocab, maxlen, bb):
            items = sentence.strip().split()
            if len(items) > maxlen:
                maxlen = len(items)
                bb = items
            for word in items[2:]:
                if word not in vocab:
                    vocab[word] = len(vocab)
            return maxlen, bb

        # for line in pairs_all:
        #     maxlen, bb = word2idx(line, vocab, maxlen, bb)

        # print(vocab)

        input_sen = config.pad_idx + np.zeros((len(pairs_all), config.maxlen))
        input_sen = input_sen.astype(np.int)

        if config.need_pos is True:
            input_sen = config.pad_idx + np.zeros((len(pairs_all), 2 * config.maxlen))
            input_sen[:, config.maxlen:] = 300 + np.zeros((len(pairs_all), config.maxlen))
            input_sen = input_sen.astype(np.int)

        '''serialize sentence'''
        input_sen = self.aspect_cal_sentence_index(input_sen, pairs_all)

        input_sen = np.array(input_sen)
        np.random.shuffle(input_sen)

        add = np.zeros(300)
        final_embedding = np.row_stack((final_embedding, add))
        train_data = MyDataset(self.read_aspect_data(input_sen))

        return train_data, final_embedding, input_sen

    def normalizeString(self, s):
        """clean symbols and lower letters"""
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?\(\)\"])", r"", s)
        s = re.sub(r"[^0-9a-zA-Z]+", r" ", s)
        return s

    def normalize(self, s):
        """clean other symbols"""
        # s = unicodeToAscii(s.strip())
        s = re.sub(r"([\[\]\"\n])", r"", s)
        return s

    def unicodeToAscii(self, s):
        """encode sentence from Unicode to Ascii"""
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    def saveVocab(self, filename, mode='w'):
        """save embedding vocab into local files"""
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',
                                                                binary=True)
        # spell = SpellChecker()
        spell = None
        i = 0
        with open(filename, mode=mode, encoding='utf-8') as file:
            for key in self.vocab.keys():
                if key in model:
                    a = key + ',' + self.normalize(str(model[key])) + "\n"
                    file.write(a)
                    i += 1
                else:
                    spell_key = spell.correction(key)
                    if spell_key in model:
                        a = key + "," + self.normalize(str(model[spell_key])) + "\n"
                        file.write(a)
                        i += 1
                    else:
                        a = key + "," + spell_key + "\n"
                        file.write(a)

    def word2idx(self, sentence, maxlen, max_items):
        """build vocab and count maxlen of sentence"""
        items = sentence.strip().split()
        if len(items) > maxlen:
            maxlen = len(items)
            max_items = items
        for word in items:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        return maxlen, max_items

    def save_vector(self):
        """save vectors"""
        lines = open('embed/word_embedding_classifier.txt',
                     encoding='utf-8').read().strip().split('\n')

        a = np.zeros((8986, 300))

        def cut_space(sentence):
            out = re.sub(r"\s{2,}", " ", sentence)
            return out

        pairs = [[s for s in l.split(',')] for l in lines]
        for i in range(len(pairs)):
            if len(pairs[i][1]) > 300:
                a[i] = np.array(list(map(eval, cut_space(pairs[i][1]).strip().split())))
                # a[i] = map(eval, a[i])
            else:
                a[i] = -1 + 2 * np.random.random(300)

        t = -1 + 2 * np.random.random(300)
        # c = np.insert(a, 12586, values=t, axis=0)
        np.save("embed/Vector_classifier.npy", a)

        # b = [[s for s in l.split()] for l in a]

        # print(b[0])

    def random_sample(self, matrix, sample_size):
        """random sample data"""
        matrix_after = []
        sample_index = np.random.randint(0, len(matrix), sample_size)
        for i in sample_index:
            # np.row_stack((matrix_after, matrix[i]))
            matrix_after.append(matrix[i])
        return np.array(matrix_after)

    def read_weak_data(self, dim_1, dim_2, dim_3, aspect):
        """read weakly data"""
        all_data = []
        for idx, (sent1, sent2, sent3) in enumerate(zip(dim_1, dim_2, dim_3)):
            items = torch.from_numpy(sent1)
            items1 = torch.from_numpy(sent2)
            items2 = torch.from_numpy(sent3)
            items3 = torch.from_numpy(aspect[sent1[1]])
            items4 = torch.from_numpy(aspect[sent2[1]])
            items5 = torch.from_numpy(aspect[sent3[1]])

            # for idx in range(len(dim_1)):
            # items = torch.from_numpy(dim_1[idx])
            # items1 = torch.from_numpy(dim_2[idx])
            # items2 = torch.from_numpy(dim_3[idx])

            data = {
                'input1': items,
                'input2': items1,
                'input3': items2,
                'aspect1': items3,
                'aspect2': items4,
                'aspect3': items5,
            }
            all_data.append(data)
        return all_data

    def read_clas_data(self, input_data, label, aspect):
        """read classification data"""
        all_data = []
        for idx, data in enumerate(input_data):
            items = torch.from_numpy(data)
            items1 = torch.tensor(int(label[idx]))
            # print('aspect', aspect[data[1]])
            items2 = torch.tensor(aspect[data[1]])  # a list of aspect index
            data = {
                'input': items,
                'label': items1,
                'aspect': items2,
            }
            all_data.append(data)
        return all_data

    def read_aspect_data(self, input_train):
        """read aspect extracting data"""
        all_data = []
        for idx in range(len(input_train)):
            items = torch.from_numpy(input_train[idx])
            data = {
                'input': items
            }
            all_data.append(data)
        return all_data

    def sentence2vec(self, sentence, wordindex):
        """serialize sentence"""
        items = sentence.strip().split()
        length = len(items)
        for word in items:
            wordindex.append(self.vocab[word])
        return length, wordindex

    def clas_sentence2vec(self, sentence, aspect):
        """serialize sentence"""
        wordindex = []
        aspindex = []
        sent_items = sentence.strip().split()
        asp_items = aspect.strip().split()
        label = 0
        obj = 0
        if sent_items[0] == "positive":
            label = 1
        if sent_items[1] == "objective":
            obj = 1
        length = len(sent_items) - 2

        for word in sent_items[2:]:
            # if word in self.vocab:
            wordindex.append(self.vocab[word])
            # else:
            #     wordindex.append(np.random.randint(len(self.vocab) - 1))
        for asp in asp_items:
            # if asp in self.vocab:
            aspindex.append(self.vocab[asp])
            # else:
            #     aspindex.append(np.random.randint(len(self.vocab) - 1))  # if not found
        return wordindex, aspindex, length, label, obj

    def week_cal_sentence_index(self, input_sent_1, input_sent_2, pairs_pos, pairs_neg):
        """serialize sentence and add extra info"""
        for line in range(len(pairs_pos)):
            wordindex = []
            length, wordindex = self.sentence2vec(pairs_pos[line], wordindex)
            input_sent_1[line][0] = length  # real length of sentence
            input_sent_1[line][1] = 10  # aspect index
            input_sent_1[line][2:length + 2] = np.array(wordindex)  # sentence
            if config.need_pos is True:
                input_sent_1[line][config.maxlen:length + config.maxlen] = [x for x in range(length)]

        for line in range(len(pairs_neg)):
            wordindex = []
            length, wordindex = self.sentence2vec(pairs_pos[line], wordindex)
            input_sent_2[line][0] = length
            input_sent_2[line][1] = 10
            input_sent_2[line][2:length + 2] = np.array(wordindex)
            if config.need_pos is True:
                input_sent_2[line][config.maxlen:length + config.maxlen] = [x for x in range(length)]
        return input_sent_1, input_sent_2

    def clas_cal_sentence_index(self, input_sent, input_asp, pairs_clas, asp_clas):
        """serialize sentence and add extra info"""
        pos_sent = []
        neg_sent = []
        asp_list = []
        for idx, (_, asp) in enumerate(zip(input_sent, input_asp)):
            wordindex, aspindex, length, label, obj = self.clas_sentence2vec(pairs_clas[idx], asp_clas[idx])
            for i, a_i in enumerate(aspindex):
                asp[i] = a_i
            asp_list.append(asp)  # a list of a list of aspect index
            input_sent[idx][0] = label  # ignore
            input_sent[idx][1] = obj  # ignore
            input_sent[idx][2] = length
            input_sent[idx][3] = idx  # idx for looking up asp_list
            input_sent[idx][4:length + 4] = np.array(wordindex)
            if config.need_pos is True:
                input_sent[idx][204:length + 204] = [x for x in range(length)]
        for sent in input_sent:
            if sent[0] == 1:
                pos_sent.append(sent)
            else:
                neg_sent.append(sent)
        return pos_sent, neg_sent, asp_list

    def aspect_cal_sentence_index(self, input_sent, pairs_all):
        """serialize sentence and add extra info"""
        for line in range(len(pairs_all)):
            wordindex = []
            length, wordindex = self.sentence2vec(pairs_all[line], wordindex)
            input_sent[line][0] = length
            input_sent[line][1] = 10
            input_sent[line][2:length + 2] = np.array(wordindex)

            if config.need_pos is True:
                input_sent[line][config.maxlen:length + config.maxlen] = [x for x in range(length)]
        return input_sent

    def clean_apriori_data(self, sentences):
        """
        filter apriori data
        methods:
        - clean stop words
        - stemming
        - fuzzy matching within sentence
        """
        stop_words = stopwords.words('english')
        eng_parser = StanfordParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

        if config.apriori_test_size < 6:
            for sent in sentences:
                print(sent)
        '''POS'''
        pos_sent = []
        for sent in sentences:
            pos_sent.append(list(eng_parser.parse(
                [w for w in sent.split()]))[0])

        '''filter noun phrase & NLTK stemming'''
        cleaned_sent = []
        for sent in pos_sent:
            wnl = WordNetLemmatizer()
            tmp_sent = []
            for s in sent.subtrees(lambda t: t.height() <= 4 and t.label() == 'NP'):
                '''clean stop words & stemming'''
                tmp = [wnl.lemmatize(w, pos='n') for w in s.leaves() if w not in stop_words]
                '''lenght <= 3 & filter repeated list'''
                if 0 < len(tmp) <= 3 and tmp not in tmp_sent:
                    tmp_sent.append(tmp)
            cleaned_sent.append(tmp_sent)

        return pos_sent

    def POS_data(self):
        """POS sentences"""
        tag = 'pos'
        idx = 19
        file_name = 'data/normalize_{}_piece/nor_{}_{}.csv'.format(tag, tag, idx)
        with open(file_name, 'r') as file:
            sentences = file.read().strip().split('\n')

        stop_words = stopwords.words('english')
        eng_parser = StanfordParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
        eng_parser.java_options = '-mx3000m'

        print('=' * 100)
        print('current tag: {}, file idx: {}'.format(tag, idx))

        '''POS'''
        print('=' * 100)
        print('Starting POS...')
        pos_sent = []
        for sent in tqdm(sentences):
            pos_sent.append(list(eng_parser.parse(
                [w for w in sent.split()]))[0])

        '''save file'''
        save_file = 'data/{}_sent/{}_sent_{}.csv'.format(tag, tag, idx)
        with open(save_file, mode='w') as file:
            for sent, pos in zip(sentences, pos_sent):
                file.write(sent + '\t')
                file.write(str(pos) + '\t')
        print('Finish! Saved in {}'.format(save_file))

    def get_apriori_data(self):
        """get data of Apriori algorithm"""
        tag = 'neg'
        # times: number of file pieces
        if tag == 'neg':
            times = 20
        elif tag == 'pos':
            times = 50
        else:
            times = 0

        for i_file in tqdm(range(times)):
            file_name = 'data/{}_sent/{}_sent_{}.csv'.format(tag, tag, i_file)
            save_file = 'data/{}_sent_clean/{}_sent_clean_{}.csv'.format(tag, tag, i_file)
            stop_words = stopwords.words('english')

            '''load data from file'''
            with open(file_name, 'r') as file:
                all_lines = file.read().split('\t')

            all_sent = []
            all_tree = []
            for idx, line in enumerate(all_lines):
                if idx % 2 == 0:
                    all_sent.append(line.strip())
                else:
                    all_tree.append(Tree.fromstring(line.strip()))

            all_sent.remove('')  # remove last empty line
            wnl = WordNetLemmatizer()

            '''stemming and get noun phrases'''
            all_pos = []
            cleaned_sent = []
            for idx, (sent, tree) in tqdm(enumerate(zip(all_sent, all_tree))):
                dict_pos = dict(tree.pos())  # tree.pos() to dict
                words = sent.split()
                # stemming for noun
                for i, w in enumerate(words):
                    if dict_pos[w] == 'NNS':
                        words[i] = wnl.lemmatize(w, pos='n')  # stemming
                        dict_pos[words[i]] = 'NN'  # update POS label

                # update origin sentence and POS dict after stemming
                all_sent[idx] = ' '.join(words)
                all_pos.append(dict_pos)

                # get noun phrase
                tmp_sent = []
                for s in tree.subtrees(lambda t: t.height() <= 4 and t.label() == 'NP'):
                    '''clean stop words & stemming'''
                    tmp = []
                    for w in s.leaves():
                        if w not in stop_words:
                            # if have been stemmed above
                            if w not in words:
                                tmp.append(wnl.lemmatize(w, pos='n'))
                            else:
                                tmp.append(w)
                    '''length <= 3 & filter repeated list'''
                    if 0 < len(tmp) <= 2 and tmp not in tmp_sent:
                        tmp_sent.append(tmp)
                cleaned_sent.append(tmp_sent)

            '''save cleaned data into file'''
            with open(save_file, 'w') as file:
                for sent, pos, cl_sent in zip(all_sent, all_pos, cleaned_sent):
                    file.write(sent + '\n')
                    file.write(str(pos) + '\n')
                    file.write('\t'.join([' '.join(NP) for NP in cl_sent]) + '\n')

    def save_normalize_file(self):
        nor_all_file = 'data/nor_all.csv'
        nor_pos_file = 'data/nor_pos.csv'
        nor_neg_file = 'data/nor_neg.csv'
        nor_clas_file = 'data/nor_clas.csv'

        with open(nor_all_file, 'w') as file:
            file.write('\n'.join(self.pairs_all))
        with open(nor_pos_file, 'w') as file:
            file.write('\n'.join(self.pairs_pos))
        with open(nor_neg_file, 'w') as file:
            file.write('\n'.join(self.pairs_neg))
        with open(nor_clas_file, 'w') as file:
            file.write('\n'.join(self.pairs_clas))

    def save_word_vocab(self):
        """save self.vocab for future process"""
        save_filename = 'data/vocab.txt'
        with open(save_filename, 'w') as file:
            file.write(str(self.vocab))

    def load_word_vocab(self):
        """load self.vocab from file"""
        vocab_filename = 'data/vocab_update.txt'
        with open(vocab_filename, 'r') as file:
            return eval(file.read())


class CornerData:
    def __init__(self):
        pass

    def pp_dataloader_weak(self, all_data, final_embedding):
        all_data_train = all_data
        embed = final_embedding
        final_embedding = torch.from_numpy(embed)
        train_dataloader = DataLoader(
            dataset=all_data_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        return final_embedding, train_dataloader

    def pp_dataloader_clas(self, data):
        train_data, valid_data, test_data, embedding = data
        # add = -1 + 2 * np.random.random(300)
        # final_embedding = np.row_stack((final_embedding, add))
        final_embedding = torch.from_numpy(embedding)

        train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        valid_dataloader = DataLoader(
            dataset=valid_data,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        test_dataloader = DataLoader(
            dataset=test_data,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        return final_embedding, train_dataloader, valid_dataloader, test_dataloader

    def pp_dataloader_aspect(self, data):
        train_data, embedding = data
        final_embedding = torch.from_numpy(embedding)
        train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        return final_embedding, train_dataloader
