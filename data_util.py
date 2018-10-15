# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : Corner-william
# @FileName     : data_util.py
# @Time         : Created at 2018/10/11
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

from __future__ import unicode_literals, print_function, division

import unicodedata

import numpy as np
import re
import torch
from io import open
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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
        lines_pos1 = open('data/Weakly_labeled_data_1.1M/camera_positive.csv',
                          encoding='utf-8').read().strip().split('\n')
        lines_pos2 = open('data/Weakly_labeled_data_1.1M/cellphone_positive.csv',
                          encoding='utf-8').read().strip().split('\n')
        lines_pos3 = open('data/Weakly_labeled_data_1.1M/laptop_positive.csv',
                          encoding='utf-8').read().strip().split('\n')
        lines_neg1 = open('data/Weakly_labeled_data_1.1M/camera_negative.csv',
                          encoding='utf-8').read().strip().split('\n')
        lines_neg2 = open('data/Weakly_labeled_data_1.1M/cellphone_negative.csv',
                          encoding='utf-8').read().strip().split('\n')
        lines_neg3 = open('data/Weakly_labeled_data_1.1M/laptop_negative.csv',
                          encoding='utf-8').read().strip().split('\n')

        lines = open('data/Labeled_data_11754/new_11754.csv',
                     encoding='gbk').read().strip().split('\n')

        '''merge data'''
        lines_pos = lines_pos1 + lines_pos2 + lines_pos3
        lines_neg = lines_neg1 + lines_neg2 + lines_neg3
        lines_all = lines_pos + lines_neg

        '''normalize sentences'''
        self.pairs_all = [self.normalizeString(s) for s in lines_all]
        self.pairs_pos = [self.normalizeString(s) for s in lines_pos]
        self.pairs_neg = [self.normalizeString(s) for s in lines_neg]
        self.pairs_classifier = [self.normalizeString(s) for s in lines]

        self.vocab = {}
        self.maxlen = 0
        self.max_items = []

        '''count maxlen and obtain bb'''
        for line in self.pairs_classifier:
            self.maxlen, self.max_items = self.word2idx(line, self.maxlen, self.max_items)

        for line in self.pairs_pos:
            self.maxlen, self.max_items = self.word2idx(line, self.maxlen, self.max_items)

        for line in self.pairs_neg:
            self.maxlen, self.max_items = self.word2idx(line, self.maxlen, self.max_items)

    @property
    def weakly_data(self):
        """get weakly train data"""
        '''if pre-save vectors from Google News'''
        if config.pp_data_weak:
            save_name = 'embed\embedding\word_embedding_classifier.txt'
            self.saveVocab(save_name)

        return self.vocab, self.pairs_pos, self.pairs_neg

    @property
    def classification_data(self):
        """get classification train data"""
        '''if pre-save vectors from Google News'''
        if config.pp_data_clas:
            save_name = 'embed\embedding\word_embedding_classifier.txt'
            self.saveVocab(save_name)

        return self.vocab, self.pairs_classifier

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

        for line in self.pairs_classifier:
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

        vocab, pairs_pos, pairs_neg = self.weakly_data

        final_embedding = np.array(np.load("embed/Vector_word_embedding_all.npy"))

        '''initialize sentence'''
        input_sen_1 = config.pad_idx + np.zeros((len(pairs_pos), config.maxlen))
        input_sen_1 = input_sen_1.astype(np.int)
        input_sen_2 = config.pad_idx + np.zeros((len(pairs_neg), config.maxlen))
        input_sen_2 = input_sen_2.astype(np.int)

        if config.need_pos is True:
            input_sen_1 = config.pad_idx + np.zeros((len(pairs_pos), 2 * config.maxlen))
            input_sen_1[:, config.maxlen:] = config.maxlen_ + np.zeros((len(pairs_pos), config.maxlen))
            input_sen_1 = input_sen_1.astype(np.int)
            input_sen_2 = config.pad_idx + np.zeros((len(pairs_neg), 2 * config.maxlen))
            input_sen_2[:, config.maxlen:] = config.maxlen_ + np.zeros((len(pairs_neg), config.maxlen))
            input_sen_2 = input_sen_2.astype(np.int)

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
                input_sen_1[line][0] = length
                input_sen_1[line][1] = 10
                input_sen_1[line][2:length + 2] = np.array(wordindex)

            for line in range(len(pairs_neg)):
                wordindex = []
                length, wordindex = sentence2vec(pairs_neg[line], vocab, wordindex)
                input_sen_2[line][0] = length
                input_sen_2[line][1] = 10
                input_sen_2[line][2:length + 2] = np.array(wordindex)
            return input_sen_1, input_sen_2

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

        all_data = MyDataset(self.read_weak_data(train_dim1, train_dim2, train_dim3))

        return all_data, final_embedding, \
               np.array(input_pos_test[0:config.weak_test_samples, :]), \
               np.array(input_neg_test[0:config.weak_test_samples, :])

    @property
    def clas_data_process(self):
        """process classification train data"""
        print("=" * 100)
        print("Classification data Process...")

        vocab, pairs_classifier = self.classification_data
        final_embedding = np.array(np.load("embed/Vector_word_embedding_all.npy"))

        maxlen = 0
        max_items = []

        for line in pairs_classifier:
            maxlen, max_items = self.word2idx(line, maxlen, max_items)

        input_sentence = config.pad_idx + np.zeros((len(pairs_classifier), 204))
        input_sentence = input_sentence.astype(np.int)

        if config.need_pos is True:
            input_sentence = config.pad_idx + np.zeros((len(pairs_classifier), 408))
            input_sentence[:, 204:] = 200 + np.zeros((len(pairs_classifier), 204))
            input_sentence = input_sentence.astype(np.int)

        pos_sentence = []
        neg_sentence = []

        '''serialize sentence and add extra info'''
        input_sentence, pos_sentence, neg_sentence = self.clas_cal_sentence_index(
            input_sentence, pos_sentence, neg_sentence, pairs_classifier)

        '''obtain train, valid, test data'''
        pos_sentence = np.array(pos_sentence)
        neg_sentence = np.array(neg_sentence)
        np.random.seed(1)
        np.random.shuffle(pos_sentence)
        np.random.seed(1)
        np.random.shuffle(neg_sentence)

        pos_sentence_train = pos_sentence[:int(len(pos_sentence) * config.clas_sr), :]
        pos_sentence_valid = pos_sentence[int(len(pos_sentence) * config.clas_sr): int(
            len(pos_sentence) * (config.clas_sr + 1) / 2), :]
        pos_sentence_test = pos_sentence[int(len(pos_sentence) * (config.clas_sr + 1) / 2):, :]

        neg_sentence_train = neg_sentence[:int(len(neg_sentence) * config.clas_sr), :]
        neg_sentence_valid = neg_sentence[int(len(neg_sentence) * config.clas_sr): int(
            len(neg_sentence) * (config.clas_sr + 1) / 2), :]
        neg_sentence_test = neg_sentence[int(len(neg_sentence) * (config.clas_sr + 1) / 2):, :]

        input_train = np.vstack((pos_sentence_train, neg_sentence_train))
        input_valid = np.vstack((pos_sentence_valid, neg_sentence_valid))
        input_test = np.vstack((pos_sentence_test, neg_sentence_test))

        np.random.shuffle(input_train)
        np.random.shuffle(input_valid)
        np.random.shuffle(input_test)

        '''initialize unknown word embedding'''
        add = np.zeros(300)
        final_embedding = np.row_stack((final_embedding, add))

        train_data = MyDataset(self.read_clas_data(input_train[:, 2:], input_train[:, 0]))
        valid_data = MyDataset(self.read_clas_data(input_valid[:, 2:], input_valid[:, 0]))
        test_data = MyDataset(self.read_clas_data(input_test[:, 2:], input_test[:, 0]))
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
        if config.pp_data_clas:
            import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin',
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

    def read_weak_data(self, dim_1, dim_2, dim_3):
        """read weakly data"""
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

    def read_clas_data(self, input_data, label):
        """read classification data"""
        all_data = []
        for idx in range(len(input_data)):
            items = torch.from_numpy(input_data[idx])
            items1 = torch.tensor(int(label[idx]))
            data = {
                'input': items,
                'label': items1,
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

    def clas_sentence2vec(self, sentence, vocab, wordindex):
        """serialize sentence"""
        items = sentence.strip().split()
        label = 0
        obj = 0
        if items[0] == "positive":
            label = 1
        if items[1] == "objective":
            obj = 1
        length = len(items) - 2

        for word in items[2:]:
            wordindex.append(vocab[word])
        return wordindex, length, label, obj

    def week_cal_sentence_index(self, input_sen_1, input_sen_2, pairs_pos, pairs_neg):
        """serialize sentence and add extra info"""
        for line in range(len(pairs_pos)):
            wordindex = []
            length, wordindex = self.sentence2vec(pairs_pos[line], wordindex)
            input_sen_1[line][0] = length  # real length of sentence
            input_sen_1[line][1] = 10  # aspect index
            input_sen_1[line][2:length + 2] = np.array(wordindex)  #
            if config.need_pos is True:
                input_sen_1[line][config.maxlen:length + config.maxlen] = [x for x in range(length)]

        for line in range(len(pairs_neg)):
            wordindex = []
            length, wordindex = self.sentence2vec(pairs_pos[line], wordindex)
            input_sen_2[line][0] = length
            input_sen_2[line][1] = 10
            input_sen_2[line][2:length + 2] = np.array(wordindex)
            if config.need_pos is True:
                input_sen_2[line][config.maxlen:length + config.maxlen] = [x for x in range(length)]
        return input_sen_1, input_sen_2

    def clas_cal_sentence_index(self, input_sen, pos_sen, neg_sen, pairs_classifier):
        """serialize sentence and add extra info"""
        for line in range(len(pairs_classifier)):
            wordindex = []
            wordindex, length, label, obj = self.clas_sentence2vec(pairs_classifier[line],
                                                                   self.vocab, wordindex)
            input_sen[line][0] = label
            input_sen[line][1] = obj
            input_sen[line][2] = length
            input_sen[line][3] = 10
            input_sen[line][4:length + 4] = np.array(wordindex)
            if config.need_pos is True:
                input_sen[line][204:length + 204] = [x for x in range(length)]
        for sentence in input_sen:
            if sentence[0] == 1:
                pos_sen.append(sentence)
            else:
                neg_sen.append(sentence)
        return input_sen, pos_sen, neg_sen

    def aspect_cal_sentence_index(self, input_sen, pairs_all):
        """serialize sentence and add extra info"""
        for line in range(len(pairs_all)):
            wordindex = []
            length, wordindex = self.sentence2vec(pairs_all[line], wordindex)
            input_sen[line][0] = length
            input_sen[line][1] = 10
            input_sen[line][2:length + 2] = np.array(wordindex)

            if config.need_pos is True:
                input_sen[line][config.maxlen:length + config.maxlen] = [x for x in range(length)]
        return input_sen


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
