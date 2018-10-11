from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import string
import random
import numpy as np
import torch
# from spellchecker import SpellChecker


def weakly_data(opt):

    lines_pos1 = open('data/'
                      'Weakly_labeled_data_1.1M/camera_positive.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_pos2 = open('data/'
                      'Weakly_labeled_data_1.1M/cellphone_positive.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_pos3 = open('data/'
                      'Weakly_labeled_data_1.1M/laptop_positive.csv',
                      encoding='utf-8').read().strip().split('\n')

    lines_pos = lines_pos1 + lines_pos2 + lines_pos3

    lines_neg1 = open('data/'
                      'Weakly_labeled_data_1.1M/camera_negativet.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_neg2 = open('data/'
                      'Weakly_labeled_data_1.1M/cellphone_negative.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_neg3 = open('data/'
                      'Weakly_labeled_data_1.1M/laptop_negative.csv',
                      encoding='utf-8').read().strip().split('\n')

    lines_neg = lines_neg1 + lines_neg2 + lines_neg3

    lines = open('data/Labeled_data_11754/new_11754.csv',
                 encoding='gbk').read().strip().split('\n')

    pairs_classifier = [normalizeString(s) for s in lines]
    pairs_pos = [normalizeString(s) for s in lines_pos]
    pairs_neg = [normalizeString(s) for s in lines_neg]
    # print(pairs_classifier[0])

    vocab = {}
    print("=" * 100)
    print("Prepare Weakly training data...")

    maxlen = 0
    bb = []

    def saveVocab(vocab1, filename, mode='w'):
        spell = SpellChecker()
        i = 0
        with open(filename, mode=mode, encoding='utf-8') as file:
            for key in vocab1.keys():
                if key in model:
                    a = key + ',' + normalize(str(model[key])) + "\n"
                    file.write(a)
                    i += 1
                else:
                    spell_key = spell.correction(key)
                    if spell_key in model:
                        a = key + "," + normalize(str(model[spell_key])) + "\n"
                        file.write(a)
                        i += 1
                    else:
                        a = key + "," + spell_key + "\n"
                        file.write(a)

    def aboutData(opt):
        # Save Word embedding
        save_name = 'embed\embedding\word_embedding_classifier.txt'
        saveVocab(vocab, save_name)

    if opt.preparedata_classification:
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin', binary=True)
        aboutData(opt)

    def word2idx(sentence, vocab, maxlen, bb):
        items = sentence.strip().split()
        if len(items) > maxlen:
            maxlen = len(items)
            bb = items
        for word in items:
            if word not in vocab:
                vocab[word] = len(vocab)
        return maxlen, bb

    for line in pairs_classifier:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    for line in pairs_pos:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    for line in pairs_neg:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    return vocab, pairs_pos, pairs_neg


def classification_data(opt):
    lines_pos1 = open('data/'
                      'Weakly_labeled_data_1.1M/camera_positive.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_pos2 = open('data/'
                      'Weakly_labeled_data_1.1M/cellphone_positive.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_pos3 = open('data/'
                      'Weakly_labeled_data_1.1M/laptop_positive.csv',
                      encoding='utf-8').read().strip().split('\n')

    lines_pos = lines_pos1 + lines_pos2 + lines_pos3

    lines_neg1 = open('data/'
                      'Weakly_labeled_data_1.1M/camera_negativet.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_neg2 = open('data/'
                      'Weakly_labeled_data_1.1M/cellphone_negative.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_neg3 = open('data/'
                      'Weakly_labeled_data_1.1M/laptop_negative.csv',
                      encoding='utf-8').read().strip().split('\n')

    lines_neg = lines_neg1 + lines_neg2 + lines_neg3

    lines = open('data/Labeled_data_11754/new_11754.csv',
                 encoding='gbk').read().strip().split('\n')

    pairs_classifier = [normalizeString(s) for s in lines]
    pairs_pos = [normalizeString(s) for s in lines_pos]
    pairs_neg = [normalizeString(s) for s in lines_neg]
    # print(pairs_classifier[0])

    vocab = {}
    print("=" * 100)
    print("Prepare Classification training data...")

    maxlen = 0
    bb = []

    def saveVocab(vocab1, filename, mode='w'):
        spell = SpellChecker()
        i = 0
        with open(filename, mode=mode, encoding='utf-8') as file:
            for key in vocab1.keys():
                if key in model:
                    a = key + ',' + normalize(str(model[key])) + "\n"
                    file.write(a)
                    i += 1
                else:
                    spell_key = spell.correction(key)
                    if spell_key in model:
                        a = key + "," + normalize(str(model[spell_key])) + "\n"
                        file.write(a)
                        i += 1
                    else:
                        a = key + "," + spell_key + "\n"
                        file.write(a)

    def aboutData(opt):
        # Save Word embedding
        save_name = 'embed\embedding\word_embedding_classifier.txt'
        saveVocab(vocab, save_name)

    if opt.preparedata_classification:
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin', binary=True)
        aboutData(opt)

    # def word2idx(sentence, vocab, maxlen, bb):
    #     items = sentence.strip().split()
    #     if len(items) > maxlen:
    #         maxlen = len(items)
    #         bb = items
    #     for word in items[2:]:
    #         if word not in vocab:
    #             vocab[word] = len(vocab)
    #     return maxlen, bb

    def word2idx(sentence, vocab, maxlen, bb):
        items = sentence.strip().split()
        if len(items) > maxlen:
            maxlen = len(items)
            bb = items
        for word in items:
            if word not in vocab:
                vocab[word] = len(vocab)
        return maxlen, bb

    for line in pairs_classifier:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    for line in pairs_pos:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    for line in pairs_neg:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    return vocab, pairs_classifier


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


def weakly_data_test(preparedata_weakly=False):

    lines_pos1 = open('data/'
                      'Weakly_labeled_data_1.1M/camera_positive.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_pos2 = open('data/'
                      'Weakly_labeled_data_1.1M/cellphone_positive.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_pos3 = open('data/'
                      'Weakly_labeled_data_1.1M/laptop_positive.csv',
                      encoding='utf-8').read().strip().split('\n')

    lines_pos = lines_pos1 + lines_pos2 + lines_pos3

    lines_neg1 = open('data/'
                      'Weakly_labeled_data_1.1M/camera_negativet.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_neg2 = open('data/'
                      'Weakly_labeled_data_1.1M/cellphone_negative.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_neg3 = open('data/'
                      'Weakly_labeled_data_1.1M/laptop_negative.csv',
                      encoding='utf-8').read().strip().split('\n')

    lines_neg = lines_neg1 + lines_neg2 + lines_neg3

    lines = open('data/Labeled_data_11754/new_11754.csv',
                 encoding='gbk').read().strip().split('\n')

    pairs_classifier = [normalizeString(s) for s in lines]
    pairs_pos = [normalizeString(s) for s in lines_pos]
    pairs_neg = [normalizeString(s) for s in lines_neg]

    vocab = {}
    print("=" * 100)
    print("Prepare weakly training data...")
    maxlen = 0
    bb = []
    count = 0

    def word2idx(sentence, vocab, maxlen, bb, count):
        items = sentence.strip().split()
        if len(items) > maxlen:
            maxlen = len(items)
            bb = items
        for word in items:
            if word not in vocab:
                vocab[word] = len(vocab)
        if len(items) > 200:
            count = 1 + count
        return count, maxlen, bb

    for line in pairs_classifier:
        count, maxlen, bb = word2idx(line, vocab, maxlen, bb, count)

    for line in pairs_pos:
        count, maxlen, bb = word2idx(line, vocab, maxlen, bb, count)

    for line in pairs_neg:
        count, maxlen, bb = word2idx(line, vocab, maxlen, bb, count)

    def saveVocab(vocab1, filename, mode='w'):
        spell = SpellChecker()
        i = 0
        with open(filename, mode=mode, encoding='utf-8') as file:
            for key in vocab1.keys():
                if key in model:
                    a = key + ',' + normalize(str(model[key])) + "\n"
                    file.write(a)
                    i += 1
                else:
                    spell_key = spell.correction(key)
                    if spell_key in model:
                        a = key + "," + normalize(str(model[spell_key])) + "\n"
                        file.write(a)
                        i += 1
                    else:
                        a = key + "," + spell_key + "\n"
                        file.write(a)
        print(i)

    def aboutData():
        # Save Word embedding
        save_name = 'embed\embedding\word_embedding_all_include_classification8.txt'
        saveVocab(vocab, save_name)
    if preparedata_weakly:
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin', binary=True)
        aboutData()
    print("max_len: ", maxlen)
    print("count len > 200 :", count)
    return vocab, pairs_pos, pairs_neg


def aspect_extract_data(opt):
    lines_pos1 = open('data/'
                      'Weakly_labeled_data_1.1M/camera_positive.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_pos2 = open('data/'
                      'Weakly_labeled_data_1.1M/cellphone_positive.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_pos3 = open('data/'
                      'Weakly_labeled_data_1.1M/laptop_positive.csv',
                      encoding='utf-8').read().strip().split('\n')

    lines_pos = lines_pos1 + lines_pos2 + lines_pos3

    lines_neg1 = open('data/'
                      'Weakly_labeled_data_1.1M/camera_negativet.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_neg2 = open('data/'
                      'Weakly_labeled_data_1.1M/cellphone_negative.csv',
                      encoding='utf-8').read().strip().split('\n')
    lines_neg3 = open('data/'
                      'Weakly_labeled_data_1.1M/laptop_negative.csv',
                      encoding='utf-8').read().strip().split('\n')

    lines_neg = lines_neg1 + lines_neg2 + lines_neg3

    lines = open('data/Labeled_data_11754/new_11754.csv',
                 encoding='gbk').read().strip().split('\n')

    lines_all = lines_pos + lines_neg

    pairs_all = [normalizeString(s) for s in lines_all]
    pairs_classifier = [normalizeString(s) for s in lines]
    pairs_pos = [normalizeString(s) for s in lines_pos]
    pairs_neg = [normalizeString(s) for s in lines_neg]
    # print(pairs_classifier[0])

    vocab = {}
    print("=" * 100)
    print("Prepare Aspect training data...")

    maxlen = 0
    bb = []

    def saveVocab(vocab1, filename, mode='w'):
        spell = SpellChecker()
        i = 0
        with open(filename, mode=mode, encoding='utf-8') as file:
            for key in vocab1.keys():
                if key in model:
                    a = key + ',' + normalize(str(model[key])) + "\n"
                    file.write(a)
                    i += 1
                else:
                    spell_key = spell.correction(key)
                    if spell_key in model:
                        a = key + "," + normalize(str(model[spell_key])) + "\n"
                        file.write(a)
                        i += 1
                    else:
                        a = key + "," + spell_key + "\n"
                        file.write(a)

    def aboutData(opt):
        # Save Word embedding
        save_name = 'embed\embedding\word_embedding_classifier.txt'
        saveVocab(vocab, save_name)

    if opt.preparedata_classification:
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin', binary=True)
        aboutData(opt)

    # def word2idx(sentence, vocab, maxlen, bb):
    #     items = sentence.strip().split()
    #     if len(items) > maxlen:
    #         maxlen = len(items)
    #         bb = items
    #     for word in items[2:]:
    #         if word not in vocab:
    #             vocab[word] = len(vocab)
    #     return maxlen, bb

    def word2idx(sentence, vocab, maxlen, bb):
        items = sentence.strip().split()
        if len(items) > maxlen:
            maxlen = len(items)
            bb = items
        for word in items:
            if word not in vocab:
                vocab[word] = len(vocab)
        return maxlen, bb

    for line in pairs_classifier:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    for line in pairs_pos:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    for line in pairs_neg:
        maxlen, bb = word2idx(line, vocab, maxlen, bb)

    return vocab, pairs_all
