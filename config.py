# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : Corner-william
# @FileName     : config.py
# @Time         : Created at 2018/10/11
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

# from models import *

os.environ[
    'CLASSPATH'] = '/home/sysu2018/stanford/postagger/stanford-postagger.jar:' \
                   '/home/sysu2018/stanford/parser/stanford-parser.jar:' \
                   '/home/sysu2018/stanford/parser/stanford-parser-3.9.2-models.jar'

'''classify model choices'''
# CLAS_MODEL_CHO = {
#     'classify': SentimentClassification,
#     'weakly': WeaklyTrainModel,
#     'aspect': ABAE
# }

'''save mode choices'''
SAVE_MODE_CHO = ['all', 'best']

'''optimizers choices'''
OPTIM_CHO = {
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'sgd': optim.SGD,
}

'''criterion choices'''
CRITERION_CHO = {
    'bce': nn.BCELoss,
    'mse': nn.MSELoss,
    'nll': nn.NLLLoss,
    'cross_entropy': nn.CrossEntropyLoss,
    'cos_embed': nn.CosineEmbeddingLoss,
    'tri_marginloss': nn.TripletMarginLoss,
}

'''training params'''
clas_lr = 0.0001
weak_lr = 0.0001
epoch = 30
batch_size = 8
optimizer = OPTIM_CHO['sgd']
criterion = CRITERION_CHO['cross_entropy']
margin = 4.0
margin_p = 2
valid_step = 2
valid_thres = 1.13

save_mode = 'best'
train_type = ''
train_phase = 'classify'  # choices: ['weakly', 'classify', 'aspect', 'ae_apriori']
# clas_model = CLAS_MODEL_CHO[train_phase]
lambda_ = 1
# pretrained_model = 'pretrainmodel/weakly/retain1/model_loss_0.91601valid_1.2928.pkl'
pretrained_model = 'pretrainmodel/every2/every2_loss_1.65873valid_1.20795.pkl'

'''model parms'''
d_input = 512
n_layers = 6
n_aspect = 24
embed_dim = 300
hidden_dim = 300
context_dim = 300
output_dim = 50
dropout = 0.1
epsilon = 1e-06
need_pos = False  # if need position information

'''dataset params'''
clas_sr = 0.5  # classify data split data
weak_sr = 0.7  # weakly data split rate
sample_size = 100000
weak_test_samples = 8000
neg_size = 20
apriori_test_size = 10
maxlen_ = 300  # 300 for word embedding
maxlen = 302  # max length of sentence, 300 for word embedding, 1 for aspect index, 1 for real length
maxlen_asp = 20  # max length of aspect
pad_idx = 109316  # pad index 108947 previous
pp_data_weak = False
pp_data_clas = False
if_retain = True  # if retain original noun as aspects in sentence without aspects

'''others params'''
plot = False  # if use plot
save_model = True  # if save model
save_model_path = '/media/sysu2018/4TBDisk/william/corner_weakly_model/tmp/'

'''Automatically choose GPU or CPU'''
device_dict = {
    -1: 'cpu',
    0: 'cuda:0',
    1: 'cuda:1',
    2: 'cuda:2',
}
if torch.cuda.is_available():
    os.system('nvidia-smi -q -d Utilization | grep Gpu > log/gpu')
    util_gpu = [int(line.strip().split()[2]) for line in open('log/gpu', 'r')]

    gpu_count = torch.cuda.device_count()
    device_choose = [i for i in range(gpu_count)]
    device = util_gpu.index(min(util_gpu))
    # device = 0
else:
    device_choose = []
    device = -1
device = device_dict[device]
device_choose.append(-1)

'''train type choices'''
from train import Instructor

inst = Instructor()
TRAIN_TYPE_CHO = {
    'weak_train': inst.weakly_train,
    'clas_train': inst.classification_train,
    'clas_train_fix': inst.classification_train_fix,
    'align_clas_train': inst.align_classification_train,
    'clas_train_crf': inst.classification_train_crf,
    'align_clas_train_crf': inst.align_classification_train_crf,
    'aspect_train': inst.aspect_train,
}


def init_config(opt):
    global clas_lr, weak_lr, epoch, batch_size, save_mode, train_type, train_phase, \
        clas_model, lambda_, pretrained_model
    global d_input, n_layers, embed_dim, hidden_dim, dropout, need_pos
    global clas_sr, weak_sr, sample_size, weak_test_samples, neg_size \
        , pp_data_weak, pp_data_clas
    global plot, save_model, device

    epoch = opt.epoch
    clas_lr = opt.clas_lr
    weak_lr = opt.weak_lr
    batch_size = opt.batch_size
    d_input = opt.d_input
    n_layers = opt.n_layers
    dropout = opt.dropout
    save_model = opt.save_model
    save_mode = opt.save_mode
    train_type = opt.train_type
    train_phase = opt.train_phase
    pp_data_weak = opt.pp_data_weak
    pp_data_clas = opt.pp_data_clas
    sample_size = opt.sample_size
    clas_sr = opt.clas_sr
    weak_sr = opt.weak_sr
    weak_test_samples = opt.weak_test_samples
    # clas_model = opt.clas_model
    pretrained_model = opt.pretrained_model
    lambda_ = opt.lambda_
    neg_size = opt.neg_size
    need_pos = opt.need_pos
    device = opt.device
