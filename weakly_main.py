from __future__ import unicode_literals, print_function, division
import string
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import model
import time
import argparse
import dataProcess
import Train


class DataProcess(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def cal_performance(run, valid, epoch):
    ''' Apply label smoothing if needed '''
    run.zero_grad()
    run = run.eval()
    valid_now = valid(run)
    print('epoch {} of {}: TEST : {}'.format(epoch, 100, valid_now))
    return valid_now


def prepare_dataloaders_weakly(all_data, test_pos, test_neg, final_embedding, opt):
    all_data_train = all_data
    test_p = test_pos
    test_n = test_neg
    embed = final_embedding
    final_embedding = torch.from_numpy(embed)

    train_data = DataLoader(
        dataset=all_data_train,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    return final_embedding, train_data, test_p, test_n


def prepare_dataloaders_classification(data, opt):
    train_data, test_data, embedding = data
    final_embedding = embedding
    # add = -1 + 2 * np.random.random(300)
    # final_embedding = np.row_stack((final_embedding, add))
    embed = torch.from_numpy(final_embedding)

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    test_data_loader = DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    return embed, train_data_loader, test_data_loader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-learning_rate', type=float, default=64)
    parser.add_argument('-d_input', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    # parser.add_argument('-save_model', default=None)
    # parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-train_model', type=int, default=0, help='0->weakly training, 1->classification training')

    parser.add_argument('-preparedata_weakly', default=False)
    parser.add_argument('-sample_size', default=100000)
    parser.add_argument('-preparedata_classification', default=False)
    parser.add_argument('-split_rate', default=0.7)
    parser.add_argument('-weakly_sr', default=0.8)
    parser.add_argument('-weakly_test_samples', default=5000)
    parser.add_argument('-classify_model', default=model.SentimentClassification)
    parser.add_argument('-pre_train_model', default=model.WeaklyTrainModel)
    parser.add_argument('-trained_model', default="pretrainmodel/loss_1.63158valid_1.5559.pkl")
    parser.add_argument('-classify_lr', default=0.0001)
    parser.add_argument('-need_pos', default=False)
    parser.add_argument('-save_model', default=True)
    # parser.add_argument('-device', default=torch.device("cpu"))

    opt = parser.parse_args()
    # opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') \
    #     if opt.device is None else torch.device(opt.device)
    opt.device = torch.device('cuda:1')

    # dataProcess.weakly_data_process(opt)

    '''
    After Training process of weakly supervise learning
    Begin sentiment Classification learning
    '''
    # all_data, final_embedding, np.array(input_pos_test[0:opt.weakly_test_samples, :]), np.array(
    #     input_neg_test[0:opt.weakly_test_samples, :])

    all_data, final_embedding, test_pos, test_neg = dataProcess.weakly_data_process(opt)
    embedding, train_data, test_pos, test_neg = \
        prepare_dataloaders_weakly(all_data, test_pos, test_neg, final_embedding, opt)
    compare_acc = []
    # acc = Train.classification_train(opt.classify_model, opt, train_data_loader, test_data_loader, embedding)
    acc = Train.weakly_train(opt.pre_train_model, train_data, test_pos, test_neg, embedding, opt)
    compare_acc.append(acc)

    plot = False

    if plot is True:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        acc = Train.classification_train(opt.classify_model, opt
                                         , train_data_loader, test_data_loader, embedding, pretrain=False)
        compare_acc.append(acc)
        plt.figure()
        plt.plot(compare_acc[0], 'r', marker='o')
        plt.plot(compare_acc[1], 'b', marker='*')
        plt.ylim((0.7, 0.9))
        plt.show()
