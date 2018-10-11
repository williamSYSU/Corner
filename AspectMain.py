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
import AspectTrain


class DataProcess(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def prepare_dataloaders_aspect(data, opt):
    train_data_, embedding_ = data
    embedding_ = torch.from_numpy(embedding_)
    train_data_loader = DataLoader(
        dataset=train_data_,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    return embedding_, train_data_loader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-learning_rate', type=float, default=64)
    parser.add_argument('-d_input', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-train_model', type=int, default=1, help='0->weakly training, 1->classification training')

    parser.add_argument('-preparedata_weakly', default=False)
    parser.add_argument('-sample_size', default=2000)
    parser.add_argument('-preparedata_classification', default=False)
    parser.add_argument('-split_rate', default=0.7)
    parser.add_argument('-weakly_sr', default=0.8)
    parser.add_argument('-weakly_test_samples', default=1000)
    parser.add_argument('-classify_model', default=model.ABAE)
    parser.add_argument('-trained_model', default="pretrainmodel/newmodel/loss_1.48206valid_1.19662.pkl")
    parser.add_argument('-classify_lr', default=0.0002)
    parser.add_argument('-need_pos', default=False)
    parser.add_argument('-neg_size', default=20)
    parser.add_argument('-lambda_', default=1)
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

    train_data, final_embedding, sentence = dataProcess.aspect_extract_data_process(opt)

    embedding, train_data_loader = prepare_dataloaders_aspect([train_data, final_embedding], opt)

    AspectTrain.aspect_train(opt.classify_model, train_data_loader, embedding, sentence, opt)


