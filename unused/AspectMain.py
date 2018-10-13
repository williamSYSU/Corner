# from __future__ import unicode_literals, print_function, division
# import string
# import random
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import torch.nn as nn
# import models
# import time
# import argparse
# import dataProcess
# import AspectTrain
#
# import config
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
# def prepare_dataloaders_aspect(data, opt):
#     train_data_, embedding_ = data
#     embedding_ = torch.from_numpy(embedding_)
#     train_data_loader = DataLoader(
#         dataset=train_data_,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         drop_last=True,
#         num_workers=4
#     )
#     return embedding_, train_data_loader
#
#
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-epoch', type=int, default=config.epoch)
#     parser.add_argument('-batch_size', type=int, default=config.batch_size)
#     parser.add_argument('-d_input', type=int, default=config.d_input)
#     parser.add_argument('-n_layers', type=int, default=config.n_layers)
#     parser.add_argument('-dropout', type=float, default=config.dropout)
#     parser.add_argument('-save_model', default=config.save_mode)
#     parser.add_argument('-save_mode', type=str, default=config.save_mode)
#     # parser.add_argument('-no_cuda', action='store_true')
#     parser.add_argument('-train_type', type=int, default=config.train_type,
#                         help='0->weakly training, 1->classification training')
#     parser.add_argument('-pp_data_weak', default=config.pp_data_weak)
#     parser.add_argument('-pp_data_clas', default=config.pp_data_clas)
#     parser.add_argument('-sample_size', default=config.sample_size)
#     parser.add_argument('-clas_sr', type=float, default=config.clas_sr)
#     parser.add_argument('-weak_sr', type=float, default=config.weak_sr)
#     parser.add_argument('-weak_test_samples', default=config.weak_test_samples)
#     parser.add_argument('-clas_model', default=config.clas_model)
#     parser.add_argument('-pretrained_model', default=config.pretrained_model)
#     parser.add_argument('-clas_lr', default=config.clas_lr)
#     parser.add_argument('-weak_lr', default=config.weak_lr)
#     parser.add_argument('-need_pos', default=config.need_pos)
#     parser.add_argument('-neg_size', default=20)
#     parser.add_argument('-lambda_', default=1)
#     parser.add_argument('-device', type=str, default=config.device)
#
#     opt = parser.parse_args()
#
#     '''
#     After Training process of weakly supervise learning
#     Begin sentiment Classification learning
#     '''
#
#     train_data, final_embedding, sentence = dataProcess.aspect_extract_data_process(opt)
#
#     embedding, train_data_loader = prepare_dataloaders_aspect([train_data, final_embedding], opt)
#
#     AspectTrain.aspect_train(opt.classify_model, train_data_loader, embedding, sentence, opt)
#
#
