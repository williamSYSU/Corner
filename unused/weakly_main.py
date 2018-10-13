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
# import data_prepare
# import Train
#
# import config
# from data_util import DataPrepare, CornerData
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
#     parser.add_argument('-device', type=str, default=config.device)
#
#     opt = parser.parse_args()
#     '''
#     After Training process of weakly supervise learning
#     Begin sentiment Classification learning
#     '''
#
#     '''prepare data'''
#     data_prepare = DataPrepare()
#     my_loader = CornerData()
#
#     '''obtain weakly test, valid and test data and dataloader'''
#     all_data, final_embedding, test_pos, test_neg = data_prepare.weakly_data_process
#     embedding, train_dataloader = my_loader.pp_dataloader_weak(all_data, final_embedding)
#
#     '''calculate accuracy'''
#     compare_acc = []
#     # acc = Train.classification_train(opt.classify_model, opt, train_data_loader, test_data_loader, embedding)
#     acc = Train.weakly_train(opt.pre_train_model, train_dataloader, test_pos, test_neg, embedding, opt)
#     compare_acc.append(acc)
#
#     if config.plot:
#         import matplotlib.pyplot as plt
#         import matplotlib.ticker as ticker
#
#         acc = Train.classification_train(opt.classify_model, opt
#                                          , train_dataloader, test_dataloader, embedding, pretrain=False)
#         compare_acc.append(acc)
#         plt.figure()
#         plt.plot(compare_acc[0], 'r', marker='o')
#         plt.plot(compare_acc[1], 'b', marker='*')
#         plt.ylim((0.7, 0.9))
#         plt.show()
