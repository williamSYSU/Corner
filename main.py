from __future__ import unicode_literals, print_function, division

import argparse

import config
from data_util import DataPrepare, CornerData
from train import Instructor


class MainInstructor:
    def __init__(self):
        """prepare data and train instruction"""
        self.data_prepare = DataPrepare()
        self.my_loader = CornerData()
        self.instructor = Instructor()

    def start_(self):
        """decide train phase"""
        if config.train_phase == 'weakly':
            self.weak_train()
        elif config.train_phase == 'classify':
            self.clas_train()
        elif config.train_phase == 'aspect':
            self.aspect_train()

    def weak_train(self):
        """start weakly training"""
        '''obtain test, valid and test data and dataloader'''
        all_data, final_embedding, test_pos, test_neg = self.data_prepare.weakly_data_process
        embedding, train_dataloader = self.my_loader.pp_dataloader_weak(all_data, final_embedding)

        '''calculate accuracy'''
        compare_acc = []
        acc = self.instructor.weakly_train(train_dataloader, test_pos, test_neg, embedding)
        compare_acc.append(acc)

    def clas_train(self):
        """start classification training"""
        '''obtain test, valid and test data and dataloader'''
        clas_train, clas_valid, classify_test_data, classify_final_embedding \
            = self.data_prepare.clas_data_process
        embedding, train_dataloader, valid_dataloader, test_dataloader \
            = self.my_loader.pp_dataloader_clas(
            (clas_train,
             clas_valid,
             classify_test_data,
             classify_final_embedding))

        '''calculate accuracy'''
        compare_acc = []
        acc = self.instructor.classification_train(config.clas_model,
                                                   train_dataloader,
                                                   valid_dataloader,
                                                   test_dataloader,
                                                   embedding)
        # acc = Train.classification_train_fix(config.clas_model,
        #                                      train_dataloader,
        #                                      test_dataloader,
        #                                      embedding)
        compare_acc.append(acc)

        '''if use plot'''
        if config.plot:
            import matplotlib.pyplot as plt

            acc = self.instructor.classification_train(config.clas_model,
                                                       train_dataloader,
                                                       valid_dataloader,
                                                       test_dataloader,
                                                       embedding,
                                                       pretrain=False)
            compare_acc.append(acc)
            plt.figure()
            plt.plot(compare_acc[0], 'r', marker='o')
            plt.plot(compare_acc[1], 'b', marker='*')
            plt.ylim((0.7, 0.9))
            plt.show()

    def aspect_train(self):
        """start aspect extracting training"""
        '''obtain test, valid and test data and dataloader'''
        train_data, final_embedding, sentence = self.data_prepare.aspect_extract_data_process
        embedding, train_data_loader = self.my_loader.pp_dataloader_aspect((train_data, final_embedding))

        '''start training'''
        self.instructor.aspect_train(config.clas_model, train_data_loader, embedding, sentence)


if __name__ == "__main__":
    '''get params from command'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=config.epoch)
    parser.add_argument('-clas_lr', default=config.clas_lr)
    parser.add_argument('-weak_lr', default=config.weak_lr)
    parser.add_argument('-batch_size', type=int, default=config.batch_size)
    parser.add_argument('-d_input', type=int, default=config.d_input)
    parser.add_argument('-n_layers', type=int, default=config.n_layers)
    parser.add_argument('-dropout', type=float, default=config.dropout)
    parser.add_argument('-save_model', default=config.save_mode)
    parser.add_argument('-save_mode', type=str, default=config.save_mode)
    # parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-train_type', type=str, default=config.train_type)
    parser.add_argument('-train_phase', type=str, default=config.train_phase)
    parser.add_argument('-pp_data_weak', default=config.pp_data_weak)
    parser.add_argument('-pp_data_clas', default=config.pp_data_clas)
    parser.add_argument('-sample_size', default=config.sample_size)
    parser.add_argument('-clas_sr', type=float, default=config.clas_sr)
    parser.add_argument('-weak_sr', type=float, default=config.weak_sr)
    parser.add_argument('-weak_test_samples', default=config.weak_test_samples)
    # parser.add_argument('-clas_model', default=config.clas_model)
    parser.add_argument('-pretrained_model', default=config.pretrained_model)
    parser.add_argument('-lambda_', default=1)
    parser.add_argument('-neg_size', default=config.neg_size)
    parser.add_argument('-need_pos', default=config.need_pos)
    parser.add_argument('-device', type=str, default=config.device)

    opt = parser.parse_args()

    config.init_config(opt)  # initialize params

    '''
    After Training process of weakly supervise learning
    Begin sentiment Classification learning
    '''
    inst = MainInstructor()
    inst.start_()
