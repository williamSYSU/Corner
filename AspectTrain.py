from __future__ import unicode_literals, print_function, division
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import argparse
import torch.optim as optim
import torch.nn as nn
import numpy as np
import dataProcess
import model
from model.marginloss import MaxMargin


class DataProcess(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def aspect_train(model, train_data, embed, sentence, opt):
    init_aspect = np.array(np.load("initAspect.npy"))
    # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
    init_aspect = torch.from_numpy(init_aspect)

    run = model.ABAE(300, 24, init_aspect, embed, opt).to(opt.device)

    # loss_func = torch.nn.TripletMarginLoss(margin=1, p=2)
    loss_func = MaxMargin(opt)
    # params = []
    # for param in run.parameters():
    #     if param.requires_grad:
    #         params.append(param)

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, run.parameters()), lr=0.0001)
    # bb = filter(lambda p: p.requires_grad, run.parameters())
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, run.parameters()), lr=0.001)
    par = run.state_dict()
    # print("init aspect matrix : ", par)
    # print("init aspect matrix : ", par["aspect_lookup_mat"])
    min_loss = float('inf')

    for epoch in range(501):
        loss_last = torch.tensor([0], dtype=torch.float)
        optimizer.zero_grad()
        # run.zero_grad()
        run = run.train()
        loss_all = 0
        for idx, sample_batch in enumerate(train_data):
            indices = np.random.choice(len(sentence), opt.batch_size * opt.neg_size)
            samples = sentence[indices].reshape(opt.batch_size, opt.neg_size, 302)
            samples = torch.from_numpy(samples).to(opt.device)
            # now = time.time()
            input_ = sample_batch['input'].to(opt.device)
            out, sentence_attention, nag_samples, reg = run(input_, samples)
            loss_last = loss_func(sentence_attention, nag_samples, out) + opt.lambda_ * reg
            # loss_last = loss_last / opt.batch_size
            loss_last.backward()
            optimizer.step()
            loss_all += loss_last
        if loss_all < min_loss:
            print("save model......")
            file_name = "AspectExtract/Aspect_Model.pkl"
            torch.save(run.state_dict(), file_name)
            min_loss = loss_all
        print('epoch {} of {}: loss : {}'.format(epoch, 500, loss_last.item()))
