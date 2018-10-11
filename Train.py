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


class DataProcess(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def weakly_train(model, train_data, test_pos, test_neg, embed, opt):
    # run = model.AttentionEncoder(300, 300, 50, embed, opt).to(opt.device)

    init_aspect = np.array(np.load("initAspect.npy"))
    # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
    init_aspect = torch.from_numpy(init_aspect)
    PreTrainABAE = model.PreTrainABAE(300, 24, init_aspect, embed, opt).to(opt.device)

    pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
    aspect_dict = PreTrainABAE.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
    aspect_dict.update(pre_trained_dict)
    PreTrainABAE.load_state_dict(aspect_dict)
    PreTrainABAE = PreTrainABAE.eval()

    trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

    run = model.WdeRnnEncoderFix(300, 300, 50, embed, trained_aspect, opt).to(opt.device)
    # context = torch.ones((opt.batch_size, 50))
    # optimizer = optim.Adagrad(params, lr=0.003)
    # optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, run.parameters()), lr=opt.learning_rate)
    loss_func = torch.nn.TripletMarginLoss(margin=4.0, p=2)
    # params = []
    # for param in run.parameters():
    #     if param.requires_grad:
    #         params.append(param)

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, run.parameters()), lr=0.0001)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, run.parameters()), lr=0.001)

    for epoch in range(501):
        run_hidden = run.initHidden(opt.batch_size)
        loss_last = torch.tensor([0], dtype=torch.float)
        optimizer.zero_grad()
        # run.zero_grad()
        for idx, sample_batch in enumerate(train_data):
            # now = time.time()
            run = run.train()
            input1 = sample_batch['input1'].to(opt.device)
            input2 = sample_batch['input2'].to(opt.device)
            input3 = sample_batch['input3'].to(opt.device)
            aspect_info = PreTrainABAE(input1)
            input1[:, 1] = aspect_info
            aspect_info = PreTrainABAE(input2)
            input2[:, 1] = aspect_info
            aspect_info = PreTrainABAE(input3)
            input3[:, 1] = aspect_info
            out1 = run(input1, run_hidden)
            out2 = run(input2, run_hidden)
            out3 = run(input3, run_hidden)
            loss_last = loss_func(out1, out2, out3)
            loss_last.backward()
            optimizer.step()
        # if epoch % 5 == 0:
        #     run.zero_grad()
        #     run = run.eval()
        #     valid_now = valid(PreTrainABAE, run, test_pos, test_neg, embed, opt)
        #     a = round((loss_last).item(), 5)
        #     b = round(valid_now, 5)
        #     if opt.save_model is True and valid_now > 1.13:
        #         file_name = "/media/sysu2018/4TBDisk/bcfox/" + "model_loss_" + str(a) + "valid_" + str(b) + ".pkl"
        #         torch.save(run.state_dict(), file_name)
        #
        #     print('epoch {} of {}: TEST : {}'.format(epoch, 500, valid_now))
        print('epoch {} of {}: loss : {}'.format(epoch, 500, loss_last.item()))


def classification_train(model, opt, train_data, valid_data, test_data, embed, pretrain=True):

    init_aspect = np.array(np.load("initAspect.npy"))
    # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
    init_aspect = torch.from_numpy(init_aspect)
    PreTrainABAE = model.PreTrainABAE(300, 24, init_aspect, embed, opt).to(opt.device)

    pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
    aspect_dict = PreTrainABAE.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
    aspect_dict.update(pre_trained_dict)
    PreTrainABAE.load_state_dict(aspect_dict)
    PreTrainABAE = PreTrainABAE.eval()

    trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

    run = model.WdeRnnEncoder(300, 300, 50, embed, trained_aspect, opt).to(opt.device)
    # params = []
    # for param in run.parameters():
    #     if param.requires_grad:
    #         params.append(param)
    # 加载预训练权重
    if pretrain is True:
        pre_trained_dict = torch.load(opt.trained_model)
        # pre_trained_dict = torch.load(opt.trained_model, map_location=lambda storage, loc: storage)
        model_dict = run.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
        model_dict.update(pre_trained_dict)
        run.load_state_dict(model_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, run.parameters()), lr=opt.classify_lr)
    all_evaluate = []
    best_test = 0
    for epoch in range(opt.epoch + 1):
        run_hidden = run.initHidden(opt.batch_size)
        # context = torch.ones((opt.batch_size, 50))
        # loss_last = torch.tensor([0], dtype=torch.float)
        optimizer.zero_grad()
        run.zero_grad()
        for idx, sample_batch in enumerate(train_data):
            run = run.train()
            input_data = sample_batch['input'].to(opt.device)
            label = sample_batch['label'].to(opt.device)
            aspect_info, _, _ = PreTrainABAE(input_data)
            input_data[:, 1] = aspect_info
            out = run(input_data, run_hidden).view(opt.batch_size, 2).to(opt.device)
            # print("result :", out.size())
            # print(label)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
        # if epoch % 5 == 0:
        #     run.zero_grad()
        #     run = run.eval()
        #     valid_now = self.valid(run)
        #     print('epoch {} of {}: TEST : {}'.format(epoch, 100, valid_now))
        print('epoch {} of {}: loss : {}'.format(epoch, opt.epoch, loss))

        if epoch % 1 == 0:
            with torch.no_grad():
                total = 0
                correct = 0
                optimizer.zero_grad()
                run.zero_grad()
                run_hidden = run.initHidden(1)
                # context = torch.ones((1, 50))
                for index, sample_batch in enumerate(valid_data):
                    run = run.eval()
                    input_data = sample_batch['input'].to(opt.device)
                    label = sample_batch['label'].to(opt.device)
                    aspect_info, _, _ = PreTrainABAE(input_data)
                    input_data[:, 1] = aspect_info
                    outputs = run(input_data, run_hidden).view(1, 2).to(opt.device)
                    _, predicted = torch.max(outputs.data, 1)
                    # print(outputs)
                    # print(predicted)
                    # print(label)
                    total += label.size(0)
                    # print(total)
                    correct += (predicted == label).sum().item()
                    # print(correct)
                acc = correct / total
                print("acc rate :", acc)

                if acc > best_test:
                    best_test = acc
                    file_name = "ClassifyModelSave/Final_model.pkl"
                    torch.save(run.state_dict(), file_name)

                all_evaluate.append(acc)

    '''
    Load the best model and Begin test
    '''
    model_test = model.WdeRnnEncoder(300, 300, 50, embed, trained_aspect, opt).to(opt.device)

    pre_trained_dict = torch.load("ClassifyModelSave/Final_model.pkl")
    model_dict = model_test.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
    model_dict.update(pre_trained_dict)
    model_test.load_state_dict(model_dict)

    with torch.no_grad():
        total = 0
        correct = 0
        model_test.zero_grad()
        run_hidden = model_test.initHidden(1)
        # context = torch.ones((1, 50))
        for index, sample_batch in enumerate(test_data):
            model_test = model_test.eval()
            input_data = sample_batch['input'].to(opt.device)
            label = sample_batch['label'].to(opt.device)
            aspect_info, _, _ = PreTrainABAE(input_data)
            input_data[:, 1] = aspect_info
            outputs = model_test(input_data, run_hidden).view(1, 2).to(opt.device)
            _, predicted = torch.max(outputs.data, 1)
            # print(outputs)
            # print(predicted)
            # print(label)
            total += label.size(0)
            # print(total)
            correct += (predicted == label).sum().item()
            # print(correct)
        acc = correct / total
        print("Test acc rate (final result) :", acc)

    return all_evaluate


def classification_train_fix(model, opt, train_data, test_data, embed, pretrain=True):
    run = model.AttentionEncoder(300, 300, 50, embed, opt).to(opt.device)
    params = []
    for param in run.parameters():
        if param.requires_grad:
            params.append(param)
    # 加载预训练权重
    if pretrain is True:
        pre_trained_dict = torch.load(opt.trained_model, map_location=lambda storage, loc: storage)
        model_dict = run.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
        model_dict.update(pre_trained_dict)
        run.load_state_dict(model_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params, lr=opt.classify_lr)
    all_evaluate = []

    for epoch in range(301):
        context = torch.ones((opt.batch_size, 50))
        # loss_last = torch.tensor([0], dtype=torch.float)
        optimizer.zero_grad()
        # run.zero_grad()
        for idx, sample_batch in enumerate(train_data):
            run = run.train()
            input_data = sample_batch['input'].to(opt.device)
            label = sample_batch['label'].to(opt.device)
            out = run(input_data, context.cuda()).view(opt.batch_size, 2).to(opt.device)
            # print("result :", out.size())
            # print(label)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
        # if epoch % 5 == 0:
        #     run.zero_grad()
        #     run = run.eval()
        #     valid_now = self.valid(run)
        #     print('epoch {} of {}: TEST : {}'.format(epoch, 100, valid_now))
        print('epoch {} of {}: loss : {}'.format(epoch, 300, loss))

        if epoch % 5 == 0:
            with torch.no_grad():
                total = 0
                correct = 0
                optimizer.zero_grad()
                run.zero_grad()
                context = torch.ones((1, 50))
                for index, sample_batch in enumerate(test_data):
                    run = run.eval()
                    input_data = sample_batch['input'].to(opt.device)
                    label = sample_batch['label'].to(opt.device)
                    outputs = run(input_data, context.cuda()).view(1, 2).to(opt.device)
                    _, predicted = torch.max(outputs.data, 1)
                    # print(outputs)
                    # print(predicted)
                    # print(label)
                    total += label.size(0)
                    # print(total)
                    correct += (predicted == label).sum().item()
                    # print(correct)
                print("acc rate :", correct / total)
                all_evaluate.append(correct / total)
    return all_evaluate


def valid(PreTrainABAE, model_trained, pos_test, neg_test, embed, opt):
    PreTrainABAE = PreTrainABAE.eval()

    with torch.no_grad():
        pos_len = len(pos_test)
        neg_len = len(neg_test)
        #         print("1")
        # context = torch.ones((1, 50))
        run_hidden = model_trained.initHidden(1)
        for idx, sentence in enumerate(pos_test):
            if idx == 0:
                sentence = torch.from_numpy(sentence).view(1, -1).to(opt.device)
                sentence[:, 1] = PreTrainABAE(sentence)
                pos_embedding = model_trained(sentence, run_hidden).view(1, 300)
            else:
                sentence = torch.from_numpy(sentence).view(1, -1).to(opt.device)
                sentence[:, 1] = PreTrainABAE(sentence)
                pos_embedding = torch.cat((
                    pos_embedding,
                    model_trained(sentence, run_hidden).view(1, 300)
                ),
                    dim=0
                )
        # print(pos_embedding.size())
        #         print("2")
        for idx, sentence in enumerate(neg_test):
            if idx == 0:
                sentence = torch.from_numpy(sentence).view(1, -1).to(opt.device)
                sentence[:, 1] = PreTrainABAE(sentence)
                neg_embedding = model_trained(sentence, run_hidden).view(1, 300)
            else:
                sentence = torch.from_numpy(sentence).view(1, -1).to(opt.device)
                sentence[:, 1] = PreTrainABAE(sentence)
                neg_embedding = torch.cat((
                    neg_embedding,
                    model_trained(sentence, run_hidden).view(1, 300)
                ),
                    dim=0
                )

        pos_embedding = pos_embedding.cpu()
        neg_embedding = neg_embedding.cpu()
        pos_embedding = pos_embedding.detach().numpy()
        neg_embedding = neg_embedding.detach().numpy()
        #         print("pos :", pos_embedding[0])
        #         print("neg :", neg_embedding[0])

        #         print("3")
        # pos_embedding = model_trained(torch.from_numpy(self.pos_test), run_hidden, context)
        # neg_embedding = model_trained(torch.from_numpy(self.neg_test), run_hidden, context)
        pos_dis = 0
        neg_dis = 0
        pos_count = 0
        neg_count = 0
        for idx, sentence1 in enumerate(pos_embedding[:-1]):
            for sentence2 in pos_embedding[idx + 1:]:
                dis = sentence1 - sentence2
                dis = dis * dis
                dis = np.sum(dis)
                dis = dis ** 0.5
                pos_dis += dis ** 0.5
                pos_count += 1
        #         print("4")
        for idx, sentence1 in enumerate(neg_embedding[:-1]):
            for sentence2 in neg_embedding[idx + 1:]:
                dis = sentence1 - sentence2
                dis = dis * dis
                dis = np.sum(dis)
                dis = dis ** 0.5
                neg_dis += dis ** 0.5
                neg_count += 1

        #         print("5")
        # valid_intra = 1 / pos_len * pos_dis + 1 / neg_len * neg_dis
        valid_intra = (pos_dis + neg_dis) / (pos_count + neg_count)

        inter_dis = 0
        for idx, sentence1 in enumerate(pos_embedding):
            for sentence2 in neg_embedding:
                dis = sentence1 - sentence2
                dis = dis * dis
                dis = np.sum(dis)
                dis = dis ** 0.5
                inter_dis += dis ** 0.5

        valid_inter = inter_dis / (pos_len * neg_len)
    #         print("6")
    return valid_inter / valid_intra


def cal_distence(sentence1, sentence2):
    distance = torch.dist(sentence1, sentence2, 2)
    return distance


def align_classification_train(model, opt, train_data, valid_data, test_data, embed, pretrain=True):

    init_aspect = np.array(np.load("initAspect.npy"))
    # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
    init_aspect = torch.from_numpy(init_aspect)
    PreTrainABAE = model.PreTrainABAE(300, 24, init_aspect, embed, opt).to(opt.device)

    pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
    aspect_dict = PreTrainABAE.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
    aspect_dict.update(pre_trained_dict)
    PreTrainABAE.load_state_dict(aspect_dict)
    # PreTrainABAE = PreTrainABAE.eval()

    trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

    run = model.align_WdeRnnEncoder(300, 300, 50, embed, trained_aspect, opt).to(opt.device)
    # params = []
    # for param in run.parameters():
    #     if param.requires_grad:
    #         params.append(param)
    # 加载预训练权重
    if pretrain is True:
        pre_trained_dict = torch.load(opt.trained_model)
        # pre_trained_dict = torch.load(opt.trained_model, map_location=lambda storage, loc: storage)
        model_dict = run.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
        model_dict.update(pre_trained_dict)
        run.load_state_dict(model_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.SGD(filter(lambda p: p.requires_grad, run.parameters()), lr=opt.classify_lr)
    optimizer_abae = optim.SGD(filter(lambda p: p.requires_grad, PreTrainABAE.parameters()), lr=opt.classify_lr)
    all_evaluate = []
    best_test = 0
    for epoch in range(opt.epoch + 1):
        run_hidden = run.initHidden(opt.batch_size)
        # context = torch.ones((opt.batch_size, 50))
        # loss_last = torch.tensor([0], dtype=torch.float)
        optimizer_rnn.zero_grad()
        optimizer_abae.zero_grad()
        run.zero_grad()
        for idx, sample_batch in enumerate(train_data):
            run = run.train()
            PreTrainABAE = PreTrainABAE.train()
            input_data = sample_batch['input'].to(opt.device)
            label = sample_batch['label'].to(opt.device)
            aspect_info, trained_aspect, reg = PreTrainABAE(input_data)
            input_data[:, 1] = aspect_info
            out = run(input_data, run_hidden, trained_aspect).view(opt.batch_size, 2).to(opt.device)
            # print("result :", out.size())
            # print(label)
            # loss = criterion(out, label) + reg.float()
            loss = criterion(out, label)
            loss.backward()
            optimizer_rnn.step()
            optimizer_abae.step()
        # if epoch % 5 == 0:
        #     run.zero_grad()
        #     run = run.eval()
        #     valid_now = self.valid(run)
        #     print('epoch {} of {}: TEST : {}'.format(epoch, 100, valid_now))
        print('epoch {} of {}: loss : {}'.format(epoch, opt.epoch, loss))

        if epoch % 1 == 0:
            with torch.no_grad():
                total = 0
                correct = 0
                optimizer_rnn.zero_grad()
                optimizer_abae.zero_grad()
                run.zero_grad()
                PreTrainABAE.zero_grad()
                run_hidden = run.initHidden(1)
                # context = torch.ones((1, 50))
                for index, sample_batch in enumerate(valid_data):
                    run = run.eval()
                    PreTrainABAE = PreTrainABAE.eval()
                    input_data = sample_batch['input'].to(opt.device)
                    label = sample_batch['label'].to(opt.device)
                    aspect_info, trained_aspect, _ = PreTrainABAE(input_data)
                    input_data[:, 1] = aspect_info
                    outputs = run(input_data, run_hidden, trained_aspect).view(1, 2).to(opt.device)
                    _, predicted = torch.max(outputs.data, 1)
                    # print(outputs)
                    # print(predicted)
                    # print(label)
                    total += label.size(0)
                    # print(total)
                    correct += (predicted == label).sum().item()
                    # print(correct)
                acc = correct / total
                print("acc rate :", acc)

                if acc > best_test:
                    best_test = acc
                    file_name = "ClassifyModelSave/Final_model.pkl"
                    file_name_aspect = "ClassifyModelSave/Final_model_aspect.pkl"
                    torch.save(run.state_dict(), file_name)
                    torch.save(PreTrainABAE.state_dict(), file_name_aspect)

                all_evaluate.append(acc)

    '''
    Load the best model and Begin test
    '''
    PreTrainABAE_test = model.PreTrainABAE(300, 24, init_aspect, embed, opt).to(opt.device)

    pre_trained_aspect = torch.load("ClassifyModelSave/Final_model_aspect.pkl")
    aspect_dict = PreTrainABAE_test.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
    aspect_dict.update(pre_trained_dict)
    PreTrainABAE_test.load_state_dict(aspect_dict)

    trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

    model_test = model.align_WdeRnnEncoder(300, 300, 50, embed, trained_aspect, opt).to(opt.device)

    pre_trained_dict = torch.load("ClassifyModelSave/Final_model.pkl")
    model_dict = model_test.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
    model_dict.update(pre_trained_dict)
    model_test.load_state_dict(model_dict)

    with torch.no_grad():
        total = 0
        correct = 0
        model_test.zero_grad()
        PreTrainABAE_test.zero_grad()
        run_hidden = model_test.initHidden(1)
        # context = torch.ones((1, 50))
        for index, sample_batch in enumerate(test_data):
            model_test = model_test.eval()
            input_data = sample_batch['input'].to(opt.device)
            label = sample_batch['label'].to(opt.device)
            aspect_info, trained_aspect, _ = PreTrainABAE_test(input_data)
            input_data[:, 1] = aspect_info
            outputs = model_test(input_data, run_hidden, trained_aspect).view(1, 2).to(opt.device)
            _, predicted = torch.max(outputs.data, 1)
            # print(outputs)
            # print(predicted)
            # print(label)
            total += label.size(0)
            # print(total)
            correct += (predicted == label).sum().item()
            # print(correct)
        acc = correct / total
        print("Test acc rate (final result) :", acc)

    return all_evaluate


def classification_train_crf(model, opt, train_data, valid_data, test_data, embed, pretrain=True):

    init_aspect = np.array(np.load("initAspect.npy"))
    # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
    init_aspect = torch.from_numpy(init_aspect)
    PreTrainABAE = model.PreTrainABAE(300, 24, init_aspect, embed, opt).to(opt.device)

    pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
    aspect_dict = PreTrainABAE.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
    aspect_dict.update(pre_trained_dict)
    PreTrainABAE.load_state_dict(aspect_dict)
    PreTrainABAE = PreTrainABAE.eval()

    trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

    run = model.RealAspectExtract(300, 300, 50, embed, trained_aspect, opt).to(opt.device)
    # params = []
    # for param in run.parameters():
    #     if param.requires_grad:
    #         params.append(param)
    # 加载预训练权重
    if pretrain is True:
        pre_trained_dict = torch.load(opt.trained_model)
        # pre_trained_dict = torch.load(opt.trained_model, map_location=lambda storage, loc: storage)
        model_dict = run.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
        model_dict.update(pre_trained_dict)
        run.load_state_dict(model_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, run.parameters()), lr=opt.classify_lr)
    all_evaluate = []
    best_test = 0
    for epoch in range(opt.epoch + 1):
        run_hidden = run.initHidden(opt.batch_size)
        # context = torch.ones((opt.batch_size, 50))
        # loss_last = torch.tensor([0], dtype=torch.float)
        optimizer.zero_grad()
        run.zero_grad()
        for idx, sample_batch in enumerate(train_data):
            run = run.train()
            input_data = sample_batch['input'].to(opt.device)
            label = sample_batch['label'].to(opt.device)
            aspect_info, _, _ = PreTrainABAE(input_data)
            input_data[:, 1] = aspect_info
            out = run(input_data, run_hidden, "train").view(opt.batch_size, 2).to(opt.device)
            # print("result :", outwww.size())
            # print(label)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
        # if epoch % 5 == 0:
        #     run.zero_grad()
        #     run = run.eval()
        #     valid_now = self.valid(run)
        #     print('epoch {} of {}: TEST : {}'.format(epoch, 100, valid_now))
        print('epoch {} of {}: loss : {}'.format(epoch, opt.epoch, loss))

        if epoch % 1 == 0:
            with torch.no_grad():
                total = 0
                correct = 0
                optimizer.zero_grad()
                run.zero_grad()
                run_hidden = run.initHidden(1)
                # context = torch.ones((1, 50))
                for index, sample_batch in enumerate(valid_data):
                    run = run.eval()
                    input_data = sample_batch['input'].to(opt.device)
                    label = sample_batch['label'].to(opt.device)
                    aspect_info, _, _ = PreTrainABAE(input_data)
                    input_data[:, 1] = aspect_info
                    outputs = run(input_data, run_hidden, "test").view(1, 2).to(opt.device)
                    _, predicted = torch.max(outputs.data, 1)
                    # print(outputs)
                    # print(predicted)
                    # print(label)
                    total += label.size(0)
                    # print(total)
                    correct += (predicted == label).sum().item()
                    # print(correct)
                acc = correct / total
                print("acc rate :", acc)

                if acc > best_test:
                    best_test = acc
                    file_name = "ClassifyModelSave/Final_model.pkl"
                    torch.save(run.state_dict(), file_name)

                all_evaluate.append(acc)

    '''
    Load the best model and Begin test
    '''
    model_test = model.RealAspectExtract(300, 300, 50, embed, trained_aspect, opt).to(opt.device)

    pre_trained_dict = torch.load("ClassifyModelSave/Final_model.pkl")
    model_dict = model_test.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
    model_dict.update(pre_trained_dict)
    model_test.load_state_dict(model_dict)

    with torch.no_grad():
        total = 0
        correct = 0
        model_test.zero_grad()
        run_hidden = model_test.initHidden(1)
        # context = torch.ones((1, 50))
        for index, sample_batch in enumerate(test_data):
            model_test = model_test.eval()
            input_data = sample_batch['input'].to(opt.device)
            label = sample_batch['label'].to(opt.device)
            aspect_info, _, _ = PreTrainABAE(input_data)
            input_data[:, 1] = aspect_info
            outputs = model_test(input_data, run_hidden, "test").view(1, 2).to(opt.device)
            _, predicted = torch.max(outputs.data, 1)
            # print(outputs)
            # print(predicted)
            # print(label)
            total += label.size(0)
            # print(total)
            correct += (predicted == label).sum().item()
            # print(correct)
        acc = correct / total
        print("Test acc rate (final result) :", acc)

    return all_evaluate


def align_classification_train_crf(model, opt, train_data, valid_data, test_data, embed, pretrain=True):

    init_aspect = np.array(np.load("initAspect.npy"))
    # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
    init_aspect = torch.from_numpy(init_aspect)
    PreTrainABAE = model.PreTrainABAE(300, 24, init_aspect, embed, opt).to(opt.device)

    pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
    aspect_dict = PreTrainABAE.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
    aspect_dict.update(pre_trained_dict)
    PreTrainABAE.load_state_dict(aspect_dict)
    # PreTrainABAE = PreTrainABAE.eval()

    trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

    run = model.CrfWdeRnnEncoder(300, 300, 50, embed, trained_aspect, opt).to(opt.device)
    # params = []
    # for param in run.parameters():
    #     if param.requires_grad:
    #         params.append(param)
    # 加载预训练权重
    if pretrain is True:
        pre_trained_dict = torch.load(opt.trained_model)
        # pre_trained_dict = torch.load(opt.trained_model, map_location=lambda storage, loc: storage)
        model_dict = run.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
        model_dict.update(pre_trained_dict)
        run.load_state_dict(model_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.SGD(filter(lambda p: p.requires_grad, run.parameters()), lr=opt.classify_lr)
    optimizer_abae = optim.SGD(filter(lambda p: p.requires_grad, PreTrainABAE.parameters()), lr=opt.classify_lr)
    all_evaluate = []
    best_test = 0
    for epoch in range(opt.epoch + 1):
        run_hidden = run.initHidden(opt.batch_size)
        # context = torch.ones((opt.batch_size, 50))
        # loss_last = torch.tensor([0], dtype=torch.float)
        optimizer_rnn.zero_grad()
        optimizer_abae.zero_grad()
        run.zero_grad()
        for idx, sample_batch in enumerate(train_data):
            run = run.train()
            PreTrainABAE = PreTrainABAE.train()
            input_data = sample_batch['input'].to(opt.device)
            label = sample_batch['label'].to(opt.device)
            aspect_info, trained_aspect, reg = PreTrainABAE(input_data)
            input_data[:, 1] = aspect_info
            out = run(input_data, run_hidden, trained_aspect, "train").view(opt.batch_size, 2).to(opt.device)
            # print("result :", out.size())
            # print(label)
            # loss = criterion(out, label) + reg.float()
            loss = criterion(out, label)
            loss.backward()
            optimizer_rnn.step()
            optimizer_abae.step()
        # if epoch % 5 == 0:
        #     run.zero_grad()
        #     run = run.eval()
        #     valid_now = self.valid(run)
        #     print('epoch {} of {}: TEST : {}'.format(epoch, 100, valid_now))
        print('epoch {} of {}: loss : {}'.format(epoch, opt.epoch, loss))

        if epoch % 1 == 0:
            with torch.no_grad():
                total = 0
                correct = 0
                optimizer_rnn.zero_grad()
                optimizer_abae.zero_grad()
                run.zero_grad()
                PreTrainABAE.zero_grad()
                run_hidden = run.initHidden(1)
                # context = torch.ones((1, 50))
                for index, sample_batch in enumerate(valid_data):
                    run = run.eval()
                    PreTrainABAE = PreTrainABAE.eval()
                    input_data = sample_batch['input'].to(opt.device)
                    label = sample_batch['label'].to(opt.device)
                    aspect_info, trained_aspect, _ = PreTrainABAE(input_data)
                    input_data[:, 1] = aspect_info
                    outputs = run(input_data, run_hidden, trained_aspect, "test").view(1, 2).to(opt.device)
                    _, predicted = torch.max(outputs.data, 1)
                    # print(outputs)
                    # print(predicted)
                    # print(label)
                    total += label.size(0)
                    # print(total)
                    correct += (predicted == label).sum().item()
                    # print(correct)
                acc = correct / total
                print("acc rate :", acc)

                if acc > best_test:
                    best_test = acc
                    file_name = "ClassifyModelSave/Final_model.pkl"
                    file_name_aspect = "ClassifyModelSave/Final_model_aspect.pkl"
                    torch.save(run.state_dict(), file_name)
                    torch.save(PreTrainABAE.state_dict(), file_name_aspect)

                all_evaluate.append(acc)

    '''
    Load the best model and Begin test
    '''
    PreTrainABAE_test = model.PreTrainABAE(300, 24, init_aspect, embed, opt).to(opt.device)

    pre_trained_aspect = torch.load("ClassifyModelSave/Final_model_aspect.pkl")
    aspect_dict = PreTrainABAE_test.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
    aspect_dict.update(pre_trained_dict)
    PreTrainABAE_test.load_state_dict(aspect_dict)

    trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

    model_test = model.CrfWdeRnnEncoder(300, 300, 50, embed, trained_aspect, opt).to(opt.device)

    pre_trained_dict = torch.load("ClassifyModelSave/Final_model.pkl")
    model_dict = model_test.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
    model_dict.update(pre_trained_dict)
    model_test.load_state_dict(model_dict)

    with torch.no_grad():
        total = 0
        correct = 0
        model_test.zero_grad()
        PreTrainABAE_test.zero_grad()
        run_hidden = model_test.initHidden(1)
        # context = torch.ones((1, 50))
        for index, sample_batch in enumerate(test_data):
            model_test = model_test.eval()
            input_data = sample_batch['input'].to(opt.device)
            label = sample_batch['label'].to(opt.device)
            aspect_info, trained_aspect, _ = PreTrainABAE_test(input_data)
            input_data[:, 1] = aspect_info
            outputs = model_test(input_data, run_hidden, trained_aspect, "test").view(1, 2).to(opt.device)
            _, predicted = torch.max(outputs.data, 1)
            # print(outputs)
            # print(predicted)
            # print(label)
            total += label.size(0)
            # print(total)
            correct += (predicted == label).sum().item()
            # print(correct)
        acc = correct / total
        print("Test acc rate (final result) :", acc)

    return all_evaluate


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-preparedata_weakly', default=False)
#     parser.add_argument('-sample_size', default=2000)
#     parser.add_argument('-preparedata_classification', default=False)
#     parser.add_argument('-split_rate', default=0.7)
#     parser.add_argument('-weakly_sr', default=0.8)
#     parser.add_argument('-weakly_test_samples', default=1000)
#     parser.add_argument('-classify_model', default=model.SentimentClassification.WdeRnnEncoder)
#     parser.add_argument('-trained_model', default="D:/mymodel/1/loss_0.12875valid_1.16588.pkl")
#     parser.add_argument('-classify_lr', default=0.001)
#     opt = parser.parse_args()
#     dataProcess.weakly_data_process(opt)
#     classify_train_data, classify_test_data, classify_final_embedding = dataProcess.classification_process(opt)
#
#
#     parser.add_argument()
#     print("Hello world")