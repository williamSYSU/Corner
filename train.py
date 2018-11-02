from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.optim as optim

import config
import models.ABAE as asp_model
import models.SentimentClassification as clas_model
import models.WeaklyTrainModel as weak_model
from layers.marginloss import MaxMargin


class Instructor:
    def __init__(self):
        pass

    def weakly_train(self, train_data, test_pos, test_neg, embed, asp_list):
        # run = models.AttentionEncoder(300, 300, 50, embed).to(config.device)

        # init_aspect = np.array(np.load("initAspect.npy"))
        # # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
        # init_aspect = torch.from_numpy(init_aspect)
        # pre_train_abae = weak_model.PreTrainABAE(init_aspect, embed).to(config.device)
        #
        # pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
        # aspect_dict = pre_train_abae.state_dict()
        # pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
        # aspect_dict.update(pre_trained_dict)
        # pre_train_abae.load_state_dict(aspect_dict)
        # pre_train_abae = pre_train_abae.eval()
        #
        # trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

        # run = weak_model.WdeRnnEncoderFix(300, 300, 50, embed, trained_aspect).to(config.device)
        run = weak_model.WdeRnnEncoderFix(300, 300, 50, embed).to(config.device)
        # context = torch.ones((config.batch_size, 50))
        # optimizer = optim.Adagrad(params, lr=0.003)
        # params = []
        # for param in run.parameters():
        #     if param.requires_grad:
        #         params.append(param)

        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, run.parameters()), lr=0.0001)
        optimizer = config.optimizer(filter(lambda p: p.requires_grad, run.parameters()), lr=config.weak_lr)
        loss_func = config.criterion(margin=config.margin, p=config.margin_p)

        for epoch in range(config.epoch):
            run_hidden = run.initHidden(config.batch_size)
            loss_last = torch.tensor([0], dtype=torch.float)
            optimizer.zero_grad()
            # run.zero_grad()
            for idx, sample_batch in enumerate(train_data):
                # now = time.time()
                run = run.train()
                input1 = sample_batch['input1'].to(config.device)
                input2 = sample_batch['input2'].to(config.device)
                input3 = sample_batch['input3'].to(config.device)
                aspect1 = sample_batch['aspect1'].to(config.device)
                aspect2 = sample_batch['aspect2'].to(config.device)
                aspect3 = sample_batch['aspect3'].to(config.device)

                # get aspect info
                # aspect_info = pre_train_abae(input1)
                # input1[:, 1] = aspect_info
                # aspect_info = pre_train_abae(input2)
                # input2[:, 1] = aspect_info
                # aspect_info = pre_train_abae(input3)
                # input3[:, 1] = aspect_info

                # feed input data
                out1 = run(input1, run_hidden, aspect1).view(config.batch_size, 300)
                out2 = run(input2, run_hidden, aspect2).view(config.batch_size, 300)
                out3 = run(input3, run_hidden, aspect3).view(config.batch_size, 300)

                # count loss
                loss_last = loss_func(out1, out2, out3)
                loss_last.backward()
                optimizer.step()
            if epoch % config.valid_step == 0:
                run.zero_grad()
                run = run.eval()
                valid_now = self.valid(asp_list, run, test_pos, test_neg, embed)
                a = round((loss_last).item(), 5)
                b = round(valid_now, 5)
                if config.save_model and valid_now > config.valid_thres:
                    file_name = config.save_model_path + "model_loss_" + str(a) + "valid_" + str(b) + ".pkl"
                    torch.save(run.state_dict(), file_name)

                print('epoch {} of {}: TEST : {}'.format(epoch, config.epoch, valid_now))
            print('epoch {} of {}: loss : {}'.format(epoch, config.epoch, loss_last.item()))

    def classification_train(self, train_data, valid_data, test_data, embed, pretrain=True):
        # init_aspect = np.array(np.load("initAspect.npy"))
        # # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
        # init_aspect = torch.from_numpy(init_aspect)
        # pre_train_abae = clas_model.PreTrainABAE(init_aspect, embed).to(config.device)
        #
        # pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
        # aspect_dict = pre_train_abae.state_dict()
        # pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
        # aspect_dict.update(pre_trained_dict)
        # pre_train_abae.load_state_dict(aspect_dict)
        # pre_train_abae = pre_train_abae.eval()
        #
        # trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

        # run = clas_model.WdeRnnEncoder(300, 300, 50, embed, trained_aspect).to(config.device)
        run = clas_model.WdeRnnEncoder(300, 300, 50, embed).to(config.device)
        # params = []
        # for param in run.parameters():
        #     if param.requires_grad:
        #         params.append(param)
        # 加载预训练权重
        if pretrain is True:
            pre_trained_dict = torch.load(config.pretrained_model)
            # pre_trained_dict = torch.load(config.pretrained_model, map_location=lambda storage, loc: storage)
            model_dict = run.state_dict()
            pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
            model_dict.update(pre_trained_dict)
            run.load_state_dict(model_dict)

        optimizer = config.optimizer(filter(lambda p: p.requires_grad, run.parameters()), lr=config.clas_lr)
        criterion = config.criterion()
        all_evaluate = []
        best_test = 0
        for epoch in range(config.epoch):
            run_hidden = run.initHidden(config.batch_size)
            # context = torch.ones((config.batch_size, 50))
            # loss_last = torch.tensor([0], dtype=torch.float)
            optimizer.zero_grad()
            run.zero_grad()
            for idx, sample_batch in enumerate(train_data):
                run = run.train()
                input_data = sample_batch['input'].to(config.device)
                label = sample_batch['label'].to(config.device)
                aspect = sample_batch['aspect'].to(config.device)

                # origin
                # aspect_info, _, _ = pre_train_abae(input_data)
                # input_data[:, 1] = aspect_info

                out = run(input_data, run_hidden, aspect).view(config.batch_size, 2).to(config.device)
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
            print('epoch {} of {}: loss : {}'.format(epoch, config.epoch, loss))

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
                        input_data = sample_batch['input'].to(config.device)
                        label = sample_batch['label'].to(config.device)
                        aspect = sample_batch['aspect'].to(config.device)
                        # aspect_info, _, _ = pre_train_abae(input_data)
                        # input_data[:, 1] = aspect_info
                        outputs = run(input_data, run_hidden, aspect).view(1, 2).to(config.device)
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
        Load the best models and Begin test
        '''
        # model_test = clas_model.WdeRnnEncoder(300, 300, 50, embed, trained_aspect).to(config.device)
        model_test = clas_model.WdeRnnEncoder(300, 300, 50, embed).to(config.device)

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
                input_data = sample_batch['input'].to(config.device)
                label = sample_batch['label'].to(config.device)
                aspect = sample_batch['aspect'].to(config.device)
                # aspect_info, _, _ = pre_train_abae(input_data)
                # input_data[:, 1] = aspect_info
                outputs = model_test(input_data, run_hidden, aspect).view(1, 2).to(config.device)
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

    def classification_train_fix(self, train_data, test_data, embed, pretrain=True):
        run = clas_model.AttentionEncoder(300, 300, 50, embed).to(config.device)
        params = []
        for param in run.parameters():
            if param.requires_grad:
                params.append(param)
        # 加载预训练权重
        if pretrain is True:
            pre_trained_dict = torch.load(config.pretrained_model, map_location=lambda storage, loc: storage)
            model_dict = run.state_dict()
            pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
            model_dict.update(pre_trained_dict)
            run.load_state_dict(model_dict)

        optimizer = config.optimizer(params, lr=config.clas_lr)
        criterion = config.criterion
        all_evaluate = []

        for epoch in range(config.epoch):
            context = torch.ones((config.batch_size, 50))
            # loss_last = torch.tensor([0], dtype=torch.float)
            optimizer.zero_grad()
            # run.zero_grad()
            for idx, sample_batch in enumerate(train_data):
                run = run.train()
                input_data = sample_batch['input'].to(config.device)
                label = sample_batch['label'].to(config.device)
                out = run(input_data, context.cuda()).view(config.batch_size, 2).to(config.device)
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
            print('epoch {} of {}: loss : {}'.format(epoch, config.epoch, loss))

            if epoch % 5 == 0:
                with torch.no_grad():
                    total = 0
                    correct = 0
                    optimizer.zero_grad()
                    run.zero_grad()
                    context = torch.ones((1, 50))
                    for index, sample_batch in enumerate(test_data):
                        run = run.eval()
                        input_data = sample_batch['input'].to(config.device)
                        label = sample_batch['label'].to(config.device)
                        outputs = run(input_data, context.cuda()).view(1, 2).to(config.device)
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

    # def valid(self, PreTrainABAE, model_trained, pos_test, neg_test, embed):
    def valid(self, asp_list, model_trained, pos_test, neg_test, embed):
        # PreTrainABAE = PreTrainABAE.eval()

        with torch.no_grad():
            pos_len = len(pos_test)
            neg_len = len(neg_test)
            #         print("1")
            # context = torch.ones((1, 50))
            run_hidden = model_trained.initHidden(1)
            pos_embedding = None
            neg_embedding = None
            for idx, sentence in enumerate(pos_test):
                if idx == 0:
                    sentence = torch.from_numpy(sentence).view(1, -1).to(config.device)
                    # sentence[:, 1] = PreTrainABAE(sentence)
                    aspect = torch.from_numpy(asp_list[sentence[0][1]]).unsqueeze(0).to(config.device)
                    pos_embedding = model_trained(sentence, run_hidden, aspect).view(1, 300)
                else:
                    sentence = torch.from_numpy(sentence).view(1, -1).to(config.device)
                    # sentence[:, 1] = PreTrainABAE(sentence)
                    aspect = torch.from_numpy(asp_list[sentence[0][1]]).unsqueeze(0).to(config.device)
                    pos_embedding = torch.cat((
                        pos_embedding,
                        model_trained(sentence, run_hidden, aspect).view(1, 300)
                    ),
                        dim=0
                    )
            # print(pos_embedding.size())
            #         print("2")
            for idx, sentence in enumerate(neg_test):
                if idx == 0:
                    sentence = torch.from_numpy(sentence).view(1, -1).to(config.device)
                    # sentence[:, 1] = PreTrainABAE(sentence)
                    aspect = torch.from_numpy(asp_list[sentence[0][1]]).unsqueeze(0).to(config.device)
                    neg_embedding = model_trained(sentence, run_hidden, aspect).view(1, 300)
                else:
                    sentence = torch.from_numpy(sentence).view(1, -1).to(config.device)
                    # sentence[:, 1] = PreTrainABAE(sentence)
                    aspect = torch.from_numpy(asp_list[sentence[0][1]]).unsqueeze(0).to(config.device)
                    neg_embedding = torch.cat((
                        neg_embedding,
                        model_trained(sentence, run_hidden, aspect).view(1, 300)
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

    def cal_distence(self, sentence1, sentence2):
        distance = torch.dist(sentence1, sentence2, 2)
        return distance

    def align_classification_train(self, train_data, valid_data, test_data, embed, pretrain=True):
        init_aspect = np.array(np.load("initAspect.npy"))
        # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
        init_aspect = torch.from_numpy(init_aspect)
        pre_train_abae = clas_model.PreTrainABAE(init_aspect, embed).to(config.device)

        pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
        aspect_dict = pre_train_abae.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
        aspect_dict.update(pre_trained_dict)
        pre_train_abae.load_state_dict(aspect_dict)
        # PreTrainABAE = PreTrainABAE.eval()

        trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

        run = clas_model.align_WdeRnnEncoder(300, 300, 50, embed, trained_aspect).to(config.device)
        # params = []
        # for param in run.parameters():
        #     if param.requires_grad:
        #         params.append(param)
        # 加载预训练权重
        if pretrain is True:
            pre_trained_dict = torch.load(config.pretrained_model)
            # pre_trained_dict = torch.load(config.pretrained_model, map_location=lambda storage, loc: storage)
            model_dict = run.state_dict()
            pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
            model_dict.update(pre_trained_dict)
            run.load_state_dict(model_dict)

        criterion = config.criterion()
        optimizer_rnn = config.optimizer(filter(lambda p: p.requires_grad, run.parameters()), lr=config.clas_lr)
        optimizer_abae = config.optimizer(filter(lambda p: p.requires_grad, pre_train_abae.parameters()),
                                          lr=config.clas_lr)
        all_evaluate = []
        best_test = 0
        for epoch in range(config.epoch + 1):
            run_hidden = run.initHidden(config.batch_size)
            # context = torch.ones((config.batch_size, 50))
            # loss_last = torch.tensor([0], dtype=torch.float)
            optimizer_rnn.zero_grad()
            optimizer_abae.zero_grad()
            run.zero_grad()
            for idx, sample_batch in enumerate(train_data):
                run = run.train()
                pre_train_abae = pre_train_abae.train()
                input_data = sample_batch['input'].to(config.device)
                label = sample_batch['label'].to(config.device)
                aspect_info, trained_aspect, reg = pre_train_abae(input_data)
                input_data[:, 1] = aspect_info
                out = run(input_data, run_hidden, trained_aspect).view(config.batch_size, 2).to(config.device)
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
            print('epoch {} of {}: loss : {}'.format(epoch, config.epoch, loss))

            if epoch % 1 == 0:
                with torch.no_grad():
                    total = 0
                    correct = 0
                    optimizer_rnn.zero_grad()
                    optimizer_abae.zero_grad()
                    run.zero_grad()
                    pre_train_abae.zero_grad()
                    run_hidden = run.initHidden(1)
                    # context = torch.ones((1, 50))
                    for index, sample_batch in enumerate(valid_data):
                        run = run.eval()
                        pre_train_abae = pre_train_abae.eval()
                        input_data = sample_batch['input'].to(config.device)
                        label = sample_batch['label'].to(config.device)
                        aspect_info, trained_aspect, _ = pre_train_abae(input_data)
                        input_data[:, 1] = aspect_info
                        outputs = run(input_data, run_hidden, trained_aspect).view(1, 2).to(config.device)
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
                        torch.save(pre_train_abae.state_dict(), file_name_aspect)

                    all_evaluate.append(acc)

        '''
        Load the best models and Begin test
        '''
        PreTrainABAE_test = clas_model.PreTrainABAE(init_aspect, embed).to(config.device)

        pre_trained_aspect = torch.load("ClassifyModelSave/Final_model_aspect.pkl")
        aspect_dict = PreTrainABAE_test.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
        aspect_dict.update(pre_trained_dict)
        PreTrainABAE_test.load_state_dict(aspect_dict)

        trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

        model_test = clas_model.align_WdeRnnEncoder(300, 300, 50, embed, trained_aspect).to(config.device)

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
                input_data = sample_batch['input'].to(config.device)
                label = sample_batch['label'].to(config.device)
                aspect_info, trained_aspect, _ = PreTrainABAE_test(input_data)
                input_data[:, 1] = aspect_info
                outputs = model_test(input_data, run_hidden, trained_aspect).view(1, 2).to(config.device)
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

    def classification_train_crf(self, train_data, valid_data, test_data, embed, pretrain=True):
        init_aspect = np.array(np.load("initAspect.npy"))
        # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
        init_aspect = torch.from_numpy(init_aspect)
        PreTrainABAE = clas_model.PreTrainABAE(init_aspect, embed).to(config.device)

        pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
        aspect_dict = PreTrainABAE.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
        aspect_dict.update(pre_trained_dict)
        PreTrainABAE.load_state_dict(aspect_dict)
        PreTrainABAE = PreTrainABAE.eval()

        trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

        run = clas_model.RealAspectExtract(300, 300, 50, embed, trained_aspect).to(config.device)
        # params = []
        # for param in run.parameters():
        #     if param.requires_grad:
        #         params.append(param)
        # 加载预训练权重
        if pretrain is True:
            pre_trained_dict = torch.load(config.pretrained_model)
            # pre_trained_dict = torch.load(config.pretrained_model, map_location=lambda storage, loc: storage)
            model_dict = run.state_dict()
            pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
            model_dict.update(pre_trained_dict)
            run.load_state_dict(model_dict)

        criterion = config.criterion()
        optimizer = config.optimizer(filter(lambda p: p.requires_grad, run.parameters()), lr=config.clas_lr)
        all_evaluate = []
        best_test = 0
        for epoch in range(config.epoch + 1):
            run_hidden = run.initHidden(config.batch_size)
            # context = torch.ones((config.batch_size, 50))
            # loss_last = torch.tensor([0], dtype=torch.float)
            optimizer.zero_grad()
            run.zero_grad()
            for idx, sample_batch in enumerate(train_data):
                run = run.train()
                input_data = sample_batch['input'].to(config.device)
                label = sample_batch['label'].to(config.device)
                aspect_info, _, _ = PreTrainABAE(input_data)
                input_data[:, 1] = aspect_info
                out = run(input_data, run_hidden, "train").view(config.batch_size, 2).to(config.device)
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
            print('epoch {} of {}: loss : {}'.format(epoch, config.epoch, loss))

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
                        input_data = sample_batch['input'].to(config.device)
                        label = sample_batch['label'].to(config.device)
                        aspect_info, _, _ = PreTrainABAE(input_data)
                        input_data[:, 1] = aspect_info
                        outputs = run(input_data, run_hidden, "test").view(1, 2).to(config.device)
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
        Load the best models and Begin test
        '''
        model_test = clas_model.RealAspectExtract(300, 300, 50, embed, trained_aspect).to(config.device)

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
                input_data = sample_batch['input'].to(config.device)
                label = sample_batch['label'].to(config.device)
                aspect_info, _, _ = PreTrainABAE(input_data)
                input_data[:, 1] = aspect_info
                outputs = model_test(input_data, run_hidden, "test").view(1, 2).to(config.device)
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

    def align_classification_train_crf(self, train_data, valid_data, test_data, embed, pretrain=True):
        init_aspect = np.array(np.load("initAspect.npy"))
        # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
        init_aspect = torch.from_numpy(init_aspect)
        PreTrainABAE = clas_model.PreTrainABAE(init_aspect, embed).to(config.device)

        pre_trained_aspect = torch.load("AspectExtract/Aspect_Model.pkl")
        aspect_dict = PreTrainABAE.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
        aspect_dict.update(pre_trained_dict)
        PreTrainABAE.load_state_dict(aspect_dict)
        # PreTrainABAE = PreTrainABAE.eval()

        trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

        run = clas_model.CrfWdeRnnEncoder(300, 300, 50, embed, trained_aspect).to(config.device)
        # params = []
        # for param in run.parameters():
        #     if param.requires_grad:
        #         params.append(param)
        # 加载预训练权重
        if pretrain is True:
            pre_trained_dict = torch.load(config.pretrained_model)
            # pre_trained_dict = torch.load(config.pretrained_model, map_location=lambda storage, loc: storage)
            model_dict = run.state_dict()
            pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
            model_dict.update(pre_trained_dict)
            run.load_state_dict(model_dict)

        criterion = config.criterion()
        optimizer_rnn = config.optimizer(filter(lambda p: p.requires_grad, run.parameters()), lr=config.clas_lr)
        optimizer_abae = config.optimizer(filter(lambda p: p.requires_grad, PreTrainABAE.parameters()),
                                          lr=config.clas_lr)
        all_evaluate = []
        best_test = 0
        for epoch in range(config.epoch + 1):
            run_hidden = run.initHidden(config.batch_size)
            # context = torch.ones((config.batch_size, 50))
            # loss_last = torch.tensor([0], dtype=torch.float)
            optimizer_rnn.zero_grad()
            optimizer_abae.zero_grad()
            run.zero_grad()
            for idx, sample_batch in enumerate(train_data):
                run = run.train()
                PreTrainABAE = PreTrainABAE.train()
                input_data = sample_batch['input'].to(config.device)
                label = sample_batch['label'].to(config.device)
                aspect_info, trained_aspect, reg = PreTrainABAE(input_data)
                input_data[:, 1] = aspect_info
                out = run(input_data, run_hidden, trained_aspect, "train").view(config.batch_size, 2).to(config.device)
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
            print('epoch {} of {}: loss : {}'.format(epoch, config.epoch, loss))

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
                        input_data = sample_batch['input'].to(config.device)
                        label = sample_batch['label'].to(config.device)
                        aspect_info, trained_aspect, _ = PreTrainABAE(input_data)
                        input_data[:, 1] = aspect_info
                        outputs = run(input_data, run_hidden, trained_aspect, "test").view(1, 2).to(config.device)
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
        Load the best models and Begin test
        '''
        PreTrainABAE_test = clas_model.PreTrainABAE(init_aspect, embed).to(config.device)

        pre_trained_aspect = torch.load("ClassifyModelSave/Final_model_aspect.pkl")
        aspect_dict = PreTrainABAE_test.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_aspect.items() if k in aspect_dict}
        aspect_dict.update(pre_trained_dict)
        PreTrainABAE_test.load_state_dict(aspect_dict)

        trained_aspect = pre_trained_aspect["aspect_lookup_mat"].data

        model_test = clas_model.CrfWdeRnnEncoder(300, 300, 50, embed, trained_aspect).to(config.device)

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
                input_data = sample_batch['input'].to(config.device)
                label = sample_batch['label'].to(config.device)
                aspect_info, trained_aspect, _ = PreTrainABAE_test(input_data)
                input_data[:, 1] = aspect_info
                outputs = model_test(input_data, run_hidden, trained_aspect, "test").view(1, 2).to(config.device)
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

    def aspect_train(self, train_data, embed, sentence):
        init_aspect = np.array(np.load("initAspect.npy"))
        # init_aspect = init_aspect / np.linalg.norm(init_aspect, axis=-1, keepdims=True)
        init_aspect = torch.from_numpy(init_aspect)

        run = asp_model.ABAE(init_aspect, embed).to(config.device)

        # loss_func = torch.nn.TripletMarginLoss(margin=1, p=2)
        loss_func = MaxMargin()
        # params = []
        # for param in run.parameters():
        #     if param.requires_grad:
        #         params.append(param)

        # optimizer = config.optimizer(filter(lambda p: p.requires_grad, run.parameters()), lr=0.0001)
        # bb = filter(lambda p: p.requires_grad, run.parameters())
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, run.parameters()), lr=0.001)
        par = run.state_dict()
        # print("init aspect matrix : ", par)
        # print("init aspect matrix : ", par["aspect_lookup_mat"])
        min_loss = float('inf')

        for epoch in range(config.epoch):
            loss_last = torch.tensor([0], dtype=torch.float)
            optimizer.zero_grad()
            # run.zero_grad()
            run = run.train()
            loss_all = 0
            for idx, sample_batch in enumerate(train_data):
                indices = np.random.choice(len(sentence), config.batch_size * config.neg_size)
                samples = sentence[indices].reshape(config.batch_size, config.neg_size, config.maxlen)
                samples = torch.from_numpy(samples).to(config.device)
                # now = time.time()
                input_ = sample_batch['input'].to(config.device)
                out, sentence_attention, nag_samples, reg = run(input_, samples)
                loss_last = loss_func(sentence_attention, nag_samples, out) + config.lambda_ * reg
                # loss_last = loss_last / config.batch_size
                loss_last.backward()
                optimizer.step()
                loss_all += loss_last
            if loss_all < min_loss:
                print("save models......")
                file_name = "AspectExtract/Aspect_Model.pkl"
                torch.save(run.state_dict(), file_name)
                min_loss = loss_all
            print('epoch {} of {}: loss : {}'.format(epoch, config.epoch, loss_last.item()))

    # TODO: aspect extraction based on Apriori
    def asp_extra_apriori(self, all_data):
        pass
