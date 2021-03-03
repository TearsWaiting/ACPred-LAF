# ---encoding:utf-8---
# @Time : 2020.11.08
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : fine_tune_ACP.py


from preprocess import data_load_kmer_finetune
from configuration import config
from util import util_metric, util_freeze, util_dim_reduction
from model import BERT_sense_scaled_01

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle


# 根据余弦相似度计算multi_sense embedding的损失
def cal_loss_dist_by_cosine(model):
    embedding = model.embedding
    loss_dist = 0

    vocab_size = embedding[0].tok_embed.weight.shape[0]
    d_model = embedding[0].tok_embed.weight.shape[1]

    Z_norm = vocab_size * (len(embedding) ** 2 - len(embedding)) / 2

    for i in range(len(embedding)):
        for j in range(len(embedding)):
            if i < j:
                cosin_similarity = torch.cosine_similarity(embedding[i].tok_embed.weight, embedding[j].tok_embed.weight)
                loss_dist -= torch.sum(cosin_similarity)
                # print('cosin_similarity.shape', cosin_similarity.shape)
    loss_dist = loss_dist / Z_norm
    return loss_dist


# 用训练集进行训练
def train_ACP(train_iter, valid_iter, test_iter, model, optimizer, criterion, config, iter_k):
    steps = 0  # 1个batch对应1个step
    best_acc = 0
    best_performance = 0

    # 每一次交叉验证都迭代config.epoch个epoch
    for epoch in range(1, config.epoch + 1):
        repres_list = []
        label_list = []

        # 遍历整个训练集
        for batch in train_iter:
            '''
            multi-scaled
            '''
            if if_multi_scaled:
                input, origin_input, label = batch
                logits, output = model(input, origin_input)
            else:
                input, label = batch
                logits, output = model(input)

                repres_list.extend(output.cpu().detach().numpy())
                label_list.extend(label.cpu().detach().numpy())

            # L1正则化
            # l1_lambda = torch.tensor(1e-5, device=torch.device("cuda:0" if config_data.cuda else "cpu"))
            # l1_reg = torch.tensor(0., device=torch.device("cuda:0" if config_data.cuda else "cpu"))
            # for param in model.parameters():
            #     l1_reg += torch.sum(torch.abs(param))
            # loss = F.cross_entropy(logits, target) + l1_lambda * l1_reg

            # print('logits', logits.shape)
            # print('label', label.shape)
            loss = criterion(logits.view(-1, config.num_class), label.view(-1))
            loss = (loss.float()).mean()

            # flooding method
            loss = (loss - b).abs() + b

            # 距离损失
            # alpha = -0.1
            # loss_dist = alpha * cal_loss_dist_by_cosine(model)
            # loss += loss_dist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

            # 每训练config.log_interval个batch/step就打印一次训练结果，并记录训练集当前的损失和准确度，用于绘图
            if steps % config.interval_log == 0:
                # torch.max(logits, 1)函数：返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                corrects = (torch.max(logits, 1)[1] == label).sum()  # .view(label.size()此处没有影响
                # corrects += (torch.max(logits, 1)[1].view(label.size()) == label).sum()

                # 因为不是每个batch的大小都等于cofig.batch_size
                the_batch_size = label.shape[0]
                train_acc = 100.0 * corrects / the_batch_size
                sys.stdout.write(
                    '\rEpoch[{}] Batch[{}] - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(epoch, steps,
                                                                                       loss,
                                                                                       train_acc,
                                                                                       corrects,
                                                                                       the_batch_size))
                print()

                # 绘图数据
                step_log_interval.append(steps)
                train_acc_record.append(train_acc)  # 训练集的准确度
                train_loss_record.append(loss)  # 训练集的平均损失

        sum_epoch = iter_k * config.epoch + epoch

        # 每训练arg.valid_interval个epoch就在交叉验证集上进行一次验证，并记录验证集当前的准确度，用于绘图
        if valid_iter and sum_epoch % config.interval_valid == 0:
            print('-' * 30 + 'Periodic Validation' + '-' * 30)
            valid_acc, valid_loss, valid_metric = model_eval(valid_iter, model, config)
            print('valid current performance')
            print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
            print(valid_metric)
            print('-' * 30 + 'Over' + '-' * 30)

            # 绘图数据
            step_valid_interval.append(sum_epoch)
            val_acc_record.append(valid_acc)  # 交叉验证集的准确度

        # 每config.test_interval个epoch迭代完都测试一下在测试集上的表现，并记录当前测试集的损失和准确度，用于绘图
        if test_iter and sum_epoch % config.interval_test == 0:
            print('#' * 60 + 'Periodic Test' + '#' * 60)

            test_acc, test_loss, test_metric, test_repres_list, test_label_list = \
                model_eval(test_iter, model, config, if_dim_reduc=True, if_ROC=False)
            # test_acc, test_loss, test_metric, roc_data, prc_data = \
            #     model_eval(test_iter, model, config, if_dim_reduc=False, if_ROC=True)

            print('test current performance')
            print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
            print(test_metric.numpy())
            print('#' * 60 + 'Over' + '#' * 60)

            # 绘图数据
            step_test_interval.append(sum_epoch)
            test_acc_record.append(test_acc)  # 测试集的准确度
            test_loss_record.append(test_loss)  # 测试集的损失

            # 满足一定的条件就保存当前的模型
            # if test_acc > best_acc:
            #     best_acc = test_acc
            #     best_performance = test_metric
            #     if config.save_best and best_acc > config.threshold:
            #         print('Save Model: {}, ACC: {:.4f}%\n'.format(config.learn_name, best_acc))
            #         # print('model.state_dict()', model.state_dict())
            #         save_model(model.state_dict(), best_acc, config.path_model_save, config.learn_name)

            if test_acc > best_acc:
                best_acc = test_acc
                best_performance = test_metric

            # 保存ROC_data
            if if_save_temp:
                # model_name = 'LAF'
                model_name = 'LAF-MSMC'
                # model_name = 'LAF-MSE'
                # model_name = 'LAF-MSC'

                # if roc_data[2] >= 0.86:
                #     with open(config.path_meta_data + 'roc_data_{}_{}.pkl'.format(model_name,roc_data[2]), 'wb') as file:
                #         pickle.dump(roc_data, file)
                #     print('roc_data save over')

                if prc_data[2] >= 0.88:
                    with open(config.path_meta_data + 'prc_data_{}_{}.pkl'.format(model_name, prc_data[2]),
                              'wb') as file:
                        pickle.dump(prc_data, file)
                    print('prc_data save over')

                if test_acc >= 83.00:
                    save_model(model.state_dict(), test_acc, config.path_model_save, model_name)
                    print('model save over')

            '''
            data t-SNE
            '''
            # if sum_epoch % 1 == 0 or epoch == 1:
            #     test_label_list = [x + 2 for x in test_label_list]
            #     repres_list.extend(test_repres_list)
            #     label_list.extend(test_label_list)
            #     print('len(repres_list)', len(repres_list))
            #     print('len(repres_list[0])', len(repres_list[0]))
            #     print('len(label_list)', len(label_list))
            #
            #     title = 'Samples Embedding t-SNE Visualisation, Epoch[{}]'.format(epoch)
            #     util_dim_reduction.t_sne(title, repres_list, label_list, None, 4)
            #     # title = 'Samples Embedding PCA Visualisation, Epoch[{}]'.format(epoch)
            #     # util_dim_reduction.pca(title, repres_list, label_list, None, 4)

            '''
            attention map
            '''
            # if sum_epoch % 10 == 0:
            #     attention_map = model.layers[0].attention_map
            #     print('attention_map.size()', attention_map.size())  # [batch_size, num_head, seq_len, seq_len]
            #     attention_map_example = attention_map[0, 0, :, :]
            #     print('attention_map_example\n', attention_map_example)
            #
            #     import seaborn as sns
            #     sns.set()
            #     fig = plt.figure()
            #     ax = sns.heatmap(attention_map_example[0:20, 0:20].cpu(), annot=True)
            #     plt.show()

            '''
            reduction feature visualization
            '''
            if sum_epoch % 10 == 0 or epoch == 1 or (epoch % 2 == 0 and epoch <= 10):
                test_label_list = [x + 2 for x in test_label_list]
                repres_list.extend(test_repres_list)
                label_list.extend(test_label_list)

                X = np.array(repres_list)  # [num_samples, n_components]
                data_index = label_list
                data_label = None
                class_num = 4
                title = 'Learned Feature Visualization, Epoch[{}]'.format(epoch)
                font = {"color": "darkred", "size": 13, "family": "serif"}
                plt.style.use("default")

                plt.figure()
                plt.scatter(X[:, 0], X[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', class_num))
                if data_label:
                    for i in range(len(X)):
                        plt.annotate(data_label[i], xy=(X[:, 0][i], X[:, 1][i]),
                                     xytext=(X[:, 0][i] + 1, X[:, 1][i] + 1))  # 这里xy是需要标记的坐标，xytext是对应的标签坐标
                plt.title(title, fontdict=font)

                if data_label is None:
                    cbar = plt.colorbar(ticks=range(class_num))
                    cbar.set_label(label='digit value', fontdict=font)
                    plt.clim(0 - 0.5, class_num - 0.5)
                plt.show()

    return best_performance, best_acc


# 对模型进行测试
# def model_eval(data_iter, model, config, if_metric=True):
#     iter_size, corrects, avg_loss = 0, 0, 0
#     metric = None
#     device = torch.device("cuda" if config.cuda else "cpu")
#     label_pred = torch.empty([0], device=device)
#     label_real = torch.empty([0], device=device)
#     pred_prob = torch.empty([0], device=device)
#
#     print('model_eval data_iter', len(data_iter))
#
#     criterion = nn.CrossEntropyLoss()
#
#     # 不写容易爆内存
#     with torch.no_grad():
#         for batch in data_iter:
#             '''
#             multi-scaled
#             '''
#             if if_multi_scaled:
#                 input, origin_inpt, label = batch
#                 logits, output = model(input, origin_inpt)
#             else:
#                 input, label = batch
#                 logits, output = model(input)
#
#             loss = criterion(logits.view(-1, config.num_class), label.view(-1))
#             loss = (loss.float()).mean()
#             avg_loss += loss
#
#             pred_prob_all = F.softmax(logits, dim=1)  # 预测概率 [batch_size, class_num]
#             pred_prob_positive = pred_prob_all[:, 1]  # 注意，极其容易出错
#             pred_prob_sort = torch.max(pred_prob_all, 1)  # 每个样本中预测的最大的概率 [batch_size]
#             pred_class = pred_prob_sort[1]  # 每个样本中预测的最大的概率所在的位置（类别） [batch_size]
#             # corrects += (torch.max(logits, 1)[1] == label).sum()
#             corrects += (pred_class == label).sum()
#
#             # print('pred_prob_all', pred_prob_all)
#             # print('pred_prob_positive', pred_prob_positive)
#             # print('pred_class', pred_class)
#
#             iter_size += label.shape[0]
#
#             if if_metric:
#                 label_pred = torch.cat([label_pred, pred_class.float()])
#                 # label_pred = torch.cat([label_pred, pred_class])
#                 label_real = torch.cat([label_real, label.float()])
#                 # label_real = torch.cat([label_real, label])
#                 pred_prob = torch.cat([pred_prob, pred_prob_positive])
#
#     if if_metric:
#         # print('label_pred', label_pred.shape)
#         # print('label_real', label_real.shape)
#         metric = util_metric.caculate_metric(label_pred, label_real, pred_prob)
#
#     avg_loss /= iter_size
#     accuracy = 100.0 * corrects / iter_size
#     print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
#                                                                   accuracy,
#                                                                   corrects,
#                                                                   iter_size))
#     return accuracy, avg_loss, metric

def model_eval(data_iter, model, config, if_metric=True, if_dim_reduc=False, if_ROC=False):
    iter_size, corrects, avg_loss = 0, 0, 0
    metric = None
    device = torch.device("cuda" if config.cuda else "cpu")
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)

    print('model_eval data_iter', len(data_iter))

    criterion = nn.CrossEntropyLoss()

    repres_list = []
    label_list = []

    # 不写容易爆内存
    with torch.no_grad():
        for batch in data_iter:
            '''
            multi-scaled
            '''
            if if_multi_scaled:
                input, origin_inpt, label = batch
                logits, output = model(input, origin_inpt)
            else:
                input, label = batch
                logits, output = model(input)

            # 用于t-SNE
            if if_dim_reduc:
                repres_list.extend(output.cpu().detach().numpy())
                label_list.extend(label.cpu().detach().numpy())

            loss = criterion(logits.view(-1, config.num_class), label.view(-1))
            loss = (loss.float()).mean()
            avg_loss += loss

            pred_prob_all = F.softmax(logits, dim=1)  # 预测概率 [batch_size, class_num]
            pred_prob_positive = pred_prob_all[:, 1]  # 注意，极其容易出错
            pred_prob_sort = torch.max(pred_prob_all, 1)  # 每个样本中预测的最大的概率 [batch_size]
            pred_class = pred_prob_sort[1]  # 每个样本中预测的最大的概率所在的位置（类别） [batch_size]
            # corrects += (torch.max(logits, 1)[1] == label).sum()
            corrects += (pred_class == label).sum()

            # print('pred_prob_all', pred_prob_all)
            # print('pred_prob_positive', pred_prob_positive)
            # print('pred_class', pred_class)

            iter_size += label.shape[0]

            if if_metric:
                label_pred = torch.cat([label_pred, pred_class.float()])
                # label_pred = torch.cat([label_pred, pred_class])
                label_real = torch.cat([label_real, label.float()])
                # label_real = torch.cat([label_real, label])
                pred_prob = torch.cat([pred_prob, pred_prob_positive])

    if if_metric:
        # print('label_pred', label_pred.shape)
        # print('label_real', label_real.shape)
        if if_ROC:
            metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_real, pred_prob, if_ROC)
        else:
            metric = util_metric.caculate_metric(label_pred, label_real, pred_prob, if_ROC)

    avg_loss /= iter_size
    accuracy = 100.0 * corrects / iter_size
    print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
                                                                  accuracy,
                                                                  corrects,
                                                                  iter_size))
    if if_dim_reduc:
        if if_ROC:
            return accuracy, avg_loss, metric, repres_list, label_list, roc_data
        else:
            return accuracy, avg_loss, metric, repres_list, label_list
    else:
        if if_ROC:
            return accuracy, avg_loss, metric, roc_data, prc_data
        else:
            return accuracy, avg_loss, metric


# 保存模型
def save_model(model_dict, best_acc, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = 'ACC->{:.4f}, {}.pt'.format(best_acc, save_prefix)
    save_path_pt = os.path.join(save_dir, filename)
    # print('model_dict', model_dict)
    torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)


# 绘图
def show_res(config_finetune):
    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.plot(step_log_interval, train_acc_record)
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Val Acc Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(step_valid_interval, val_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Test Acc Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(step_test_interval, test_acc_record)

    plt.savefig('../figure/' + config_finetune.learn_name + '.png')
    plt.show()


def k_fold_CV(train_iter_orgin, test_iter, config_pretrain, config_finetune, k):
    # global valid_performance
    valid_performance = torch.zeros([1, 7])  # 7是指标数量

    for iter_k in range(k):

        print('=' * 50, 'iter_k=', iter_k + 1, '=' * 50)

        # 在训练集上进行交叉验证，划分训练集和交叉验证的验证集
        train_iter = [x for i, x in enumerate(train_iter_orgin) if i % k != iter_k]  # 划分后的训练集
        valid_iter = [x for i, x in enumerate(train_iter_orgin) if i % k == iter_k]  # 划分后的验证集
        print('----------data selection----------')
        print('train_iter index', [i for i, x in enumerate(train_iter_orgin) if i % k != iter_k])
        print('valid_iter index', [i for i, x in enumerate(train_iter_orgin) if i % k == iter_k])

        print('----------sample----------')
        print('sample valid_iter[0]', valid_iter[0])

        print('len(train_iter_orgin)', len(train_iter_orgin))  # 原训练集的batch数
        print('len(train_iter)', len(train_iter))  # 划分后，训练集的batch数
        print('len(valid_iter)', len(valid_iter))  # 划分后，交叉验证的验证集的batch数
        if test_iter:
            print('len(test_iter)', len(test_iter))  # 测试集的batch数

        global step_log_interval, step_valid_interval, step_test_interval, \
            train_acc_record, train_loss_record, val_acc_record, test_acc_record, \
            test_loss_record
        # 数据记录容器,用于绘图
        step_log_interval = []
        step_valid_interval = []
        step_test_interval = []

        train_acc_record = []
        train_loss_record = []
        val_acc_record = []
        test_acc_record = []
        test_loss_record = []

        # 创建微调模型
        model = BERT_sense_02.BERT(config_pretrain)

        # 加载预训练模型
        model = load_model(model, config_finetune.path_pretrained_model)

        if config_finetune.cuda: model.cuda()

        # 冻结嵌入层和Encoder Block
        # util_freeze.freeze_by_names(model, ['embedding', 'layers'])
        util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge', 'soft_attention', 'layers'])

        # 解冻部分层
        for name, child in model.named_children():
            for sub_name, sub_child in child.named_children():
                if name == 'layers' and (sub_name == '2'):
                    print('Encoder 2 Is Unfreezing')
                    for param in sub_child.parameters():
                        param.requires_grad = True

        print('Model Freezing')

        # 打印模型结构参数
        print('-' * 50, 'Model.named_parameters', '-' * 50)
        for name, value in model.named_parameters():
            print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))

        # 选择优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=config_finetune.lr,
                                     weight_decay=config_finetune.reg)  # L2正则化
        # optimizer = torch.optim.AdamW(params=model.parameters(), lr=config_finetune.lr)
        # optimizer = torch.optim.Adagrad(params=model.parameters(), lr=config_finetune.lr)

        criterion = nn.CrossEntropyLoss()
        model.train()

        # 训练
        train_ACP(train_iter, valid_iter, test_iter, model, optimizer, criterion, config_finetune, iter_k)

        # 交叉验证,每一折交叉验证训练完毕后都测试一下在交叉验证集上的表现
        print('=' * 40 + 'Cross Validation' + '=' * 40)
        ACC, loss, valid_metric = model_eval(valid_iter, model, config_finetune)
        valid_metric = valid_metric.view(1, -1)
        if torch.sum(valid_performance) == 0:
            valid_performance = valid_metric
        else:
            valid_performance = torch.cat((valid_performance, valid_metric), dim=0)

        print('valid current performance')
        print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(valid_metric.numpy()))

        print('valid mean performance')
        print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(torch.mean(valid_performance, dim=0, keepdim=True).numpy()))
        print('=' * 40 + 'Cross Validation Over' + '=' * 40)

        # 绘图
        show_res(config_finetune)

    return valid_performance


def train_test(train_iter_orgin, test_iter, config_pretrain, config_finetune):
    print('=' * 50, 'train-test', '=' * 50)
    iter_k = 0
    print('len(train_iter)', len(train_iter_orgin))  # 划分后，训练集的batch数
    print('len(test_iter)', len(test_iter))  # 测试集的batch数

    # 数据记录容器,用于绘图
    global step_log_interval, step_valid_interval, step_test_interval, \
        train_acc_record, train_loss_record, val_acc_record, test_acc_record, test_loss_record

    step_log_interval = []
    step_valid_interval = []
    step_test_interval = []

    train_acc_record = []
    train_loss_record = []
    val_acc_record = []
    test_acc_record = []
    test_loss_record = []

    # 创建微调模型
    # model = BERT_plus_02.BERT(config_pretrain)
    # model = BERT_sense_02.BERT(config_pretrain)
    # model = BERT_scaled_01.BERT(config_pretrain)
    # model = BERT_scaled_02.BERT(config_pretrain)
    model = BERT_sense_scaled_01.BERT(config_pretrain)
    # print('model', model)

    # 加载预训练模型
    # model = load_model(model, config_finetune.path_pretrained_model)

    if config_finetune.cuda: model.cuda()

    # 冻结嵌入层和Encoder Block
    # util_freeze.freeze_by_names(model, ['embedding', 'layers'])
    # util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge', 'layers'])
    # util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge', 'soft_attention', 'layers'])
    # util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge'])
    # util_freeze.freeze_by_names(model, ['embedding_merge'])
    # util_freeze.freeze_by_names(model, ['embedding'])

    # 解冻部分层
    # for name, child in model.named_children():
    #     for sub_name, sub_child in child.named_children():
    #         if name == 'layers' and (sub_name == '3'):
    #             print('Encoder Is Unfreezing')
    #             for param in sub_child.parameters():
    #                 param.requires_grad = True

    print('-' * 50, 'Model.named_parameters', '-' * 50)
    for name, value in model.named_parameters():
        print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))

    print('-' * 10, 'Model Freezing', '-' * 10)

    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        # print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        # print("该层参数和：" + str(l))
        k = k + l
    print('=' * 50, "总参数数量:" + str(k), '=' * 50)

    # 选择优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=config_finetune.lr,
    #                              weight_decay=config_finetune.reg)  # L2正则化
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config_finetune.lr, weight_decay=config_finetune.reg)
    # optimizer = torch.optim.Adagrad(params=model.parameters(), lr=config_finetune.lr)
    # optimizer = torch.optim.Adamax(params=model.parameters(), lr=config_finetune.lr)

    criterion = nn.CrossEntropyLoss()
    model.train()

    '''
    embedding t-SNE
    '''
    # embedding = model.embedding.tok_embed.weight
    # embedding = embedding.cpu().detach().numpy()
    # embed_index = list(config_finetune.token2index.values())
    # embed_label = list(config_finetune.token2index.keys())
    # print('embedding', embedding)
    # print('embedding.shape', embedding.shape)
    # print('embed_index', embed_index)
    # print('embed_label', embed_label)
    # print('len(token2index)', len(config_finetune.token2index))
    # title = 'embedding t-SNE origin'
    # util_dim_reduction.t_sne(title, embedding, embed_index, embed_label, len(embed_index))
    # title = 'embedding PCA origin'
    # util_dim_reduction.pca(title, embedding, embed_index, embed_label, len(embed_index))

    '''
    multi-sense embedding t-SNE
    '''
    # embed_num = config_train.num_embedding
    # print('embed_num', embed_num)
    #
    # embedding_list = []
    # embed_type_list = []
    # for j in range(embed_num):
    #     embedding = model.embedding[j].tok_embed.weight
    #     embedding = embedding.cpu().detach().numpy()
    #     embed_type = list(config_train.token2index.values())
    #     embedding_list.extend(embedding)
    #     embed_type_list.extend(embed_type)
    #
    # print('len(embedding_list)', len(embedding_list))
    # print('len(embedding_list[0])', len(embedding_list[0]))
    # print('embed_type_list', embed_type_list)
    # print('vocab_size', config_train.vocab_size)
    # util_dim_reduction.t_sne(embedding_list, embed_type_list, config_train.vocab_size)

    # 训练
    print('=' * 50 + 'Start Training' + '=' * 50)
    best_performance, best_acc = train_ACP(train_iter_orgin, None, test_iter, model, optimizer, criterion,
                                           config_finetune, iter_k)

    # 绘图
    show_res(config_finetune)

    # 测试集测试
    print('*' * 60 + 'The Last Test' + '*' * 60)
    test_acc, test_loss, test_metric = model_eval(test_iter, model, config_finetune)
    print('test current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(test_metric.numpy())

    global test_performance
    test_performance = test_metric

    # 满足一定的条件就保存当前的模型
    if test_acc > best_acc:
        best_acc = test_acc
        best_performance = test_metric
        if config_finetune.save_best and best_acc >= config_finetune.threshold:
            print('Save Model: {}, ACC: {:.4f}%\n'.format(config_finetune.learn_name, best_acc))

            # print('model.state_dict()', model.state_dict())
            save_model(model.state_dict(), best_acc, config_finetune.path_model_save, config_finetune.learn_name)

    print('*' * 60 + 'The Last Test Over' + '*' * 60)

    '''
    embedding t-SNE
    '''
    # embedding = model.embedding.tok_embed.weight
    # embedding = embedding.cpu().detach().numpy()
    # embed_index = list(config_finetune.token2index.values())
    # embed_label = list(config_finetune.token2index.keys())
    # print('embedding', embedding)
    # print('embedding.shape', embedding.shape)
    # print('embed_index', embed_index)
    # print('embed_label', embed_label)
    # print('len(token2index)', len(config_finetune.token2index))
    # title = 'embedding t-SNE last'
    # util_dim_reduction.t_sne(title, embedding, embed_index, embed_label, len(embed_index))
    # title = 'embedding PCA last'
    # util_dim_reduction.pca(title, embedding, embed_index, embed_label, len(embed_index))

    '''
    multi-sense embedding t-SNE
    '''
    # embed_num = config_train.num_embedding
    # print('embed_num', embed_num)
    #
    # embedding_list = []
    # embed_type_list = []
    # for j in range(embed_num):
    #     embedding = model.embedding[j].tok_embed.weight
    #     embedding = embedding.cpu().detach().numpy()
    #     embed_type = list(config_train.token2index.values())
    #     embedding_list.extend(embedding)
    #     embed_type_list.extend(embed_type)
    #
    # print('len(embedding_list)', len(embedding_list))
    # print('len(embedding_list[0])', len(embedding_list[0]))
    # print('embed_type_list', embed_type_list)
    # print('vocab_size', config_train.vocab_size)
    # util_dim_reduction.t_sne(embedding_list, embed_type_list, config_train.vocab_size)

    return best_performance


def select_dataset(config_finetune):
    # Anti-peptide
    # config_finetune.path_train_data = '../data/task_data/anti-peptide/tsv/train data/Anti-angiogenic Peptides.tsv'
    # config_finetune.path_test_data = '../data/task_data/anti-peptide/tsv/test data 2/Anti-angiogenic Peptides.tsv'

    # config_finetune.path_train_data = '../data/task_data/anti-peptide/tsv/train data/Anti-bacterial Peptides.tsv'
    # config_finetune.path_test_data = '../data/task_data/anti-peptide/tsv/test data 2/Anti-bacterial Peptides.tsv'

    # config_finetune.path_train_data = '../data/task_data/anti-peptide/tsv/train data/Anti-cancer Peptides.tsv'
    # config_finetune.path_test_data = '../data/task_data/anti-peptide/tsv/test data 2/Anti-cancer Peptides.tsv'

    # config_finetune.path_train_data = '../data/task_data/anti-peptide/tsv/train data/Anti-inflammatory Peptides.tsv'
    # config_finetune.path_test_data = '../data/task_data/anti-peptide/tsv/test data 2/Anti-inflammatory Peptides.tsv'
    # #
    # config_finetune.path_train_data = '../data/task_data/anti-peptide/tsv/train data/Anti-viral peptides.tsv'
    # config_finetune.path_test_data = '../data/task_data/anti-peptide/tsv/test data 2/Anti-viral peptides.tsv'

    # config_finetune.path_train_data = '../data/task_data/anti-peptide/tsv/train data/Cell-penetrating peptides.tsv'
    # config_finetune.path_test_data = '../data/task_data/anti-peptide/tsv/test data 2/Cell-penetrating peptides.tsv'

    # config_finetune.path_train_data = '../data/task_data/anti-peptide/tsv/train data/Quorum sensing peptides.tsv'
    # config_finetune.path_test_data = '../data/task_data/anti-peptide/tsv/test data 2/Quorum sensing peptides.tsv'

    # config_finetune.path_train_data = '../data/task_data/anti-peptide/tsv/train data/Surface-binding peptides.tsv'
    # config_finetune.path_test_data = '../data/task_data/anti-peptide/tsv/test data 2/Surface-binding peptides.tsv'

    # peptide detectality
    # config_finetune.path_train_data = '../data/task_data/detection/tsv/detect_train.tsv'
    # config_finetune.path_test_data = '../data/task_data/detection/tsv/detect_test.tsv'

    # ACP
    config_finetune.path_train_data = '../data/ACP_dataset/tsv/ACP_mixed_train.tsv'
    config_finetune.path_test_data = '../data/ACP_dataset/tsv/ACP_mixed_test.tsv'
    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/ACP_new_2_train.tsv'
    # config_finetune.path_test_data = '../data/task_data/ACP/tsv/ACP_new_2_test.tsv'

    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/LEE_Dataset.tsv'
    # config_finetune.path_test_data = '../data/task_data/ACP/tsv/Independent dataset.tsv'
    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/ACP2_main_train.tsv'
    # config_finetune.path_test_data = '../data/task_data/ACP/tsv/ACP2_main_test.tsv'
    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/ACP2_alternate_train.tsv'
    # config_finetune.path_test_data = '../data/task_data/ACP/tsv/ACP2_alternate_test.tsv'
    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/ACP_DL_740.tsv'
    # config_finetune.path_test_data = None
    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/ACP_DL_240.tsv'
    # config_finetune.path_test_data = None
    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/ACPred-Fuse_ACP_Train500.tsv'
    # config_finetune.path_test_data = '../data/task_data/ACP/tsv/ACPred-Fuse_ACP_Test2710.tsv'
    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/ACP_FL_train_500.tsv'
    # config_finetune.path_test_data = '../data/task_data/ACP/tsv/ACP_FL_test_164.tsv'

    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/Tyagi_train.tsv'
    # config_finetune.path_test_data = '../data/task_data/ACP/tsv/Tyagi_test.tsv'
    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/Hajisharifi_train.tsv'
    # config_finetune.path_test_data = '../data/task_data/ACP/tsv/Hajisharifi_test.tsv'

    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/Tyagi_dataset.tsv'
    # config_finetune.path_test_data = None
    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/Hajisharifi_dataset.tsv'
    # config_finetune.path_test_data = None

    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/Hajisharifi_dataset.tsv'
    # config_finetune.path_test_data = '../data/task_data/ACP/tsv/Tyagi_dataset.tsv'

    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/HC_dataset.tsv'
    # config_finetune.path_test_data = None

    # config_finetune.path_train_data = '../data/task_data/ACP/tsv/HC_train.tsv'
    # config_finetune.path_test_data = '../data/task_data/ACP/tsv/HC_test.tsv'

    # DNA-MS
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/5hmC/5hmC_M.musculus/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/5hmC/5hmC_M.musculus/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/4mC/4mC_F.vesca/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/4mC/4mC_F.vesca/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/4mC/4mC_Tolypocladium/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/4mC/4mC_Tolypocladium/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_A.thaliana/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_A.thaliana/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_C.elegans/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_C.elegans/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_D.melanogaster/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_D.melanogaster/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_F.vesca/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_F.vesca/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_H.sapiens/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_H.sapiens/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_R.chinensis/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_R.chinensis/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_T.thermophile/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_T.thermophile/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_Tolypocladium/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_Tolypocladium/test.tsv'
    # config_finetune.path_train_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/train.tsv'
    # config_finetune.path_test_data = '../data/task_data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/test.tsv'

    # sORF 序列太长，处理不了


def train_start():
    np.set_printoptions(linewidth=400, precision=4)

    time_start = time.time()

    # 加载配置
    config_finetune = config.get_finetune_config()
    # config_pretrain = pickle.load(open(config_finetune.path_pretrained_config, 'rb'))
    # config_pretrain = pickle.load(open('../data/config_data/BERT_pretrain_00 BERT_plus_02 small layer_2.pkl', 'rb'))
    # config_pretrain = pickle.load(open('../data/config_data/BERT_pretrain_00 BERT_plus_02 middle layer_4.pkl', 'rb'))

    # 不采取预训练的话
    config_pretrain = config.get_pretrain_config()
    torch.cuda.set_device(1)
    # 选择数据集
    select_dataset(config_finetune)

    # 一些必要的路径
    # config_finetune.path_pretrained_model = '../model_save/pre_train/BERT_pretrain_00 BERT_plus_02 small_last_2.pt'
    # config_finetune.path_pretrained_model = '../model_save/pre_train/BERT_pretrain_00 BERT_plus_02 middle layer_4_last_3.pt'
    # print('path_pretrained_model', config_finetune.path_pretrained_model)

    global if_multi_scaled, if_save_temp
    if_multi_scaled = True
    # if_multi_scaled = False
    # if_save_temp = True
    if_save_temp = False

    if if_multi_scaled:
        residue2idx = pickle.load(open('../data/kmer_residue2idx.pkl', 'rb'))
    else:
        residue2idx = pickle.load(open(config_finetune.path_meta_data + 'residue2idx.pkl', 'rb'))

    config_pretrain.vocab_size = len(residue2idx)
    config_finetune.token2index = residue2idx
    print('old config_pretrain.vocab_size', config_pretrain.vocab_size)

    # 选择要使用的GPU
    # torch.cuda.set_device(0)
    # torch.cuda.set_device(config_pretrain.device)

    # 加载训练集和测试集
    '''
    multi-scaled
    '''
    if if_multi_scaled:
        config_finetune.k_mer = config_pretrain.k_mer
        train_iter_orgin, test_iter = data_load_kmer_finetune.load_unified_train_and_test_data(config_finetune)
        # 更新token2index
        # residue2idx = pickle.load(open(config_finetune.path_meta_data + 'kmer_residue2idx.pkl', 'rb'))
        # config_pretrain.vocab_size = len(residue2idx)
        # config_finetune.token2index = residue2idx
        # print('new config_pretrain.vocab_size', config_pretrain.vocab_size)
    else:
        train_iter_orgin, test_iter = data_load_finetune.load_unified_train_and_test_data(config_finetune)

    print('config_pretrain.max_len', config_pretrain.max_len)
    print('config_finetune.max_len', config_finetune.max_len)

    '''feature train'''
    # import feature_selection as fs
    # train_base_path = '../data/hand-crafted feature/ACP_mixed_train/'
    # test_base_path = '../data/hand-crafted feature/ACP_mixed_test/'
    # # filename = 'AAC.xls' # 20
    # # filename = 'APAAC.xls' # 24
    # # filename = 'ATC.xls' # 7
    # # filename = 'PKL.xls'  # 254
    # filename = 'Pse-AAC.xls' # 23
    # train_iter_orgin = fs.select_feature(train_base_path, filename, config_finetune)
    # test_iter = fs.select_feature(test_base_path, filename, config_finetune)

    k = config_finetune.k_fold  # k折交叉验证
    valid_performance = 0
    best_performance = 0

    if k == -1:
        # train and test
        best_performance = train_test(train_iter_orgin, test_iter, config_pretrain, config_finetune)
    else:
        # k折交叉验证
        valid_performance = k_fold_CV(train_iter_orgin, test_iter, config_pretrain, config_finetune, k)

    # 打印训练结果
    print('*=' * 50 + 'Result Report' + '*=' * 50)

    if k != -1:
        print('valid mean performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(torch.mean(valid_performance, dim=0, keepdim=False).numpy()))

        print('valid_performance list')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(valid_performance.numpy()))
    else:
        print('last test performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(test_performance))
        print()
        print('best_performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(best_performance))

    print('*=' * 50 + 'Report Over' + '*=' * 50)

    with open(config_finetune.path_config_data + config_finetune.learn_name + '.pkl', 'wb') as file:
        pickle.dump(config_finetune, file)
    print('-' * 50, 'Config Save Over', '-' * 50)

    time_end = time.time()
    print('total time cost', time_end - time_start, 'seconds')
    return best_performance


if __name__ == '__main__':
    global b
    b = 0.06
    threshold = 0.06
    # b = 0.05
    # threshold = 0.10
    step = 0.01
    repeat_num = 1

    X = []
    Y = []
    while b <= threshold:
        X.append(b)
        y = []
        for i in range(repeat_num):
            print('#' * 50, 'b[{}/{}], i[{}/{}]'.format(b, threshold, i, repeat_num), '#' * 50)

            best_performance = train_start()
            acc = best_performance[0]
            y.append(acc)

        b += step
        Y.append(np.mean(y))

    # Y = torch.tensor(Y).view(1, -1)
    # Y = Y.float()
    # Y = torch.mean(Y, dim=0)

    print('X:\n', X)
    print('Y:\n', Y)
    # plt.figure()
    # plt.plot(X, Y, 'bo', X, Y, 'r--')
    # plt.show()
