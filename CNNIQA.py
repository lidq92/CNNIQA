"""
Pytorch implementation of the following paper:
Kang L, Ye P, Li Y, et al. Convolutional neural networks for no-reference image quality assessment[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1733-1740.
"""
# 
# Implemented by Dingquan Li
# Email: dingquanli@pku.edu.cn
# Date: 2018/4/19
#
# source activate ~/anaconda3/envs/tensorflow/
# tensorboard --logdir='./logs' --port=6006
# CUDA_VISIBLE_DEVICES=1 python CNNIQA.py 0
#-*-coding:utf-8-*-

import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import numpy as np
import os, sys
import yaml
from scipy import stats
from IQADataset import IQADataset
from IQAmodel import CNNIQAnet

def measure(sq, q, sq_std):

    sq = np.reshape(np.asarray(sq), (-1,))
    sq_std = np.reshape(np.asarray(sq_std), (-1,))
    q = np.reshape(np.asarray(q), (-1,))

    srocc = stats.spearmanr(sq, q)[0]
    krocc = stats.stats.kendalltau(sq, q)[0]
    plcc = stats.pearsonr(sq, q)[0]
    rmse = np.sqrt(((sq - q) ** 2).mean())
    outlier_ratio = (np.abs(sq - q) > 2*sq_std).mean()
    return (srocc, krocc, plcc, rmse, outlier_ratio)

if __name__ == '__main__':
    EXP_ID = sys.argv[1] if len(sys.argv) > 1 else str(0) #
    conf_file = sys.argv[2] if len(sys.argv) > 2 else 'config.yaml' #
    with open(conf_file) as f:
        conf = yaml.load(f)
    if len(sys.argv) > 3:
        conf['database'] = sys.argv[3]  # database
    if len(sys.argv) > 4:
        conf['model'] = sys.argv[4]  # model
    database = conf['database']
    model = conf['model']
    print('EXP_ID: ' + EXP_ID)
    print('database: ' + database)  #
    print('model: ' + model)  #
    conf.update(conf[database])
    conf.update(conf[model])

    test_ratio = conf['test_ratio']

    trainloader = IQADataset(conf, EXP_ID, 'train')
    trainloader = torch.utils.data.DataLoader(trainloader,
                                              batch_size=conf['batch_size'],
                                              shuffle=True,
                                              num_workers=4)
    valloader = IQADataset(conf, EXP_ID, 'val')
    valloader = torch.utils.data.DataLoader(valloader)
    if conf['test_ratio']:
        testloader = IQADataset(conf, EXP_ID, 'test')
        test_index = testloader.index #
        testloader = torch.utils.data.DataLoader(testloader)

    net = CNNIQAnet(ker_size=conf['kernel_size'], 
                    n_kers=conf['n_kernels'], 
                    n1_nodes=conf['n1_nodes'], 
                    n2_nodes=conf['n2_nodes'])
    print(net)
    if conf['use_cuda']:
        net.cuda()


    if conf['enableTensorboard']:  # Tensorboard Visualization
        if not os.path.exists('logs'):
            os.makedirs('logs')
        from logger import Logger  #
        logger1 = Logger('./logs/EXP' + EXP_ID + '-' + 
                         database + '-' + model + '-train')
        logger2 = Logger('./logs/EXP' + EXP_ID + '-' + 
                         database + '-' + model + '-val')
        if test_ratio > 0 and conf['test_during_training']:
            logger3 = Logger('./logs/EXP' + EXP_ID + '-' + 
                             database + '-' + model + '-test')
    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models/' + model + '-' + database + \
                         '-EXP' + str(EXP_ID)
    if not os.path.exists('results'):
        os.makedirs('results')
    save_result_file = 'results/' + model + '-' + database + \
                       '-EXP' + str(EXP_ID)


    # Train the network
    criterion = nn.L1Loss()  #  L1 loss
    optimizer = torch.optim.Adam(net.parameters(), 
                                 lr=conf['learning_rate'],
                                 weight_decay=conf['weight_decay'])
    step = 1
    best_val_criterion = -1 # 
    for epoch in range(conf['n_epochs']):  # loop over the dataset multiple times

        net.train()
        L = 0
        for i, data in enumerate(trainloader, 0):
            patch, label, label_std = data
            if conf['use_cuda']:
                patch = patch.cuda()
                label = label.cuda()
            patch = Variable(patch.float())
            label = Variable(label.float())
            optimizer.zero_grad()
            outputs = net(patch, train=True)
            loss = criterion(outputs, label)
            L += loss.data[0]
            loss.backward()
            optimizer.step()
            step += 1
        train_loss = L / (i + 1)
        # print(train_loss)

        net.eval()
        # Val
        L = 0
        q = []
        sq = []
        sq_std = []
        for i, data in enumerate(valloader, 0):
            patch, label, label_std = data
            sq.append(label)
            sq_std.append(label_std)
            patch = torch.cat(patch) #
            if conf['use_cuda']:
                patch = patch.cuda()
                label = label.cuda()
            patch = Variable(patch.float())
            label = Variable(label.float())
            outputs = net(patch, train=False)
            q.append(outputs.data.mean())
            qq = torch.mean(outputs.data,0)
            loss = criterion(Variable(qq), label)
            L += loss.data[0]
        val_loss = L / (i+1)
        measure_values = measure(sq, q, sq_std)
        val_SROCC, val_KROCC, val_PLCC, val_RMSE, val_OR = measure_values

        # Test
        if test_ratio > 0 and conf['test_during_training']:
            L = 0
            q = []
            sq = []
            sq_std = []
            for i, data in enumerate(testloader, 0):
                patch, label, label_std = data
                sq.append(label)
                sq_std.append(label_std)
                patch = torch.cat(patch) #
                if conf['use_cuda']:
                    patch = patch.cuda()
                    label = label.cuda()
                patch = Variable(patch.float())
                label = Variable(label.float())
                outputs = net(patch, train=False)
                q.append(outputs.data.mean())
                qq = torch.mean(outputs.data,0)
                loss = criterion(Variable(qq), label)
                L += loss.data[0]
            test_loss = L / (i + 1)
            measure_values = measure(sq, q, sq_std)
            SROCC, KROCC, PLCC, RMSE, OR = measure_values

        if conf['enableTensorboard']:  # record training curves
            logger1.scalar_summary("loss: ", train_loss, epoch)  #
            logger2.scalar_summary("loss: ", val_loss, epoch)  #
            logger2.scalar_summary("SROCC: ", val_SROCC, epoch)  #
            logger2.scalar_summary("KROCC: ", val_KROCC, epoch)  #
            logger2.scalar_summary("PLCC: ", val_PLCC, epoch)  #
            logger2.scalar_summary("RMSE: ", val_RMSE, epoch)  #
            logger2.scalar_summary("OR: ", val_OR, epoch)  #
            if test_ratio > 0 and conf['test_during_training']:
                logger3.scalar_summary("loss: ", test_loss, epoch)  #
                logger3.scalar_summary("SROCC: ", SROCC, epoch)  #
                logger3.scalar_summary("KROCC: ", KROCC, epoch)  #
                logger3.scalar_summary("PLCC: ", PLCC, epoch)  #
                logger3.scalar_summary("RMSE: ", RMSE, epoch)  #
                logger3.scalar_summary("OR: ", OR, epoch)  #
        

        # Update the model with the best val
        # when epoch is larger than n_epochs/6
        # This is to avoid the situation that the model will not be updated 
        # due to the impact of randomly initializations of the networks
        if val_PLCC > best_val_criterion and epoch > conf['n_epochs']/6: #

            print("EXP ID={}: ".format(EXP_ID) + 
                  "Update best model using best_val_criterion " + 
                  "in epoch {}".format(epoch))
            print("Val results: " + 
                   "val loss={:.4f}, ".format(val_loss) + 
                   "SROCC={:.4f}, ".format(val_SROCC) + 
                   "KROCC={:.4f}, ".format(val_KROCC) + 
                   "PLCC={:.4f}, ".format(val_PLCC) + 
                   "RMSE={:.4f}, ".format(val_RMSE) +
                   "OR={:.2f}%, ".format(val_OR * 100))
            f = open(save_result_file+'.txt', 'w')
            f.write("EXP ID={}: ".format(EXP_ID) + 
                  "Update best model using best_val_criterion " + 
                  "in epoch {}".format(epoch) + "\n")
            f.write("Val results: " + 
                   "val loss={:.4f}, ".format(val_loss) + 
                   "SROCC={:.4f}, ".format(val_SROCC) + 
                   "KROCC={:.4f}, ".format(val_KROCC) + 
                   "PLCC={:.4f}, ".format(val_PLCC) + 
                   "RMSE={:.4f}, ".format(val_RMSE) +
                   "OR={:.2f}%, ".format(val_OR * 100) + '\n')
            if test_ratio > 0 and conf['test_during_training']:
                print("Test results: " + 
                       "test loss={:.4f}, ".format(test_loss) + 
                       "SROCC={:.4f}, ".format(SROCC) + 
                       "KROCC={:.4f}, ".format(KROCC) + 
                       "PLCC={:.4f}, ".format(PLCC) + 
                       "RMSE={:.4f}, ".format(RMSE) +
                       "OR={:.2f}%, ".format(OR * 100))
                f.write("Test results: " + 
                       "test loss={:.4f}, ".format(test_loss) + 
                       "SROCC={:.4f}, ".format(SROCC) + 
                       "KROCC={:.4f}, ".format(KROCC) + 
                       "PLCC={:.4f}, ".format(PLCC) + 
                       "RMSE={:.4f}, ".format(RMSE) +
                       "OR={:.2f}%, ".format(OR * 100) + '\n')
                np.save(save_result_file, 
                        (sq, sq_std, q, test_loss, 
                        SROCC, KROCC, PLCC, RMSE, OR, test_index))
            f.close()

            torch.save(net.state_dict(), trained_model_file)
            torch.save(net, trained_model_file + '.full')
            best_val_criterion = val_PLCC  # update best val

    # Test
    if test_ratio > 0:
        net.load_state_dict(torch.load(trained_model_file))  #  
        for parameter in net.parameters():
            parameter.requires_grad = False #
        net.eval()

        L = 0
        q = []
        sq = []
        sq_std = []
        for i, data in enumerate(testloader, 0):
            patch, label, label_std = data
            sq.append(label)
            sq_std.append(label_std)
            patch = torch.cat(patch) #
            if conf['use_cuda']:
                patch = patch.cuda()
                label = label.cuda()
            patch = Variable(patch.float())
            label = Variable(label.float())
            outputs = net(patch, train=False)
            q.append(outputs.data.mean())
            qq = torch.mean(outputs.data,0)
            loss = criterion(Variable(qq), label)
            L += loss.data[0]
        test_loss = L[0] / (i + 1)
        measure_values = measure(sq, q, sq_std)
        SROCC, KROCC, PLCC, RMSE, OR = measure_values

        print("Test results: " + 
               "test loss={:.4f}, ".format(test_loss) + 
               "SROCC={:.4f}, ".format(SROCC) + 
               "KROCC={:.4f}, ".format(KROCC) + 
               "PLCC={:.4f}, ".format(PLCC) + 
               "RMSE={:.4f}, ".format(RMSE) +
               "OR={:.2f}%, ".format(OR * 100))
        np.save(save_result_file, 
                (sq, sq_std, q, test_loss, 
                SROCC, KROCC, PLCC, RMSE, OR, test_index))