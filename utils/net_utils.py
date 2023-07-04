import os
import gc
import time
import glob
import json
import cv2
import matplotlib.cm as cm
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from utils.tools import *
from utils.config import *
from data import joint_transforms
from data.data_loader import NoduleData
from utils.loss_functions import BinaryDiceLoss, DiceLoss
#
import argparse
import yaml
from sklearn import metrics


cls_criterion1 = nn.CrossEntropyLoss()
dice_criterion = DiceLoss()
kd_criterion = nn.KLDivLoss()
cls_criterion2 = nn.CrossEntropyLoss()


def prepare_net(config, model, _use='train'):
    normalize = transforms.Normalize(mean=config['Means'], std=config['Stds'])
    if config['mask_dilation']:
        mask_dilation = joint_transforms.JointMaskDilation()
    if _use == 'train':
        if config['optim'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         config['lr'], weight_decay=config['weight_decay'])
        if config['optim'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(),
                                            config['lr'], weight_decay=config['weight_decay'])
        if config['optim'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        config['lr'], momentum=config['momentum'],
                                        weight_decay=config['weight_decay'])

        if config['mask_dilation']:
            train_joint_transformer = transforms.Compose([mask_dilation,
                                                          joint_transforms.JointRandomHorizontalFlip(p=0.5),
                                                          joint_transforms.JointRandomVerticalFlip(p=0.5),
                                                          joint_transforms.JointRandomRotation(config['rota_factor'],
                                                                                               expend=False, p=0.5),
                                                          joint_transforms.JointZoomOut(config['zoom_factor']),
                                                          joint_transforms.JointRandomObjectCrop(),
                                                          joint_transforms.JointResize(config['img_size'])
                                                          ])
        else:
            train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip(p=0.5),
                                                          joint_transforms.JointRandomVerticalFlip(p=0.5),
                                                          joint_transforms.JointRandomRotation(config['rota_factor'],
                                                                                               expend=False, p=0.5),
                                                          joint_transforms.JointZoomOut(config['zoom_factor']),
                                                          joint_transforms.JointRandomObjectCrop(),
                                                          joint_transforms.JointResize(config['img_size'])
                                                          ])
        train_dataset = NoduleData(root=config['DataRoot'], mode='train',
                                   joint_transform=train_joint_transformer,
                                   transform=transforms.Compose([transforms.ToTensor(), normalize]),
                                   use_cam=config['Using_cam'])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=config['batchsize'],
                                                    shuffle=True,
                                                    num_workers=config['num_workers'])

        if config['mask_dilation']:
            valid_joint_transformer = transforms.Compose([mask_dilation, joint_transforms.JointResize(config['img_size'])])
        else:
            valid_joint_transformer = transforms.Compose([joint_transforms.JointResize(config['img_size'])])
        valid_dataset = NoduleData(root=config['DataRoot'], mode='validation',
                                   joint_transform=valid_joint_transformer,
                                   transform=transforms.Compose([transforms.ToTensor(), normalize]),
                                   use_cam=config['Using_cam'])
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=config['batchsize'],
                                                   shuffle=False,
                                                   num_workers=config['num_workers'])
        return optimizer, train_loader, valid_loader

    elif _use == 'test':
        if config['mask_dilation']:
            test_joint_transformer = transforms.Compose([mask_dilation, joint_transforms.JointResize(config['img_size'])])
        else:
            test_joint_transformer = transforms.Compose([joint_transforms.JointResize(config['img_size'])])
        test_dataset = NoduleData(root=config['DataRoot'], mode='test',
                                   joint_transform=test_joint_transformer,
                                   transform=transforms.Compose([transforms.ToTensor(), normalize]),
                                   use_cam=config['Using_cam'])
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=config['batchsize'], # 1?
                                                   shuffle=False,
                                                   num_workers=config['num_workers'])
        return test_loader


def train_net(optimizer, train_loader, valid_loader, model, config):
    best_MACC = 0.0
    best_MAUC = 0.0
    best_loss = np.inf

    if config['lr_decay'] == None:
        lr_decay = 0.1
    else:
        lr_decay = config['lr_decay']

    for epoch in range(1, config['num_epoch'] + 1):
        print('----------Epoch Start Training----------', time.ctime())
        adjust_learning_rate(optimizer, epoch, config['num_epoch'], config['lr'],
                             config['lr_decay_freq'], lr_decay)

        if config['Using_cam'] and config['Using_aggregation']:
            train_result, train_losses, train_MACC, train_MAUC, train_MIoU = train_cam_agg(train_loader, model, optimizer, epoch, config)
            print(train_result)

            if epoch % config['test_freq'] == 0:
                val_result, val_losses, val_MACC, val_MAUC, val_MIoU = val_cam_agg(valid_loader, model, epoch, config)
                print(val_result)

                if val_MACC >= best_MACC:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestacc=True, best=val_MACC)
                    best_MACC = val_MACC
                if val_MAUC >= best_MAUC:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestauc=True, best=val_MAUC)
                    best_MAUC = val_MAUC
                if val_losses <= best_loss:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestloss=True, best=val_losses)
                    best_loss = val_losses

            if epoch % config['save_model_freq'] == 0:
                save_checkpoint(model, config['arch'], config['model'], epoch, log=True, train_log=train_result, valid_log=val_result)
            else:
                save_checkpoint(model, config['arch'], config['model'], epoch, only_log=True, train_log=train_result, valid_log=val_result)


        if config['Using_cam'] and not config['Using_aggregation']:
            train_result, train_losses, train_MACC, train_MAUC, train_MIoU = train_cam(train_loader, model, optimizer, epoch, config)
            print(train_result)

            if epoch % config['test_freq'] == 0:
                val_result, val_losses, val_MACC, val_MAUC, val_MIoU = val_cam(valid_loader, model, epoch, config)
                print(val_result)

                if val_MACC >= best_MACC:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestacc=True, best=val_MACC)
                    best_MACC = val_MACC
                if val_MAUC >= best_MAUC:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestauc=True, best=val_MAUC)
                    best_MAUC = val_MAUC
                if val_losses <= best_loss:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestloss=True, best=val_losses)
                    best_loss = val_losses

            if epoch % config['save_model_freq'] == 0:
                save_checkpoint(model, config['arch'], config['model'], epoch, log=True, train_log=train_result, valid_log=val_result)
            else:
                save_checkpoint(model, config['arch'], config['model'], epoch, only_log=True, train_log=train_result, valid_log=val_result)


        if not config['Using_cam'] and config['Using_aggregation']:
            train_result, train_losses, train_MACC, train_MAUC = train_agg(train_loader, model, optimizer, epoch, config)
            print(train_result)

            if epoch % config['test_freq'] == 0:
                val_result, val_losses, val_MACC, val_MAUC = val_agg(valid_loader, model, epoch, config)
                print(val_result)

                if val_MACC >= best_MACC:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestacc=True, best=val_MACC)
                    best_MACC = val_MACC
                if val_MAUC >= best_MAUC:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestauc=True, best=val_MAUC)
                    best_MAUC = val_MAUC
                if val_losses <= best_loss:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestloss=True, best=val_losses)
                    best_loss = val_losses

            if epoch % config['save_model_freq'] == 0:
                save_checkpoint(model, config['arch'], config['model'], epoch, log=True, train_log=train_result, valid_log=val_result)
            else:
                save_checkpoint(model, config['arch'], config['model'], epoch, only_log=True, train_log=train_result, valid_log=val_result)


        if not config['Using_cam'] and not config['Using_aggregation']:
            train_result, train_losses, train_MACC, train_MAUC = train_vanilla(train_loader, model, optimizer, epoch, config)
            print(train_result)

            if epoch % config['test_freq'] == 0:
                val_result, val_losses, val_MACC, val_MAUC = val_vanilla(valid_loader, model, epoch, config)
                print(val_result)

                if val_MACC >= best_MACC:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestacc=True, best=val_MACC)
                    best_MACC = val_MACC
                if val_MAUC >= best_MAUC:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestauc=True, best=val_MAUC)
                    best_MAUC = val_MAUC
                if val_losses <= best_loss:
                    save_checkpoint(model, config['arch'], config['model'], epoch, _bestloss=True, best=val_losses)
                    best_loss = val_losses

            if epoch % config['save_model_freq'] == 0:
                save_checkpoint(model, config['arch'], config['model'], epoch, log=True, train_log=train_result, valid_log=val_result)
            else:
                save_checkpoint(model, config['arch'], config['model'], epoch, only_log=True, train_log=train_result, valid_log=val_result)

    save_checkpoint(model, config['arch'], config['model'], epoch)


def test_net(test_loader, model, config, args):

    if config['Using_cam'] and config['Using_aggregation']:
        test_result, MACC, MPRE, MREC, MF1S, MAUC, MIoU, acc1, acc2, auc1, auc2 = test_cam_agg(test_loader, model, config, args)
        print(test_result)

    if config['Using_cam'] and not config['Using_aggregation']:
        test_result, MACC, MPRE, MREC, MF1S, MAUC, MIoU = test_cam(test_loader, model, config, args)
        print(test_result)

    if not config['Using_cam'] and config['Using_aggregation']:
        test_result, MACC, MPRE, MREC, MF1S, MAUC, acc1, acc2, auc1, auc2 = test_agg(test_loader, model, config, args)
        print(test_result)

    if not config['Using_cam'] and not config['Using_aggregation']:
        test_result, MACC, MPRE, MREC, MF1S, MAUC = test_vanilla(test_loader, model, config, args)
        print(test_result)


def train_cam_agg(train_loader, model, optimizer, epoch, config):
    assert config['Using_cam'] and config['Using_aggregation']

    T = config['distill_temp']

    Cls1_losses = 0.
    Dice_losses = 0.
    KD_losses = 0.
    Cls2_losses = 0.
    losses = 0.
    MIoU = 0.

    model.train()
    end = time.time()

    for i, (inputs, masks, labels) in enumerate(train_loader):

        inputs = inputs.cuda()
        im_h = inputs.size(2)
        im_w = inputs.size(3)
        bs = inputs.size(0)
        labels = labels.cuda()
        masks = masks.cuda()

        out_classes, cam_classes_refined, cam_classes, aux_classes = model(inputs)

        cls1Loss = cls_criterion1(out_classes, labels)

        cls2Loss = cls_criterion2(aux_classes, labels)

        diceLoss = dice_criterion(cam_classes_refined, masks)

        distillLoss = kd_criterion(F.log_softmax(out_classes / T, dim=1),
                                   F.softmax(aux_classes / T, dim=1))  * T * T

        # loss_cls1 = 1; loss_cls2 = 1; loss_cam = 0.6; loss_dill = 0.5
        loss = loss_cls1 * cls1Loss + loss_cam * diceLoss + loss_cls2 * cls2Loss + loss_dill * distillLoss

        Cls1_losses += loss_cls1 * cls1Loss.item()
        Dice_losses += loss_cam * diceLoss.item()
        Cls2_losses += loss_cls2 * cls2Loss.item()
        KD_losses += loss_dill * distillLoss.item()
        losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        MIoU += get_MIoU(cam_classes_refined, masks)

        out_preds = F.softmax(out_classes, dim=1)
        aux_preds = F.softmax(aux_classes, dim=1)

        if i == 0:
            y_gt_cls = labels.detach().cpu().numpy()
            y_pred_cls1 = out_preds.detach().cpu().numpy()
            y_pred_cls2 = aux_preds.detach().cpu().numpy()
        else:
            y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
            y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.detach().cpu().numpy()), axis=0)
            y_pred_cls2 = np.concatenate((y_pred_cls2, aux_preds.detach().cpu().numpy()), axis=0)

    # Epoch losses
    Cls1_losses /= len(train_loader)
    Dice_losses /= len(train_loader)
    Cls2_losses /= len(train_loader)
    KD_losses /= len(train_loader)
    losses /= len(train_loader)
    # Epoch metrics
    Cls1_ACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
    Cls2_ACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))

    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
    Cls1_AUC = metrics.auc(fpr1, tpr1)

    fpr2, tpr2, thresholds2 = metrics.roc_curve(y_gt_cls, y_pred_cls2[:, 1])
    Cls2_AUC = metrics.auc(fpr2, tpr2)

    y_pred_mcls = np.add(y_pred_cls1, y_pred_cls2) / 2
    MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
    fprm, tprm, thresholdsm = metrics.roc_curve(y_gt_cls, y_pred_mcls[:, 1])
    MAUC = metrics.auc(fprm, tprm)

    MIoU /= len(train_loader)

    epoch_time = time.time() - end

    result_train = 'Epoch: {}, Time: {:.3f}, ' \
                   'Train Loss: {:.4f}, Cls1_loss: {:.4f}, Cls2_loss: {:.4f}, Dice_loss: {:.4f}, KD_loss: {:.4f},' \
                   'Cls1_ACC: {:.4f}, Cls2_ACC: {:.4f}, Mean_ACC: {:.4f}, Mean_AUC: {:.4f}, Cam_MIoU: {:.4f}'.format(epoch, epoch_time,
                                                                                  losses, Cls1_losses, Cls2_losses,
                                                                                  Dice_losses, KD_losses,
                                                                                  Cls1_ACC, Cls2_ACC, MACC, MAUC, MIoU)

    return result_train, losses, MACC, MAUC, MIoU


def train_cam(train_loader, model, optimizer, epoch, config):
    assert config['Using_cam'] and not config['Using_aggregation']

    Cls1_losses = 0.
    Dice_losses = 0.
    losses = 0.
    MIoU = 0.

    model.train()
    end = time.time()

    for i, (inputs, masks, labels) in enumerate(train_loader):

        inputs = inputs.cuda()
        im_h = inputs.size(2)
        im_w = inputs.size(3)
        bs = inputs.size(0)
        labels = labels.cuda()
        masks = masks.cuda()

        out_classes, cam_classes_refined, cam_classes = model(inputs)

        cls1Loss = cls_criterion1(out_classes, labels)

        diceLoss = dice_criterion(cam_classes_refined, masks)

        loss = loss_cls1 * cls1Loss + loss_cam * diceLoss

        Cls1_losses += loss_cls1 * cls1Loss.item()
        Dice_losses += loss_cam * diceLoss.item()
        losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        MIoU += get_MIoU(F.softmax(cam_classes_refined), masks)

        out_preds = F.softmax(out_classes, dim=1)

        if i == 0:
            y_gt_cls = labels.detach().cpu().numpy()
            y_pred_cls1 = out_preds.detach().cpu().numpy()
        else:
            y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
            y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.detach().cpu().numpy()), axis=0)

    # Epoch losses
    Cls1_losses /= len(train_loader)
    Dice_losses /= len(train_loader)
    losses /= len(train_loader)
    # Epoch metrics
    MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))

    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
    MAUC = metrics.auc(fpr1, tpr1)

    MIoU /= len(train_loader)

    epoch_time = time.time() - end

    result_train = 'Epoch: {}, Time: {:.3f}, ' \
                   'Train Loss: {:.4f}, Cls1_loss: {:.4f}, Dice_loss: {:.4f},' \
                   'Cls1_ACC: {:.4f}, Cls1_AUC: {:.4f}, Cam_MIoU: {:.4f}'.format(epoch, epoch_time, losses, Cls1_losses, Dice_losses, MACC, MAUC, MIoU)

    return result_train, losses, MACC, MAUC, MIoU


def train_agg(train_loader, model, optimizer, epoch, config):
    assert not config['Using_cam'] and config['Using_aggregation']
    T = config['distill_temp']
    Cls1_losses = 0.
    KD_losses = 0.  # KL散度
    Cls2_losses = 0.
    losses = 0.

    model.train()
    end = time.time()

    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.cuda()
        im_h = inputs.size(2)
        im_w = inputs.size(3)
        bs = inputs.size(0)
        labels = labels.cuda()

        out_classes, aux_classes = model(inputs)

        cls1Loss = cls_criterion1(out_classes, labels)

        cls2Loss = cls_criterion2(aux_classes, labels)

        distillLoss = kd_criterion(F.log_softmax(out_classes / T, dim=1),
                                   F.softmax(aux_classes / T, dim=1)) * T * T

        loss = loss_cls1 * cls1Loss + loss_cls2 * cls2Loss + loss_dill * distillLoss

        Cls1_losses += loss_cls1 * cls1Loss.item()
        Cls2_losses += loss_cls2 * cls2Loss.item()
        KD_losses += loss_dill * distillLoss.item()
        losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out_preds = F.softmax(out_classes, dim=1)
        aux_preds = F.softmax(aux_classes, dim=1)

        if i == 0:
            y_gt_cls = labels.detach().cpu().numpy()
            y_pred_cls1 = out_preds.detach().cpu().numpy()
            y_pred_cls2 = aux_preds.detach().cpu().numpy()
        else:
            y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
            y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.detach().cpu().numpy()), axis=0)
            y_pred_cls2 = np.concatenate((y_pred_cls2, aux_preds.detach().cpu().numpy()), axis=0)

    # Epoch losses
    Cls1_losses /= len(train_loader)
    Cls2_losses /= len(train_loader)
    KD_losses /= len(train_loader)
    losses /= len(train_loader)
    # Epoch metrics
    Cls1_ACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
    Cls2_ACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))

    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
    Cls1_AUC = metrics.auc(fpr1, tpr1)

    fpr2, tpr2, thresholds2 = metrics.roc_curve(y_gt_cls, y_pred_cls2[:, 1])
    Cls2_AUC = metrics.auc(fpr2, tpr2)

    y_pred_mcls = np.add(y_pred_cls1, y_pred_cls2) / 2
    MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
    fprm, tprm, thresholdsm = metrics.roc_curve(y_gt_cls, y_pred_mcls[:, 1])
    MAUC = metrics.auc(fprm, tprm)

    epoch_time = time.time() - end

    result_train = 'Epoch: {}, Time: {:.3f}, ' \
                   'Train Loss: {:.4f}, Cls1_loss: {:.4f}, Cls2_loss: {:.4f}, KD_loss: {:.4f},' \
                   'Cls1_ACC: {:.4f}, Cls2_ACC: {:.4f}, Mean_ACC: {:.4f}, Mean_AUC: {:.4f}'.format(epoch, epoch_time,
                                                                                  losses, Cls1_losses, Cls2_losses,
                                                                                  KD_losses,
                                                                                  Cls1_ACC, Cls2_ACC, MACC, MAUC)

    return result_train, losses, MACC, MAUC


def train_vanilla(train_loader, model, optimizer, epoch, config):
    assert not config['Using_cam'] and not config['Using_aggregation']

    Cls1_losses = 0.
    losses = 0.

    model.train()
    end = time.time()

    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.cuda()
        im_h = inputs.size(2)
        im_w = inputs.size(3)
        bs = inputs.size(0)
        labels = labels.cuda()

        out_classes = model(inputs)

        cls1Loss = cls_criterion1(out_classes, labels)

        loss = loss_cls1 * cls1Loss

        Cls1_losses += loss_cls1 * cls1Loss.item()
        losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out_preds = F.softmax(out_classes, dim=1)

        if i == 0:
            y_gt_cls = labels.detach().cpu().numpy()
            y_pred_cls1 = out_preds.detach().cpu().numpy()
        else:
            y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
            y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.detach().cpu().numpy()), axis=0)

    # Epoch losses
    Cls1_losses /= len(train_loader)
    losses /= len(train_loader)
    # Epoch metrics
    MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
    MAUC = metrics.auc(fpr1, tpr1)

    epoch_time = time.time() - end

    result_train = 'Epoch: {}, Time: {:.3f}, ' \
                   'Train Loss: {:.4f}, Cls1_loss: {:.4f}, ' \
                   'Cls1_ACC: {:.4f}, Cls1_AUC: {:.4f}'.format(epoch, epoch_time, losses, Cls1_losses, MACC, MAUC)

    return result_train, losses, MACC, MAUC


def val_cam_agg(valid_loader, model, epoch, config):
    assert config['Using_cam'] and config['Using_aggregation']

    T = config['distill_temp']

    Cls1_losses = 0.
    Dice_losses = 0.
    KD_losses = 0.
    Cls2_losses = 0.
    losses = 0.
    MIoU = 0.

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, masks, labels) in enumerate(valid_loader):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            labels = labels.cuda()
            masks = masks.cuda()

            out_classes, cam_classes_refined, cam_classes, aux_classes = model(inputs)

            cls1Loss = cls_criterion1(out_classes, labels)

            cls2Loss = cls_criterion2(aux_classes, labels)

            diceLoss = dice_criterion(cam_classes_refined, masks)

            distillLoss = kd_criterion(F.log_softmax(out_classes / T, dim=1),
                                       F.softmax(aux_classes / T, dim=1)) * T * T

            loss = loss_cls1 * cls1Loss + loss_cam * diceLoss + loss_cls2 * cls2Loss + loss_dill * distillLoss

            Cls1_losses += loss_cls1 * cls1Loss.item()
            Dice_losses += loss_cam * diceLoss.item()
            Cls2_losses += loss_cls2 * cls2Loss.item()
            KD_losses += loss_dill * distillLoss.item()
            losses += loss.item()

            MIoU += get_MIoU(cam_classes_refined, masks)

            out_preds = F.softmax(out_classes, dim=1)
            aux_preds = F.softmax(aux_classes, dim=1)

            if i == 0:
                y_gt_cls = labels.detach().cpu().numpy()
                y_pred_cls1 = out_preds.cpu().numpy()
                y_pred_cls2 = aux_preds.cpu().numpy()
            else:
                y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
                y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.cpu().numpy()), axis=0)
                y_pred_cls2 = np.concatenate((y_pred_cls2, aux_preds.cpu().numpy()), axis=0)


        # Epoch losses
        Cls1_losses /= len(valid_loader)
        Dice_losses /= len(valid_loader)
        Cls2_losses /= len(valid_loader)
        KD_losses /= len(valid_loader)
        losses /= len(valid_loader)
        # Epoch metrics
        Cls1_ACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        Cls2_ACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))

        fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
        Cls1_AUC = metrics.auc(fpr1, tpr1)

        fpr2, tpr2, thresholds2 = metrics.roc_curve(y_gt_cls, y_pred_cls2[:, 1])
        Cls2_AUC = metrics.auc(fpr2, tpr2)

        y_pred_mcls = np.add(y_pred_cls1, y_pred_cls2) / 2
        MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
        fprm, tprm, thresholdsm = metrics.roc_curve(y_gt_cls, y_pred_mcls[:, 1])
        MAUC = metrics.auc(fprm, tprm)

        MIoU /= len(valid_loader)

        epoch_time = time.time() - end

        result_val = 'Epoch: {}, Time: {:.3f}, ' \
                       'Valid Loss: {:.4f}, Cls1_loss: {:.4f}, Cls2_loss: {:.4f}, Dice_loss: {:.4f}, KD_loss: {:.4f},' \
                       'Cls1_ACC: {:.4f}, Cls2_ACC: {:.4f}, Mean_ACC: {:.4f}, Mean_AUC: {:.4f}, Cam_MIoU: {:.4f}'.format(epoch, epoch_time,
                                                                                     losses, Cls1_losses, Cls2_losses,
                                                                                     Dice_losses, KD_losses,
                                                                                     Cls1_ACC, Cls2_ACC, MACC, MAUC, MIoU)

        return result_val, losses, MACC, MAUC, MIoU


def val_cam(valid_loader, model, epoch, config):
    assert config['Using_cam'] and not config['Using_aggregation']

    Cls1_losses = 0.
    Dice_losses = 0.
    losses = 0.
    MIoU = 0.

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, masks, labels) in enumerate(valid_loader):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            labels = labels.cuda()
            masks = masks.cuda()

            out_classes, cam_classes_refined, cam_classes = model(inputs)

            cls1Loss = cls_criterion1(out_classes, labels)

            diceLoss = dice_criterion(cam_classes_refined, masks)

            loss = loss_cls1 * cls1Loss + loss_cam * diceLoss

            Cls1_losses += loss_cls1 * cls1Loss.item()
            Dice_losses += loss_cam * diceLoss.item()
            losses += loss.item()

            MIoU += get_MIoU(F.softmax(cam_classes_refined), masks)

            out_preds = F.softmax(out_classes, dim=1)

            if i == 0:
                y_gt_cls = labels.detach().cpu().numpy()
                y_pred_cls1 = out_preds.cpu().numpy()
            else:
                y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
                y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.cpu().numpy()), axis=0)

        # Epoch losses
        Cls1_losses /= len(valid_loader)
        Dice_losses /= len(valid_loader)
        losses /= len(valid_loader)
        # Epoch metrics
        MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
        MAUC = metrics.auc(fpr1, tpr1)

        MIoU /= len(valid_loader)

        epoch_time = time.time() - end

        result_val = 'Epoch: {}, Time: {:.3f}, ' \
                     'Valid Loss: {:.4f}, Cls1_loss: {:.4f}, Dice_loss: {:.4f},' \
                     'Cls1_ACC: {:.4f}, Cls1_AUC: {:.4f}, Cam_MIoU: {:.4f}'.format(epoch, epoch_time, losses, Cls1_losses, Dice_losses, MACC, MAUC, MIoU)

        return result_val, losses, MACC, MAUC, MIoU


def val_agg(valid_loader, model, epoch, config):
    assert not config['Using_cam'] and config['Using_aggregation']

    T = config['distill_temp']

    Cls1_losses = 0.
    KD_losses = 0.
    Cls2_losses = 0.
    losses = 0.

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            labels = labels.cuda()

            out_classes, aux_classes = model(inputs)

            cls1Loss = cls_criterion1(out_classes, labels)

            cls2Loss = cls_criterion2(aux_classes, labels)

            distillLoss = kd_criterion(F.log_softmax(out_classes / T, dim=1),
                                       F.softmax(aux_classes / T, dim=1)) * T * T

            loss = loss_cls1 * cls1Loss + loss_cls2 * cls2Loss + loss_dill * distillLoss

            Cls1_losses += loss_cls1 * cls1Loss.item()
            Cls2_losses += loss_cls2 * cls2Loss.item()
            KD_losses += loss_dill * distillLoss.item()
            losses += loss.item()

            out_preds = F.softmax(out_classes, dim=1)
            aux_preds = F.softmax(aux_classes, dim=1)

            if i == 0:
                y_gt_cls = labels.detach().cpu().numpy()
                y_pred_cls1 = out_preds.cpu().numpy()
                y_pred_cls2 = aux_preds.cpu().numpy()
            else:
                y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
                y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.cpu().numpy()), axis=0)
                y_pred_cls2 = np.concatenate((y_pred_cls2, aux_preds.cpu().numpy()), axis=0)


        # Epoch losses
        Cls1_losses /= len(valid_loader)
        Cls2_losses /= len(valid_loader)
        KD_losses /= len(valid_loader)
        losses /= len(valid_loader)
        # Epoch metrics
        Cls1_ACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        Cls2_ACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))

        fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
        Cls1_AUC = metrics.auc(fpr1, tpr1)

        fpr2, tpr2, thresholds2 = metrics.roc_curve(y_gt_cls, y_pred_cls2[:, 1])
        Cls2_AUC = metrics.auc(fpr2, tpr2)

        y_pred_mcls = np.add(y_pred_cls1, y_pred_cls2) / 2
        MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
        fprm, tprm, thresholdsm = metrics.roc_curve(y_gt_cls, y_pred_mcls[:, 1])
        MAUC = metrics.auc(fprm, tprm)

        epoch_time = time.time() - end

        result_val = 'Epoch: {}, Time: {:.3f}, ' \
                       'Valid Loss: {:.4f}, Cls1_loss: {:.4f}, Cls2_loss: {:.4f}, KD_loss: {:.4f},' \
                       'Cls1_ACC: {:.4f}, Cls2_ACC: {:.4f}, Mean_ACC: {:.4f}, Mean_AUC: {:.4f}'.format(epoch, epoch_time,
                                                                                     losses, Cls1_losses, Cls2_losses,
                                                                                     KD_losses,
                                                                                     Cls1_ACC, Cls2_ACC, MACC, MAUC)

        return result_val, losses, MACC, MAUC


def val_vanilla(valid_loader, model, epoch, config):
    assert not config['Using_cam'] and not config['Using_aggregation']

    Cls1_losses = 0.
    losses = 0.

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            labels = labels.cuda()

            out_classes = model(inputs)

            cls1Loss = cls_criterion1(out_classes, labels)

            loss = loss_cls1 * cls1Loss

            Cls1_losses += loss_cls1 * cls1Loss.item()

            losses += loss.item()

            out_preds = F.softmax(out_classes, dim=1)

            if i == 0:
                y_gt_cls = labels.detach().cpu().numpy()
                y_pred_cls1 = out_preds.cpu().numpy()
            else:
                y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
                y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.cpu().numpy()), axis=0)

        # Epoch losses
        Cls1_losses /= len(valid_loader)
        losses /= len(valid_loader)
        # Epoch metrics
        MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
        MAUC = metrics.auc(fpr1, tpr1)

        epoch_time = time.time() - end

        result_val = 'Epoch: {}, Time: {:.3f}, ' \
                     'Valid Loss: {:.4f}, Cls1_loss: {:.4f},' \
                     'Cls1_ACC: {:.4f}, Cls1_AUC: {:.4f}'.format(epoch, epoch_time, losses, Cls1_losses, MACC, MAUC)

        return result_val, losses, MACC, MAUC


def test_cam_agg(test_loader, model, config, args):
    assert config['Using_cam'] and config['Using_aggregation']
    MIoU = 0.0

    preds_path = os.path.join(args.weight.split('/')[0],args.weight.split('/')[1],args.weight.split('/')[2],'test')
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, masks, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            labels = labels.cuda()
            masks = masks.cuda()

            out_classes, cam_classes_refined, cam_classes, aux_classes = model(inputs)

            MIoU += get_MIoU(cam_classes_refined, masks)

            out_preds = F.softmax(out_classes, dim=1)
            aux_preds = F.softmax(aux_classes, dim=1)

            if i == 0:
                y_gt_cls = labels.detach().cpu().numpy()
                y_pred_cls1 = out_preds.cpu().numpy()
                y_pred_cls2 = aux_preds.cpu().numpy()
            else:
                y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
                y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.cpu().numpy()), axis=0)
                y_pred_cls2 = np.concatenate((y_pred_cls2, aux_preds.cpu().numpy()), axis=0)

        acc1 = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        pre1 = metrics.precision_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        rec1 = metrics.recall_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        f1_1 = metrics.f1_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
        auc1 = metrics.auc(fpr1, tpr1)

        acc2 = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))
        pre2 = metrics.precision_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))
        rec2 = metrics.recall_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))
        f1_2 = metrics.f1_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))
        fpr2, tpr2, thresholds2 = metrics.roc_curve(y_gt_cls, y_pred_cls2[:, 1])
        auc2 = metrics.auc(fpr2, tpr2)

        y_pred_mcls = np.add(y_pred_cls1, y_pred_cls2) / 2
        MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
        MPRE = metrics.precision_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
        MREC = metrics.recall_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
        MF1S = metrics.f1_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
        fprm, tprm, thresholdsm = metrics.roc_curve(y_gt_cls, y_pred_mcls[:, 1])
        MAUC = metrics.auc(fprm, tprm)

        MIoU /= len(test_loader)

        detail_df1 = pd.DataFrame(
            {'true_label': y_gt_cls, 'predicted_label': np.argmax(y_pred_cls1, axis=1), 'p0': y_pred_cls1[:, 0],
             'p1': y_pred_cls1[:, 1]})
        detail_df1.reset_index(inplace=True, drop=True)
        detail_df1.to_csv(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_detailed1_results.tsv'),
                          index=False,
                          sep='\t')

        detail_df2 = pd.DataFrame(
            {'true_label': y_gt_cls, 'predicted_label': np.argmax(y_pred_cls2, axis=1), 'p0': y_pred_cls2[:, 0],
             'p1': y_pred_cls2[:, 1]})
        detail_df2.reset_index(inplace=True, drop=True)
        detail_df2.to_csv(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_detailed2_results.tsv'),
                          index=False,
                          sep='\t')

        detail_dfm = pd.DataFrame(
            {'true_label': y_gt_cls, 'predicted_label': np.argmax(y_pred_mcls, axis=1), 'p0': y_pred_mcls[:, 0],
             'p1': y_pred_mcls[:, 1]})
        detail_dfm.reset_index(inplace=True, drop=True)
        detail_dfm.to_csv(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_detailedm_results.tsv'),
                          index=False,
                          sep='\t')

        test_result = 'Classification Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, AUC: {:.4f}, MIoU: {:.4f};' \
                      'Cls1 Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, AUC: {:.4f};' \
                      'Cls2 Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, AUC: {:.4f}'\
                                                        .format(MACC, MPRE, MREC, MF1S, MAUC, MIoU,
                                                                  acc1, pre1, rec1, f1_1, auc1,
                                                                  acc2, pre2, rec2, f1_2, auc2)
        file = open(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_metric_results.txt'), "w")
        file.write(test_result + '\n')
        file.close()

        return test_result, MACC, MPRE, MREC, MF1S, MAUC, MIoU, acc1, acc2, auc1, auc2


def test_cam(test_loader, model, config, args):
    assert config['Using_cam'] and not config['Using_aggregation']
    MIoU = 0.0

    preds_path = os.path.join(args.weight.split('/')[0],args.weight.split('/')[1],args.weight.split('/')[2],'test')
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, masks, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            labels = labels.cuda()
            masks = masks.cuda()

            out_classes, cam_classes_refined, cam_classes= model(inputs)

            MIoU += get_MIoU(cam_classes_refined, masks)

            out_preds = F.softmax(out_classes, dim=1)

            if i == 0:
                y_gt_cls = labels.detach().cpu().numpy()
                y_pred_cls1 = out_preds.cpu().numpy()
            else:
                y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
                y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.cpu().numpy()), axis=0)

        MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        MPRE = metrics.precision_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        MREC = metrics.recall_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        MF1S = metrics.f1_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
        MAUC = metrics.auc(fpr1, tpr1)

        MIoU /= len(test_loader)

        detail_df = pd.DataFrame(
            {'true_label': y_gt_cls, 'predicted_label': np.argmax(y_pred_cls1, axis=1), 'p0': y_pred_cls1[:, 0],
             'p1': y_pred_cls1[:, 1]})
        detail_df.reset_index(inplace=True, drop=True)
        detail_df.to_csv(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_detailed_results.tsv'),
                         index=False,
                         sep='\t')

        test_result = 'Classification Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, AUC: {:.4f}, ' \
                      'MIoU: {:.4f}'.format(MACC, MPRE, MREC, MF1S, MAUC, MIoU)
        file = open(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_metric_results.txt'), "w")
        file.write(test_result + '\n')
        file.close()

        return test_result, MACC, MPRE, MREC, MF1S, MAUC, MIoU


def test_agg(test_loader, model, config, args):
    assert not config['Using_cam'] and config['Using_aggregation']

    preds_path = os.path.join(args.weight.split('/')[0],args.weight.split('/')[1],args.weight.split('/')[2],'test')
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            labels = labels.cuda()

            out_classes, aux_classes = model(inputs)

            out_preds = F.softmax(out_classes, dim=1)
            aux_preds = F.softmax(aux_classes, dim=1)

            if i == 0:
                y_gt_cls = labels.detach().cpu().numpy()
                y_pred_cls1 = out_preds.cpu().numpy()
                y_pred_cls2 = aux_preds.cpu().numpy()
            else:
                y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
                y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.cpu().numpy()), axis=0)
                y_pred_cls2 = np.concatenate((y_pred_cls2, aux_preds.cpu().numpy()), axis=0)


        acc1 = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        pre1 = metrics.precision_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        rec1 = metrics.recall_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        f1_1 = metrics.f1_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
        auc1 = metrics.auc(fpr1, tpr1)

        acc2 = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))
        pre2 = metrics.precision_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))
        rec2 = metrics.recall_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))
        f1_2 = metrics.f1_score(y_gt_cls, np.argmax(y_pred_cls2, axis=1))
        fpr2, tpr2, thresholds2 = metrics.roc_curve(y_gt_cls, y_pred_cls2[:, 1])
        auc2 = metrics.auc(fpr2, tpr2)

        y_pred_mcls = np.add(y_pred_cls1, y_pred_cls2) / 2
        MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
        MPRE = metrics.precision_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
        MREC = metrics.recall_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
        MF1S = metrics.f1_score(y_gt_cls, np.argmax(y_pred_mcls, axis=1))
        fprm, tprm, thresholdsm = metrics.roc_curve(y_gt_cls, y_pred_mcls[:, 1])
        MAUC = metrics.auc(fprm, tprm)


        detail_df1 = pd.DataFrame(
            {'true_label': y_gt_cls, 'predicted_label': np.argmax(y_pred_cls1, axis=1), 'p0': y_pred_cls1[:, 0],
             'p1': y_pred_cls1[:, 1]})
        detail_df1.reset_index(inplace=True, drop=True)
        detail_df1.to_csv(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_detailed1_results.tsv'),
                         index=False,
                         sep='\t')

        detail_df2 = pd.DataFrame(
            {'true_label': y_gt_cls, 'predicted_label': np.argmax(y_pred_cls2, axis=1), 'p0': y_pred_cls2[:, 0],
             'p1': y_pred_cls2[:, 1]})
        detail_df2.reset_index(inplace=True, drop=True)
        detail_df2.to_csv(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_detailed2_results.tsv'),
                          index=False,
                          sep='\t')

        detail_dfm = pd.DataFrame(
            {'true_label': y_gt_cls, 'predicted_label': np.argmax(y_pred_mcls, axis=1), 'p0': y_pred_mcls[:, 0],
             'p1': y_pred_mcls[:, 1]})
        detail_dfm.reset_index(inplace=True, drop=True)
        detail_dfm.to_csv(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_detailedm_results.tsv'),
                          index=False,
                          sep='\t')

        test_result = 'Classification Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, AUC: {:.4f};' \
                      'Cls1 Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, AUC: {:.4f};' \
                      'Cls2 Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, AUC: {:.4f}'\
                                                        .format(MACC, MPRE, MREC, MF1S, MAUC,
                                                                  acc1, pre1, rec1, f1_1, auc1,
                                                                  acc2, pre2, rec2, f1_2, auc2)

        file = open(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_metric_results.txt'), "w")
        file.write(test_result + '\n')
        file.close()

        return test_result, MACC, MPRE, MREC, MF1S, MAUC, acc1, acc2, auc1, auc2


def test_vanilla(test_loader, model, config, args):
    assert not config['Using_cam'] and not config['Using_aggregation']

    preds_path = os.path.join(args.weight.split('/')[0],args.weight.split('/')[1],args.weight.split('/')[2],'test')
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            im_h = inputs.size(2)
            im_w = inputs.size(3)
            bs = inputs.size(0)
            labels = labels.cuda()

            out_classes = model(inputs)

            out_preds = F.softmax(out_classes, dim=1)

            if i == 0:
                y_gt_cls = labels.detach().cpu().numpy()
                y_pred_cls1 = out_preds.cpu().numpy()
            else:
                y_gt_cls = np.concatenate((y_gt_cls, labels.detach().cpu().numpy()), axis=0)
                y_pred_cls1 = np.concatenate((y_pred_cls1, out_preds.cpu().numpy()), axis=0)

        MACC = metrics.accuracy_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        MPRE = metrics.precision_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        MREC = metrics.recall_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        MF1S = metrics.f1_score(y_gt_cls, np.argmax(y_pred_cls1, axis=1))
        fpr1, tpr1, thresholds1 = metrics.roc_curve(y_gt_cls, y_pred_cls1[:, 1])
        MAUC = metrics.auc(fpr1, tpr1)

        detail_df = pd.DataFrame(
            {'true_label': y_gt_cls, 'predicted_label': np.argmax(y_pred_cls1, axis=1), 'p0': y_pred_cls1[:, 0],
             'p1': y_pred_cls1[:, 1]})
        detail_df.reset_index(inplace=True, drop=True)
        detail_df.to_csv(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth') + '_detailed_results.tsv'),
                         index=False,
                         sep='\t')

        test_result = 'Classification Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, AUC: {:.4f}'.format(MACC, MPRE, MREC, MF1S, MAUC)

        file = open(os.path.join(preds_path, args.weight.split('/')[-1].strip('.pth')+ '_metric_results.txt'), "w")
        file.write(test_result + '\n')
        file.close()

        return test_result, MACC, MPRE, MREC, MF1S, MAUC