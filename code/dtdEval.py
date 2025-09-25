import os
import cv2
import lmdb
import torch
import jpegio
import numpy as np
import torch.nn as nn
import math
import logging
import torch.optim as optim
import pickle
import six
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # need pytorch>1.6

from data_loader import get_dataset_by_choice
from losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, LovaszLoss
from dtd import *
from albumentations.pytorch import ToTensorV2
import torchvision
import argparse
import tempfile



data_root = 'data/'
model_pth = 'code/result/DTD_base/DTDMyself-9.pth'
lmdb_name = 'DocTamperV1-TestingSet'
data_minq = 75


test_data, test_dataset_name = get_dataset_by_choice(4)
train_loader1 = DataLoader(dataset=test_data, batch_size=6, num_workers=8, shuffle=False)


class IOUMetric:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


model = seg_dtd_clrdis2('', 2)
model_pth = r'/pubdata/zhengshiqiang/checkpoint/upload_temp/0-DTD-clrdis-iou0.8239.pth'

# model = torch.nn.DataParallel(model)

def eval_net_dtd(model, test_data, plot=False, device='cuda:3'):
    train_loader1 = DataLoader(dataset=test_data, batch_size=12, num_workers=8, shuffle=False)
    LovaszLoss_fn = LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)


    ckpt = torch.load(model_pth, map_location='cpu')
    new_state_dict = {}
    for key, value in ckpt['state_dict'].items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)

    #     model.load_state_dict(torch.load(args.pth, map_location=torch.device('cpu')), strict=False)
    model.to(device)
    model.eval()
    iou = IOUMetric(2)
    precisons = []
    recalls = []
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(train_loader1)):
            data, target, dct_coef, qs, q = batch_samples['image'], batch_samples['label'], batch_samples['rgb'], \
                                            batch_samples['q'], batch_samples['i']
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(
                dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))

            pred, features_ave, bc_media = model(data, dct_coef, qs)

            predt = pred.argmax(1)
            pred = pred.cpu().data.numpy()
            targt = target.squeeze(1)
            matched = (predt * targt).sum((1, 2))
            pred_sum = predt.sum((1, 2))
            target_sum = targt.sum((1, 2))
            precisons.append((matched / (pred_sum + 1e-8)).mean().item())
            recalls.append((matched / target_sum).mean().item())
            pred = np.argmax(pred, axis=1)
            iou.add_batch(pred, target.cpu().data.numpy())

            # break

        acc, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
        precisons = np.array(precisons).mean()
        recalls = np.array(recalls).mean()
        print('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu, precisons, recalls,
                                                        (2 * precisons * recalls / (precisons + recalls + 1e-8))))


eval_net_dtd(model, test_data)
