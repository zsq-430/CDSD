import json
import os

from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.cuda.amp import GradScaler

from data_loader import get_dataset_by_choice, TamperDataset
from losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, LovaszLoss
from dtd import seg_dtd, seg_dtd_clrdis2
from swins import *  # Import swins module to ensure BasicLayer class is available
# from model_architectures.dtd_VPH_change import *
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
import random
import lmdb
import tempfile
import six
from PIL import Image
import math
from tqdm import tqdm


def pearson_correlation(x, y):
    # Calculate the mean of vectors
    std_x = torch.std(x)
    std_y = torch.std(y)

    # Calculate numerator and denominator
    cov = torch.mean((x - torch.mean(x)) * (y - torch.mean(y)))

    denominator = std_x * std_y

    # Calculate Pearson correlation coefficient
    r = cov / denominator

    return r


# Wrapper for torch
def pearson_coefficient(x, y):
    """
    Calculate Pearson coefficient between two feature maps
    Args:
        x: shape=(batch_size, channels, height, width)
        y: shape=(batch_size, channels, height, width)
    Returns:
        pearson_coefficient: shape=(batch_size,)
    """
    batch_size, channels, height, width = x.size()

    x = torch.mean(x, dim=[2, 3])
    y = torch.mean(y, dim=[2, 3])

    corr_matrix = torch.zeros(batch_size, batch_size)
    for i in range(batch_size):
        for j in range(batch_size):
            corr_matrix[i][j] = pearson_correlation(x[i], y[j])

    f2_norm = torch.norm(corr_matrix, p=2)

    return f2_norm









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
        acc = np.diag(self.hist).sum() / (self.hist.sum() + 1e-5)
        acc_cls = np.diag(self.hist) / (self.hist.sum(axis=1) + 1e-5)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        prec = np.diag(self.hist) / (self.hist.sum(axis=0) + 1e-5)
        recall = np.diag(self.hist) / (self.hist.sum(axis=1) + 1e-5)

        return acc, acc_cls, iu, mean_iu, fwavacc, prec, recall

    def init_hist(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))




class Options:
    def __init__(self):
        self.device = 'cuda:0'
        self.model_names = ["0_DTD-CD"]
        self.model_name = self.model_names[0]
        self.model_name = self.model_name.split('_')[-1]
        print('Training model: ' + self.model_name)
        self.lr = 3e-4
        self.bottom_lr = 1e-5


        self.start_epoch = 0
        self.epochs = 10
        self.batch_size = 12
        self.numw = 8
        
        # Quick validation switch - when set to True, only run 10 batches for quick testing
        self.quick_test = False
        self.quick_test_batches = 10  # Number of batches for quick testing

        self.train_data_path = "data/DocTamperV1-TrainingSet/"
        self.test_data_path = 'data/DocTamperV1-TestingSet/'

        self.model_version_name = 'cddtd' + self.device
        self.checkpoint = Path(os.path.join('checkpoint/upload_temp', self.model_version_name))
        # Create folder if it does not exist
        self.checkpoint.mkdir(parents=True, exist_ok=True)

        # Whether to load weights
        self.load_model = False
        self.load_model_path = 'checkpoint/DTDbase/DTDBase-7.pth'


def eval_net_dtd(opt, model, test_loader1, plot=False):
    model.eval()
    iou = IOUMetric(2)
    precisons = []
    recalls = []
    device = opt.device
    
    # Quick validation mode: limit test batch count
    if opt.quick_test:
        max_test_batches = min(opt.quick_test_batches, len(test_loader1))
        print(f"Quick validation mode: test limited to {max_test_batches} batches")
    else:
        max_test_batches = len(test_loader1)
    
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(test_loader1)):
            # Quick validation mode: exit after reaching limit
            if opt.quick_test and batch_idx >= max_test_batches:
                break
            data, target, dct_coef, qs, q = batch_samples['image'], batch_samples['label'], batch_samples['rgb'], \
                                            batch_samples['q'], batch_samples['i']
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(
                dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))

            if opt.model_name == 'DTD-CD':
                pred, features_ave, bc_median = model(data, dct_coef, qs)
            elif opt.model_name == 'DTD':
                pred = model(data, dct_coef, qs)
            else:
                pred = model(data)

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

        # iu, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
        acc, acc_cls, iu, mean_iu, fwavacc, prec, recall = iou.evaluate()
        precisons = np.array(precisons).mean()
        recalls = np.array(recalls).mean()
        print('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu, precisons, recalls,
                                                        (2 * precisons * recalls / (precisons + recalls + 1e-8))))

        return {'iou': iu, 'pre': precisons, 'rec': recalls,
                'f1': (2 * precisons * recalls / (precisons + recalls + 1e-8))}


print('Model loading phase')
'''
Model loading phase
'''

if __name__ == '__main__':
    opt = Options()

    plot = True
    loss_forgery = 1
    loss_bc = 1
    loss_cor = 1
    mse_loss = nn.MSELoss()

    device = opt.device

    if opt.model_name == 'DTD':
        model = seg_dtd('', 2)
    elif opt.model_name == 'DTD-CD':
        model = seg_dtd_clrdis2('', 2)
    else:
        pass

    # Whether to load existing network weights
    if opt.load_model:
        model_path = opt.load_model_path
        ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])

    train_data1, train_dataset_name = get_dataset_by_choice(0, iter=600000)
    train_loader1 = iter(DataLoader(dataset=train_data1, batch_size=opt.batch_size, num_workers=opt.numw))
    print("Current test dataset being computed")
    print(train_dataset_name)
    print("Current method being computed")
    print(opt.model_name)

    test_data1, test_dataset_name = get_dataset_by_choice(1, data_minq=75)
    test_loader1 = iter(DataLoader(dataset=test_data1, batch_size=opt.batch_size, num_workers=opt.numw))

    print("test_loader1")
    print(len(test_loader1))



    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    train_data_size = train_data1.__len__()
    iter_per_epoch = len(train_loader1)

    test_data_size = test_data1.__len__()
    iter_per_epoch = len(test_loader1)
    print('Training dataset size: ', train_data_size, 'Test dataset size: ', test_data_size)

    totalstep = opt.epochs * iter_per_epoch
    warmupr = 1 / opt.epochs
    warmstep = 200
    lr_min = 1e-5
    lr_min /= opt.lr
    lr_dict = {i: (
        (((1 + math.cos((i - warmstep) * math.pi / (totalstep - warmstep))) / 2) + lr_min) if (i > warmstep) else (
                i / warmstep + lr_min)) for i in range(totalstep)}
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_dict[epoch])

    LovaszLoss_fn = LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
    scaler = GradScaler()
    model.to(device)

    model.train()  # Set the model to training mode
    
    # Quick validation mode prompt
    if opt.quick_test:
        print(f"Quick validation mode: will run {opt.quick_test_batches} batches")
    else:
        print(f"Normal training mode: will run {opt.epochs} epochs")
    
    for epoch in range(opt.start_epoch, opt.epochs):
        total_loss = 0
        avg_loss = 0
        tmp_i = epoch * train_data_size
        iter_i = epoch * iter_per_epoch
        if (epoch != 0):
            train_data1 = TamperDataset([opt.train_data_path], False, tmp_i)
            train_loader1 = iter(DataLoader(dataset=train_data1, batch_size=opt.batch_size, num_workers=opt.numw))

        train_nums = [0] * len(train_loader1)
        random.shuffle(train_nums)
        train_loader_size = len_train = len(train_nums)
        
        # Quick validation mode: limit batch count
        if opt.quick_test:
            max_batches = min(opt.quick_test_batches, len(train_nums))
            print(f"Quick validation mode: limited to {max_batches} batches")
        else:
            max_batches = len(train_nums)
            
        for batch_idx in tqdm(range(max_batches)):
            this_train_id = train_nums[batch_idx]
            if this_train_id == 0:
                batch_samples = next(train_loader1)
            data, target, catnetinput, qs, q, bc = batch_samples['image'], batch_samples['label'], batch_samples['rgb'], \
                                                   batch_samples['q'], batch_samples['i'], batch_samples['bc']
            data, target, catnetinput, qs, bc = Variable(data.to(device)), Variable(target.to(device)), Variable(
                catnetinput.to(device)), Variable(qs.unsqueeze(1).to(device)), Variable(bc.to(device))
            with autocast():

                if opt.model_name == 'DTD':
                    pred = model(data, catnetinput, qs)
                elif opt.model_name == 'DTD-CD':
                    pred, features_ave, bc_median = model(data, catnetinput, qs)
                else:
                    pred = model(data)

                LovaszLoss = LovaszLoss_fn(pred, target)
                BCELoss = SoftCrossEntropy_fn(pred, target)

                # loss = 1. * LovaszLoss_fn(pred, target) + SoftCrossEntropy_fn(pred, target)
                # loss = 5. * LovaszLoss_fn(pred, target) + SoftCrossEntropy_fn(pred, target)

                if opt.model_name == 'DTD-CD':
                    bc_loss1 = mse_loss(bc_median[0].squeeze().to(torch.float32), bc)
                    bc_loss2 = mse_loss(bc_median[1].squeeze().to(torch.float32), bc)
                    pearson_loss_1 = pearson_coefficient(features_ave[0], features_ave[1])
                    pearson_loss_2 = pearson_coefficient(features_ave[2], features_ave[3])
                    # Loss balancing strategy
                    loss = ((1. * LovaszLoss + BCELoss) / (0.114 + 0.216)) * loss_forgery + \
                           (bc_loss1 / 0.016 + bc_loss2 / 0.017) * loss_bc + \
                           (pearson_loss_1 / 0.439 + pearson_loss_2 / 0.290) * loss_cor
                else:
                    loss = 1. * LovaszLoss + BCELoss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # break

            scheduler.step(iter_i + batch_idx)
            total_loss += loss.item()
            if batch_idx % 500 == 0:
                print(f"Batch {batch_idx} loss: {loss.item()}")

        avg_loss = total_loss / len(train_loader1)
        print(f"Average loss: {avg_loss}")


        eval_result_dict = eval_net_dtd(opt, model, test_loader1)

        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        iou_value = eval_result_dict['iou'][1]

        filename = os.path.join(opt.checkpoint, 'DTDBase-{epoch}-iou{iou:.4f}.pth'.format(epoch=epoch, iou=iou_value))
        torch.save(state, filename)
        
        # Quick validation mode: automatically exit after specified batch count in first epoch
        if opt.quick_test and epoch == 0:
            print(f"\nQuick validation completed! Ran {opt.quick_test_batches} batches")
            print(f"Model saved to: {filename}")
            print("Quick validation mode ended, program exiting.")
            break
