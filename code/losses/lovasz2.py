"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

__all__ = ["LovaszLoss"]


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    # 计算gt_sorted的长度
    p = len(gt_sorted)
    # 计算gt_sorted的总和
    gts = torch.sum(gt_sorted.long())
    # gts = gt_sorted.sum()

    # 计算intersection
    intersection = gts - gt_sorted.float().cumsum(0)


    # 计算union
    union = gts + (1 - gt_sorted).float().cumsum(0)


    # 计算jaccard
    jaccard = 1.0 - intersection / union


    # 如果p大于1，则将jaccard的第一个元素减去jaccard的第一个元素
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    # 返回jaccard

    return jaccard


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Logits at each prediction (between -infinity and +infinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------


def _lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore: void class labels
        函数的参数如下：
        probas：预测概率，形状为[B, C, H, W]的张量。
        labels：真实标签，形状为[B, H, W]的张量。
        classes：要计算损失的类别列表。
        per_image：如果为True，则计算每个图像的损失，否则计算整个批次的损失。
        ignore：要忽略的类别列表。
        函数的计算过程如下：

        将预测概率转换为二值掩码。
        计算每个类别在真实标签和预测掩码之间的交并比。
        计算每个类别在真实标签和预测掩码之间的IoU损失。
        将每个类别的IoU损失加权平均，得到最终的损失值。
    """
    # 如果per_image为True，则计算每张图像的Lovasz-Softmax损失
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            for prob, lab in zip(probas, labels)
        )
    # 如果per_image为False，则计算单张图像的Lovasz-Softmax损失
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def _lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    # 如果probas的长度为0，则返回0
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    # 获取probas的类别数
    C = probas.size(1)
    # 初始化losses
    losses = []
    # 获取classes参数

    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes

    # 遍历class_to_sum中的每一个类别
    for c in class_to_sum:
        # 获取fg，以c为索引(该类下，哪些类是1)
        fg = (labels == c).type_as(probas)  # foreground for class c

        # 如果classes参数为present，且fg的数量为0，则跳过本次循环
        if classes == "present" and fg.sum() == 0:
            continue
        # 如果C为1，则classes参数为第一个类别，则使用probas的第一个类别
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            # 获取这个类下的预测概率
            class_pred = probas[:, c]

        # 计算errors
        # 概率离标签距离的矩阵
        errors = (fg - class_pred).abs()

        # 根据errors的排序，获取perm(最大的错误数和对应的标签）
        errors_sorted, perm = torch.sort(errors, 0, descending=True)

        perm = perm.data

        # 获取fg_sorted
        fg_sorted = fg[perm]

        # print(torch.dot(errors_sorted.to(torch.float32), _lovasz_grad(fg_sorted)).to(torch.float32))
        # 计算losses

        # losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
        losses.append(torch.dot(errors_sorted.to(torch.float16), _lovasz_grad(fg_sorted).to(torch.float16)))
    # 返回平均losses
    return mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)

    C = probas.size(1)
    probas = torch.movedim(probas, 0, -1)  # [B, C, Di, Dj, Dk...] -> [B, C, Di...Dk, C]
    probas = probas.contiguous().view(-1, C)  # [P, C]

    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators.
    """
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszLoss(_Loss):
    # 定义Lovasz损失函数
    def __init__(
            self,
            mode: str,
            per_image: bool = False,
            ignore_index: Optional[int] = None,
            from_logits: bool = True,
    ):
        """Implementation of Lovasz loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary','multiclass' or'multilabel'
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            per_image: If True loss computed per each image and then averaged, else computed per whole batch

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        图像分割任务中Lovasz损失函数的实现。
        支持二分类、多分类和多标签的情况。

        参数：
            mode：损失模式，'binary'、'multiclass'或'multilabel'
            ignore_index：表示被忽略像素的标签（不参与损失计算）
            per_image：如果为True，则每个图像计算损失并求平均；如果为False，则整个批次计算损失

        形状：
            - **y_pred** - torch.Tensor，形状为(N, C, H, W)
            - **y_true** - torch.Tensor，形状为(N, H, W)或(N, C, H, W)

        参考：
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        # 根据模式判断损失函数的模式
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        # 设置损失函数的模式
        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, y_pred, y_true):

        # 如果模式为二分类或多分类，则使用Lovasz-Hinge损失函数
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            loss = _lovasz_hinge(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        # 如果模式为多分类，则使用Lovasz-Softmax损失函数
        elif self.mode == MULTICLASS_MODE:
            # 计算梯度

            y_pred = y_pred.softmax(dim=1)

            loss = _lovasz_softmax(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        else:
            # 如果模式不是BINARY_MODE，则抛出异常
            raise ValueError("Wrong mode {}.".format(self.mode))
        # 返回损失
        return loss
