"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division
from typing import Optional

import torch
import torch.nn.functional as F
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
    p = len(gt_sorted)
    # gts = gt_sorted.sum()
    gts = torch.sum(gt_sorted)
    # 计算并集除交集
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    # 计算降低速度
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    # print("jaccard")
    #
    # print(jaccard)
    # print(torch.max(jaccard), torch.min(jaccard), jaccard.size())
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


# 定义一个函数，用于将二进制分数进行扁平化
def _flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    # 将scores和labels的形状转换为（-1）
    scores = scores.view(-1)
    labels = labels.view(-1)
    # 如果ignore为None，则返回scores和labels
    if ignore is None:
        return scores, labels
    # 找到labels不等于ignore的索引
    valid = labels != ignore
    # 将valid的索引对应的scores和labels赋值给vscores和vlabels
    vscores = scores[valid]
    vlabels = labels[valid]
    # 返回vscores和vlabels

    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------


def _flatten_probas_in(probas, labels, ignore=None):
    """Flattens predictions in the batch
    """


    # print("probas.size()")
    # print(probas[0,:,:10,:10])
    # print(probas[1,:,:10,:10])
    # print("labels.size()")
    # print(labels[0])

    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)

    C = probas.size(1)
    # probas = torch.movedim(probas, 0, -1)  # [B, C, Di, Dj, Dk...] -> [B, C, Di...Dk, C]
    probas = probas.permute(0,2,3,1)
    probas = probas.contiguous().view(-1, C)  # [P, C]

    labels = labels.view(-1)
    # print("probas.size()")
    # print(probas[:10])
    # print("labels.size()")
    # print(labels[:10])

    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]

    return vprobas, vlabels

def _lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore: void class labels
    """
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(*_flatten_probas_in(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas_in(probas, labels, ignore), classes=classes)
    return loss


# 定义Lovasz-Softmax损失函数，用于多分类问题
# Args:
#     @param probas: [P, C] 预测类别概率（0到1之间）
#     @param labels: [P] 标签（0到C-1之间）
#     @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#
def _lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """

    # print("probas[0]")
    # print(probas.size())
    # print("labels[0]")
    # print(labels.size())
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    # 计算类别数量
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    # 记录每个类别的loss
    for c in class_to_sum:
        # 计算每个类别的fg，也就是某一类别正确的个数
        fg = (labels == c).type_as(probas)  # foreground for class c
        # 如果是该类下没有正类，则不影响
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            # 读取该类下的所有概率
            class_pred = probas[:, c]
        # 该类下，概率与正确端点的距离
        errors = (fg - class_pred).abs()
        # 按照错误值排序
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]

        # 计算Lovasz-Softmax损失
        # 点乘后相加，是一个数
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))

    return mean(losses)





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
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            per_image: If True loss computed per each image and then averaged, else computed per whole batch

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, y_pred, y_true):

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            loss = _lovasz_hinge(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        elif self.mode == MULTICLASS_MODE:
            y_pred = y_pred.softmax(dim=1)

            loss = _lovasz_softmax(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        else:
            raise ValueError("Wrong mode {}.".format(self.mode))
        return loss
