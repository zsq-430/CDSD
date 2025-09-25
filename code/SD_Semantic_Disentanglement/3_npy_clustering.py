import os
import random
from copy import copy

import cv2
import hdbscan
import numpy as np

from scipy.ndimage import binary_erosion, label, binary_dilation

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm




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


def get_cluster_pixels(image, labeled_image, cluster_label):
    cluster_pixels = image[labeled_image == cluster_label]
    cluster_image = copy(image)
    cluster_mask = labeled_image == cluster_label
    cluster_image[~cluster_mask] = 0
    print('cluster_pixels')
    print(cluster_pixels.shape)
    print(np.mean(cluster_pixels))
    print(np.median(cluster_pixels))
    plt.imshow(cluster_image, cmap='nipy_spectral')
    plt.axis('off')
    plt.show()


def label_and_number(image):
    labeled_image, num_labels = label(image)

    colored_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)

    for label_value in range(1, num_labels + 1):
        label_indices = np.where(labeled_image == label_value)

        color = np.random.randint(0, 256, size=3, dtype=np.uint8)

        colored_image[label_indices[0], label_indices[1]] = color

        label_position = (int(np.mean(label_indices[1])), int(np.mean(label_indices[0])))
        label_text = str(label_value)
        plt.text(label_position[0], label_position[1], label_text, color='white', fontsize=8, ha='center')
    plt.imshow(colored_image, cmap='nipy_spectral')
    plt.axis('off')
    plt.show()

    return colored_image


def top_k_mean(values, k):
    sorted_values = np.sort(values)[::-1]
    k_count = int(k * len(values))

    top_k_values = sorted_values[:k_count]
    return np.median(top_k_values)


folder_path = r'F:\code\npy-output\182-DTD_result\npy_output\DTD_182-BK-Certificate-npy-output'


target_dir = r'D:\dataset\bk\gt\img'




filenames = []
filenames2 = []



file_list = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.basename(file_path) not in filenames and os.path.basename(file_path) not in filenames2:
            file_list.append(file_path)



iou = IOUMetric(2)
iou.init_hist()
precisons = []
recalls = []
result_dict = []

iou_filelist = {}
for file_path in tqdm(file_list):
    pred = np.load(file_path)

    target_path = os.path.join(target_dir, os.path.basename(file_path).replace('.tif.npy', '_gt.jpg'))




    print(target_path)

    target = cv2.imread(target_path, 0)
    target = (target > 128).astype(np.uint8)



    height, width = target.shape[:2]
    pred = pred[:height, :width]
    downsampled_image = pred / 255

    filtered_image_array = np.where(downsampled_image < 0.1, 0, downsampled_image)

    copy_filtered_image_array = copy(filtered_image_array)
    copy_filtered_image_array[copy_filtered_image_array > 0] = 1
    copy_filtered_image_array = copy_filtered_image_array.astype(np.uint8)

    dilated_image = binary_dilation(copy_filtered_image_array, structure=np.ones((5, 7)), iterations=7)
    eroded_image = binary_erosion(dilated_image, structure=np.ones((5, 7)), iterations=7)

    labeled_mask, num_labels = label(eroded_image)

    component_sizes = np.bincount(labeled_mask.flatten())

    min_area = int(pred.shape[0] * pred.shape[1] * 6e-5)
    max_area = int(pred.shape[0] * pred.shape[1] * 0.005)

    filtered_image = copy(eroded_image)

    label_mapping = {}

    for label_id, size in enumerate(component_sizes):
        if size <= min_area and label_id != 0:
            filtered_image[labeled_mask == label_id] = 0
        elif size >= max_area and label_id != 0:
            filtered_image[labeled_mask == label_id] = 0

    labeled_mask, num_labels = label(filtered_image)
    component_sizes = np.bincount(labeled_mask.flatten())

    cluster_means = []

    for label_id in range(1, num_labels + 1):
        cluster_pixels = downsampled_image[labeled_mask == label_id]
        non_zero_pixels = cluster_pixels[cluster_pixels != 0]
        cluster_mean = top_k_mean(cluster_pixels, 0.3)
        cluster_means.append(cluster_mean)

    if len(cluster_means) > 2:
        n_clusters = 2

        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=n_clusters, gen_min_span_tree=True)
        hdbscan_clusterer.fit(np.array(cluster_means).reshape(-1, 1))
        cluster_labels = hdbscan_clusterer.labels_
        cluster_centers = hdbscan_clusterer.probabilities_
        n_clusters = hdbscan_clusterer.labels_.max()

        cluster_labels = np.insert(cluster_labels, 0, -1)



        sorted_clusters = sorted(enumerate(cluster_means), key=lambda x: x[1], reverse=True)
        sorted_values = [x[1] for x in sorted_clusters]
        sorted_indices = [x[0] for x in sorted_clusters]
        result_index = 0
        outliner = []

        for i in range(n_clusters):
            if cluster_labels[sorted_indices[i] + 1] != -1:
                result_index = sorted_indices[i]
                break
            else:
                outliner.append(sorted_indices[i])

        max_center_index = cluster_labels[result_index + 1]
        for outliner_ in outliner:
            cluster_labels[outliner_ + 1] = max_center_index

        result_image = copy(filtered_image)
        for label_id, size in enumerate(component_sizes):
            if label_id not in np.where(cluster_labels == max_center_index)[0] and label_id != 0:
                result_image[labeled_mask == label_id] = 0



    else:
        threshold = 0.5
        result_image = filtered_image
        result_image = np.where(result_image >= threshold, 1, 0)

    result_image = result_image.astype(int)

    result_image = np.expand_dims(result_image, axis=0)
    target = np.expand_dims(target, axis=0)

    threshold_downsampled_image = np.where(downsampled_image < 0.5, 0, downsampled_image)

    show_output_path = r'E:\data_vision\SVD-autok-0.3loss\YJ-identification-v5-npy-output2'
    if not os.path.exists(show_output_path):
        os.makedirs(show_output_path)

    base_filename = os.path.basename(file_path)
    clusting_path = os.path.join(show_output_path, f'{base_filename}_clusting.png')
    target_path = os.path.join(show_output_path, f'{base_filename}_target.png')

    cv2.imwrite(clusting_path, result_image[0] * 255)
    cv2.imwrite(target_path, target[0] * 255)
    prediction_path = os.path.join(show_output_path, f'{base_filename}_pre.png')
    cv2.imwrite(prediction_path, (threshold_downsampled_image* 255).astype('uint8'))

    print(result_image.shape)
    print(target.shape)

    matched = (result_image * target).sum((1, 2))
    pred_sum = result_image.sum((1, 2))
    target_sum = target.sum((1, 2))

    precisons.append((matched / (pred_sum + 1e-8)).mean().item())
    recalls.append((matched / target_sum + 1e-8).mean().item())
    iou_tamper = (matched / (pred_sum + target_sum - matched + 1e-8)).mean().item()
    iou_filelist[file_path] = iou_tamper
    iou.add_batch(result_image, target)

acc, acc_cls, iu, mean_iu, fwavacc, prec, recall = iou.evaluate()
precisons_mean = np.array(precisons).mean()
recalls_mean = np.array(recalls).mean()

print('[val] iou:{} pre:{} rec:{} f1:{} '.format(iu, precisons_mean, recalls_mean,
                                                 (2 * precisons_mean * recalls_mean / (
                                                         precisons_mean + recalls_mean + 1e-8)), ))
print('new_pre:{},new_recall:{}'.format(prec, recall))
