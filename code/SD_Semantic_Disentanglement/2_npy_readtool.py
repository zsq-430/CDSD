import json
import os

import cv2
import lmdb
import numpy as np
import six as six
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm


# 本代码读取eval生成的npy文件，并按照json文件中的文件名，拼回原图


def find_empty_lists(data, filename):
    result = []
    for item in data:
        if item.get('filename') == filename:
            result.append(item)
    return result


# 读取JSON文件
with open('mdb_json/BK-Certificate.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


print(len(data))

filename_list = []
for item in data:
    # 如果存在则不添加
    if item['filename'] in filename_list:
        continue
    else:
        filename_list.append(item['filename'])



envs_numpy = lmdb.open('npy_write\BK-Certificate-npy', max_readers=64, readonly=True, lock=False, readahead=False,
                       meminit=False)





'''
图片版输出
'''
for filename_ in tqdm(filename_list):
    result = find_empty_lists(data, filename_)

    height = result[0]['height']
    width = result[0]['width']
    # height_pad = height - height % 512 + 512
    # width_pad = width - width % 512 + 512

    height_pad = height + 512
    width_pad = width + 512


    # 新建一张这么大的numpy
    # image_sum = np.zeros((height_pad, width_pad, 3), dtype=np.uint8)
    # label_sum = np.zeros((height_pad, width_pad), dtype=np.uint8)
    numpy_sum = np.zeros((height_pad, width_pad), dtype=np.float16)
    for result_ in result:
        index = result_['index']
        # print(index)
        # 读取图片
        with envs_numpy.begin(write=False) as txn:
            # print(index)
            npy_key = 'image-%09d' % index
            # print("Trying key:", npy_key)
            npy_bytes = txn.get(npy_key.encode('utf-8'))

            # print(type(npy_bytes))
            # 读取值（value）并将其转换为 NumPy 数组
            # prob_map_buf = np.frombuffer(npy_bytes, dtype=np.float16)
            prob_map_buf = np.frombuffer(npy_bytes, dtype=np.float32)
            # print(prob_map_buf.shape)
            prob_map = prob_map_buf.reshape(512, 512)

            prob_map = prob_map * 255

            # # 将prob_map用cv保存到指定文件夹
            # cv2.imwrite('npy_write/BK-Certificate-show/' +filename_+'-'+ str(index) + '.png', prob_map)

            i = result_['i']
            j = result_['j']
            start_x = i * 512
            end_x = start_x + 512
            start_y = j * 512
            end_y = start_y + 512


            numpy_sum[start_x:end_x, start_y:end_y] = prob_map



    numpy_sum = numpy_sum[:height, :width]
    # 将prob_map用cv保存到指定文件夹
    # 保存为npy文件
    np_save_dir = r'npy_write\BK-Certificate-npy-output'
    np_save_path = os.path.join(np_save_dir, filename_ + '.npy')
    if not os.path.exists(np_save_dir):
        os.makedirs(np_save_dir)

    np.save(np_save_path, numpy_sum)



