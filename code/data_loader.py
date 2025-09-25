'''
包装数据集提取
'''

import json
import os
import random
import cv2
import lmdb
import tempfile
import pickle
import numpy as np
import six
import torch
import torchvision
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm


class TamperDataset(Dataset):
    def __init__(self, roots, mode, i0, steps=8192, pilt=False, casia=False, ranger=1, max_nums=None,
                 max_readers=64, data_process='ori'):
        self.cnts = []
        self.lens = []
        self.envs = []
        self.local_path = "/home/zhengshiqiang/code/DTD_LP/code/data"
        for root in roots:
            if '$' in root:
                root_use, nums = root.split('$')
                nums = int(nums)
                self.envs.append(
                    lmdb.open(root_use, max_readers=max_readers, readonly=True, lock=False, readahead=False,
                              meminit=False))
                with self.envs[-1].begin(write=False) as txn:
                    str = 'num-samples'.encode('utf-8')
                    nSamples = int(txn.get(str))
                    if not (max_nums is None):
                        nSamples = min(nSamples, max_nums)
                self.lens.append(nSamples * nums)
                self.cnts.append(nSamples)
            else:

                print(root)
                self.envs.append(
                    lmdb.open(root, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False))
                with self.envs[-1].begin(write=False) as txn:
                    str = 'num-samples'.encode('utf-8')
                    nSamples = int(txn.get(str))
                    if not (max_nums is None):
                        nSamples = min(nSamples, max_nums)
                self.lens.append(nSamples)
                self.cnts.append(nSamples)

        with open(r'/pubdata/zhengshiqiang/code/DTDMyself_copy/code/tool/DTD_MedianColor.json', 'r') as f:
            self.bc_list = json.load(f)

        self.lens = np.array(self.lens)
        self.sums = np.cumsum(self.lens)
        self.len_sum = len(self.sums)
        self.nSamples = self.lens.sum()
        self.i0 = i0
        self.steps = steps
        self.mode = mode
        with open(os.path.join(self.local_path, r'qt_list/qt_table.pk'), 'rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k, v in pks.items():
            self.pks[k] = torch.Tensor(v)
        npr = np.arange(self.nSamples)

        np.random.seed(i0)
        self.idxs = np.random.choice(self.nSamples, self.nSamples, replace=False)

        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                                                                                       std=(0.229, 0.224, 0.225))])

        self.data_process = data_process

    def calnum(self, num):
        if num < self.lens[0]:
            return 0, num % (self.cnts[0])
        else:
            for li, l in enumerate(self.sums):
                if ((l <= num) and ((li == self.len_sum) or (num < self.sums[li + 1]))):
                    return (li + 1), ((num - l) % (self.cnts[li + 1]))

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        itm_num = self.idxs[idx]
        env_num, index = self.calnum(itm_num)
        with self.envs[env_num].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0) != 0).astype(np.uint8)
            H, W = mask.shape
            if ((H != 512) or (W != 512)):
                return self.__getitem__(random.randint(0, self.nSamples - 1))
            if ((idx + self.i0) < 600000):
                q = random.randint(100 - np.clip((idx + self.i0) * random.uniform(0, 1) // self.steps, 0, 25), 100)
                q2 = random.randint(100 - np.clip((idx + self.i0) * random.uniform(0, 1) // self.steps, 0, 25), 100)
                q3 = random.randint(100 - np.clip((idx + self.i0) * random.uniform(0, 1) // self.steps, 0, 25), 100)
            else:
                q = random.randint(75, 100)
                q2 = random.randint(75, 100)
                q3 = random.randint(75, 100)
            use_qtb = self.pks[q]
            if random.uniform(0, 1) < 0.5:
                im = im.rotate(90)
                mask = np.rot90(mask, 1)
            mask = self.totsr(image=mask.copy())['image']
            if random.uniform(0, 1) < 0.5:
                im = self.hflip(im)
                mask = self.hflip(mask)
            if random.uniform(0, 1) < 0.5:
                im = self.vflip(im)
                mask = self.vflip(mask)

            with tempfile.NamedTemporaryFile(delete=True, prefix=str(idx)) as tmp:
                choicei = random.randint(0, 2)
                if choicei > 1:
                    im.save(tmp, "JPEG", quality=q3)
                    im = Image.open(tmp)
                if choicei > 0:
                    im.save(tmp, "JPEG", quality=q2)
                    im = Image.open(tmp)
                im.save(tmp, "JPEG", quality=q)

                im = Image.open(tmp)

                if self.data_process == 'gray':
                    im = im.convert('L')
                    im = im.convert('RGB')

                if self.data_process == 'hsv':
                    im = np.array(im)
                    hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

                    h, s, v = cv2.split(hsv_im)
                    h = h + random.randint(-8, 8)
                    h = np.mod(h, 360)

                    h = h.astype(np.uint8)
                    s = s.astype(np.uint8)
                    v = v.astype(np.uint8)
                    hsv_im = cv2.merge((h, s, v))

                    im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2RGB)
                    im = im.astype(np.uint8)
                    im = Image.fromarray(im)

                gray_im = im.convert("L")
                gray_im = np.array(gray_im, dtype=np.float32)
                height, width = gray_im.shape[:2]
                blocks = []
                for y in range(0, height, 8):
                    for x in range(0, width, 8):
                        block = gray_im[y:y + 8, x:x + 8]
                        blocks.append(block)

                quantized_blocks = []
                for block in blocks:
                    dct_block = cv2.dct(np.float32(block))
                    quantized_blocks.append(dct_block)

                if self.data_process == "dct_float":
                    quantized_image = np.zeros_like(gray_im).astype(np.float32)
                    for i, block in enumerate(quantized_blocks):
                        y = (i // (width // 8)) * 8
                        x = (i % (width // 8)) * 8
                        quantized_image[y:y + 8, x:x + 8] = block
                    dct = quantized_image.astype(np.float32)
                    im = im.convert('RGB')
                    rgb = np.abs(dct)
                else:
                    quantized_image = np.zeros_like(gray_im).astype(np.int64)
                    for i, block in enumerate(quantized_blocks):
                        y = (i // (width // 8)) * 8
                        x = (i % (width // 8)) * 8
                        quantized_image[y:y + 8, x:x + 8] = block
                    dct = quantized_image.astype(np.int32)
                    im = im.convert('RGB')
                    rgb = np.clip(np.abs(dct), 0, 20)

                bc = self.bc_list[index]
                bc = torch.tensor(bc)
                bc = bc / 255

            return {
                'image': self.toctsr(im),
                'label': mask.long(),
                'rgb': rgb,
                'q': use_qtb,
                'i': q,
                'bc': bc,
                'index': index,
            }


class Eval_TamperDataset(Dataset):
    def __init__(self, roots, mode, minq=90, qtb=90, max_readers=64, data_process='ori'):
        self.envs = lmdb.open(roots, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False)
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums = self.nSamples

        self.minq = minq
        minq = 75

        self.mode = mode
        self.local_path = "/home/zhengshiqiang/code/DTD_LP/code/data"
        with open(os.path.join(self.local_path, r'qt_list/qt_table.pk'), 'rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k, v in pks.items():
            self.pks[k] = torch.LongTensor(v)
        lmdb_name = 'DocTamperV1-TestingSet'

        with open(os.path.join(self.local_path,
                               r'qt_list/pks/' + lmdb_name + '_%d.pk' % minq),
                  'rb') as f:
            self.record = pickle.load(f)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                                                                                       std=(0.229, 0.224, 0.225))])

        self.data_process = data_process

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        with self.envs.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0) != 0).astype(np.uint8)
            H, W = mask.shape
            record = self.record[index]
            choicei = len(record) - 1
            q = int(record[-1])
            use_qtb = self.pks[q]
            if choicei > 1:
                q2 = int(record[-3])
                use_qtb2 = self.pks[q2]
            if choicei > 0:
                q1 = int(record[-2])
                use_qtb1 = self.pks[q1]
            mask = self.totsr(image=mask.copy())['image']

            if self.minq == 100:
                q2 = q1 = q = 100

            with tempfile.NamedTemporaryFile(delete=True, prefix=str(index)) as tmp:
                if choicei > 1:
                    im.save(tmp, "JPEG", quality=q2)
                    im = Image.open(tmp)
                if choicei > 0:
                    im.save(tmp, "JPEG", quality=q1)
                    im = Image.open(tmp)
                im.save(tmp, "JPEG", quality=q)
                im = Image.open(tmp)

                if self.data_process == 'gray':
                    im = im.convert('L')
                    im = im.convert('RGB')

                if self.data_process == 'hsv':
                    im = np.array(im)
                    hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

                    h, s, v = cv2.split(hsv_im)
                    h = h + random.randint(-8, 8)
                    h = np.mod(h, 360)

                    h = h.astype(np.uint8)
                    s = s.astype(np.uint8)
                    v = v.astype(np.uint8)
                    hsv_im = cv2.merge((h, s, v))

                    im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2RGB)
                    im = im.astype(np.uint8)
                    im = Image.fromarray(im)

                gray_im = im.convert("L")
                gray_im = np.array(gray_im, dtype=np.float32)
                height, width = gray_im.shape[:2]
                blocks = []
                for y in range(0, height, 8):
                    for x in range(0, width, 8):
                        block = gray_im[y:y + 8, x:x + 8]
                        blocks.append(block)

                quantized_blocks = []
                for block in blocks:
                    dct_block = cv2.dct(np.float32(block))
                    quantized_blocks.append(dct_block)

                if self.data_process == "dct_float":
                    quantized_image = np.zeros_like(gray_im).astype(np.float32)
                    for i, block in enumerate(quantized_blocks):
                        y = (i // (width // 8)) * 8
                        x = (i % (width // 8)) * 8
                        quantized_image[y:y + 8, x:x + 8] = block
                    dct = quantized_image.astype(np.float32)
                    im = im.convert('RGB')
                    rgb = np.abs(dct)
                else:
                    quantized_image = np.zeros_like(gray_im).astype(np.int64)
                    for i, block in enumerate(quantized_blocks):
                        y = (i // (width // 8)) * 8
                        x = (i % (width // 8)) * 8
                        quantized_image[y:y + 8, x:x + 8] = block
                    dct = quantized_image.astype(np.int32)
                    im = im.convert('RGB')
                    rgb = np.clip(np.abs(dct), 0, 20)

            return {
                'image': self.toctsr(im),
                'label': mask.long(),
                'rgb': rgb,
                'q': use_qtb,
                'i': q,
                'index': index,
            }


class TamperDataset_ours(Dataset):
    def __init__(self, roots, mode, color_mode="rgb", minq=95, qtb=90, max_readers=64, hue_level=0):
        self.envs = lmdb.open(roots, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False)
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums = self.nSamples
        self.minq = minq
        self.mode = mode
        self.local_path = "/home/zhengshiqiang/code/DTD_LP/code/data"
        with open(os.path.join(self.local_path, 'qt_list/qt_table.pk'), 'rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k, v in pks.items():
            self.pks[k] = torch.LongTensor(v)
        lmdb_name = 'DocTamperV1-TestingSet'
        with open(os.path.join(self.local_path,
                               'qt_list/pks/' + lmdb_name + '_%d.pk' % minq),
                  'rb') as f:
            self.record = pickle.load(f)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()

        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

        self.color_mode = color_mode
        print('颜色模式为')
        print(self.color_mode)

        self.hue_level = hue_level
        print('扰动强度')
        print(self.hue_level)

        if self.color_mode not in ['hue', 'gray', 'rgb']:
            raise ValueError("Invalid color mode. Allowed values are 'hue', 'gray', 'rgb'.")

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        with self.envs.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            image_bytes = txn.get(img_key.encode('utf-8'))
            im = Image.frombytes('RGB', (512, 512), image_bytes)


            lbl_key = 'label-%09d' % index
            label_bytes = txn.get(lbl_key.encode('utf-8'))
            label = Image.frombytes('L', (512, 512), label_bytes)



            mask = np.array(label)
            mask = np.where(mask == 0, 0, 1)
            mask = mask.astype(np.uint8)

            H, W = mask.shape
            mask = self.totsr(image=mask.copy())['image']

            q = 100
            use_qtb = self.pks[q]

            with tempfile.NamedTemporaryFile(delete=True) as tmp:

                im.save(tmp, "JPEG", quality=q)
                im = Image.open(tmp)

                if self.color_mode == 'gray':
                    im = im.convert('L')
                    im = im.convert('RGB')

                if self.color_mode == "hue":
                    im = np.array(im)
                    hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

                    h, s, v = cv2.split(hsv_im)
                    h = h + random.uniform(-3.6 * self.hue_level,
                                           3.6 * self.hue_level)  # Adjust H by a random value between -30 and 30
                    h = np.mod(h, 360)

                    h = h.astype(np.uint8)
                    s = s.astype(np.uint8)
                    v = v.astype(np.uint8)
                    hsv_im = cv2.merge((h, s, v))

                    im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2RGB)
                    im = im.astype(np.uint8)
                    im = Image.fromarray(im)

                gray_im = im.convert("L")
                gray_im = np.array(gray_im, dtype=np.float32)
                height, width = gray_im.shape[:2]
                blocks = []

                for y in range(0, height, 8):
                    for x in range(0, width, 8):
                        block = gray_im[y:y + 8, x:x + 8]
                        blocks.append(block)

                quantized_blocks = []
                for block in blocks:
                    dct_block = cv2.dct(np.float32(block))
                    quantized_blocks.append(dct_block)

                quantized_image = np.zeros_like(gray_im).astype(np.int64)
                for i, block in enumerate(quantized_blocks):
                    y = (i // (width // 8)) * 8
                    x = (i % (width // 8)) * 8
                    quantized_image[y:y + 8, x:x + 8] = block
                dct = quantized_image.astype(np.int32)

                im = im.convert('RGB')

                im = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
                im = self.toctsr(im)
            return {
                'image': im,
                'label': mask.long(),
                'rgb': np.clip(np.abs(dct), 0, 20),
                'q': use_qtb,
                'i': q,
                'index': index,
            }


import random
from torch.utils.data import Dataset


def get_dataset_by_choice(choice, iter=0, data_minq=75, purpose='train'):
    dataset_dict = {
        0: 'DocTamperV1-TrainingSet',
        1: 'DocTamperV1-TestingSet',
        2: 'DocTamperV1-FCD',
        3: 'DocTamperV1-SCD',
        4: 'ZPY-Bookcover-1-4_v2',
        5: 'BK-Certificate',
        6: 'YJ-identification-v5',
    }

    dataset_name = dataset_dict.get(choice)

    if dataset_name:
        print(f"您已选择测试集：{dataset_name}")
    else:
        print("无效的选择，请输入正确的数字（0,1,2,3,4,5,6）。")
        return None

    data_root = '/pubdata/zhengshiqiang/dataset'

    if choice == 0:
        train_data_path = os.path.join(data_root, dataset_name)
        train_data = TamperDataset([train_data_path], False, iter, data_process='ori')
        dataset = train_data

    elif 1 <= choice <= 3:
        test_data = Eval_TamperDataset(
            os.path.join(data_root, dataset_name),
            False,
            minq=data_minq,
            data_process='ori',
        )
        dataset = test_data
    elif choice in [4, 5, 6]:
        test_data = TamperDataset_ours(
            os.path.join(data_root, dataset_name),
            False,
            minq=75
        )
        dataset = test_data

    return dataset, dataset_name


if __name__ == '__main__':
    device = 'cuda:4'

    test_data = get_dataset_by_choice(1)
    test_loader = DataLoader(dataset=test_data, batch_size=4, num_workers=8, shuffle=False)
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(test_loader, mininterval=5)):
            data, target, dct_coef, qs, q, index = batch_samples['image'], batch_samples['label'], batch_samples['rgb'], \
                                                   batch_samples['q'], batch_samples['i'], batch_samples['index']
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(
                dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))

            print(data.size())
            print(target.size())

