import os
import cv2
import lmdb
import torch
import numpy as np
import torch.nn as nn
import pickle
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader


from data_loader import get_dataset_by_choice
from dtd import *
from albumentations.pytorch import ToTensorV2
import torchvision
import argparse
import tempfile







class TamperDataset(Dataset):
    def __init__(self, roots, mode, minq=95, qtb=90, max_readers=64):
        self.envs = lmdb.open(roots, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False)
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums = self.nSamples
        self.minq = minq
        self.mode = mode
        with open('code_tool/qt_table.pk', 'rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k, v in pks.items():
            self.pks[k] = torch.LongTensor(v)
        lmdb_name = 'DocTamperV1-TestingSet'
        with open(
                'code_tool/pks/' + lmdb_name + '_%d.pk' % minq,
                'rb') as f:
            self.record = pickle.load(f)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        #  Note: DTD-cls uses 0.5, 0.5, 0.5
        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                             std=(0.5, 0.5, 0.5))])

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        with self.envs.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            image_bytes = txn.get(img_key.encode('utf-8'))
            # Convert byte array to PIL image object
            im = Image.frombytes('RGB', (512, 512), image_bytes)
            # Read label
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


                gray_im = im.convert("L")
                # DCT
                gray_im = np.array(gray_im, dtype=np.float32)
                height, width = gray_im.shape[:2]
                blocks = []
                for y in range(0, height, 8):
                    for x in range(0, width, 8):
                        block = gray_im[y:y + 8, x:x + 8]
                        blocks.append(block)

                # DCT quantization
                quantized_blocks = []
                for block in blocks:
                    dct_block = cv2.dct(np.float32(block))
                    # Quantization
                    quantized_blocks.append(dct_block)

                # Reconstruct image
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

    def init_hist(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))

def parse_and_assign_variables(combination_string):
    # Split string into components
    components = combination_string.split(', ')
    settings = {}

    # Predefine which keys need to be converted to integers
    int_keys = ['hue_level']

    for component in components:
        key, value = component.split('=')
        # Check if key is in the list that needs to be converted to integer
        if key in int_keys:
            settings[key] = int(value)  # Convert to integer
        else:
            settings[key] = value

    # Set global variables as needed, here is an example of how to set them
    global color_mode, hue_level, model_name
    # lmdb_name = settings['lmdb_name']
    color_mode = settings['color_mode']
    hue_level = settings['hue_level']
    model_name = settings['model_name']

    return settings

def select_lmdb(choice):
    lmdb_dict = {
        1: 'ZPY-Bookcover-1-4_v2', # ATBC
        2: 'BK-Certificate', # PSC
    }
    lmdb_name = lmdb_dict.get(choice)

    # Check if it is a valid option
    if lmdb_name:
        print(f"You have selected {lmdb_name}")
        return lmdb_name
    else:
        print("Invalid choice, please enter a correct number.")
        return None





def eval_net_dtd(device='cuda:0'):

    model_name = 'DTD-CD'


    # Example usage
    combo_string = "model_name=DTD-CD, color_mode=rgb, hue_level=1"
    parsed_settings = parse_and_assign_variables(combo_string)

    print(f"lmdb_name={lmdb_name}, color_mode={color_mode}, hue_level={hue_level}, model_name={model_name}")



    if model_name == 'DTD-CD':
        model = seg_dtd_clrdis2('', 2)


    test_data, test_dataset_name = get_dataset_by_choice(4)
    train_loader1 = DataLoader(dataset=test_data, batch_size=6, num_workers=8, shuffle=False)



    model_pth = r'checkpoint/upload_temp/10-DTD-clrdis-iou0.8239.pth'

    ckpt = torch.load(model_pth, map_location='cpu')


    new_state_dict = {}
    for key, value in ckpt['state_dict'].items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value


    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()



    save_path = 'checkpoint/upload_temp'
    os.makedirs(save_path, exist_ok=True)
    npy_savepath = os.path.join(save_path, lmdb_name + '-' + 'npy')
    npy_savepath = npy_savepath.encode('utf-8')
    env = lmdb.open(npy_savepath, map_size=int(2e10))


    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(train_loader1, mininterval=50)):
            data, target, dct_coef, qs, q, index = batch_samples['image'], batch_samples['label'], batch_samples['rgb'], \
                batch_samples['q'], batch_samples['i'], batch_samples['index']

            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(
                dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))

            pred, features_ave, bc_media = model(data, dct_coef, qs)

            prob_map = torch.softmax(pred, dim=1)
            prob_map = prob_map[:, 1, :, :]


            with env.begin(write=True) as txn:
                for i in range(len(index)):
                    img_key = 'image-%09d' % index[i]
                    prob_map_npy = prob_map[i].cpu().data.numpy()
                    prob_map_buf = prob_map_npy.tobytes()
                    txn.put(img_key.encode('utf-8'), prob_map_buf)



if __name__ == "__main__":

    import argparse

    # Create parser
    parser = argparse.ArgumentParser(description='Example program.')

    # Add parameters
    parser.add_argument('choice', type=int, default=4, nargs='?', help='Select which database')
    parser.add_argument('--verbose', action='store_true', help='Enter verbose mode if set')

    # Parse arguments
    args = parser.parse_args()

    choice = args.choice
    # Use parameters
    if args.verbose:
        print("Verbose mode started")
    print(f"Selected dataset parameter is {args.choice}")

    global lmdb_name
    lmdb_name = select_lmdb(choice)

    eval_net_dtd()
