# Implemented by Dingquan Li
# Email: dingquanli@pku.edu.cn
# Date: 2018/4/19

import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import h5py


def default_loader(path):
    return Image.open(path).convert('L') #


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


def NonOverlappingCropPatches(im, patch_size=32, stride=32):
    w, h = im.size
    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patch = LocalNormalization(patch[0].numpy())
            patches = patches + (patch,)
    return patches


class IQADataset(Dataset):
    def __init__(self, conf, exp_id=0, status='train', loader=default_loader):
        self.loader = loader
        im_dir = conf['im_dir']
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        datainfo = conf['datainfo']

        Info = h5py.File(datainfo, 'r')
        index = Info['index'][:, int(exp_id) % 1000]
        ref_ids = Info['ref_ids'][0, :]
        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1-test_ratio) * len(index)):]
        train_index, val_index, test_index = [],[],[]
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        self.mos = Info['subjective_scores'][0, self.index]
        self.mos_std = Info['subjective_scoresSTD'][0, self.index]
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()\
                        [::2].decode() for i in self.index]
        
        self.patches = ()
        self.label = []
        self.label_std = []
        for idx in range(len(self.index)):
            # print("Preprocessing Image: {}".format(im_names[idx]))
            im = self.loader(os.path.join(im_dir, im_names[idx]))

            patches = NonOverlappingCropPatches(im, self.patch_size, self.stride)
            if status == 'train':
                self.patches = self.patches + patches #
                for i in range(len(patches)):
                    self.label.append(self.mos[idx])
                    self.label_std.append(self.mos_std[idx])
            else:
                self.patches = self.patches + (torch.stack(patches), ) #
                self.label.append(self.mos[idx])
                self.label_std.append(self.mos_std[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], (torch.Tensor([self.label[idx]]), torch.Tensor([self.label_std[idx]]))
