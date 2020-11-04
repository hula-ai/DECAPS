from torch.utils.data import Dataset
import csv
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import pydicom
from config import options


def fix_name_with_split(s, c, n):
    words = s.split(c)
    path_ = os.path.join(c.join(words[:n]), c.join(words[n:]))
    return path_


class DeepLesion(Dataset):
    def __init__(self, mode, input_size=(options.img_h, options.img_w), data_len=None):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        root = '/home/cougarnet.uh.edu/amobiny/Desktop/miccai_capsnet/dataset/deep_lesion'
        self.img_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/DeepLesion/deep_lesion'
        self.min_hu = -1024
        self.max_hu = 3071
        if mode == 'train':
            self.is_train = True
            image_list_file = os.path.join(root, 'deeplesion_train.npz')
        else:
            self.is_train = False
            image_list_file = os.path.join(root, 'deeplesion_test.npz')

        self.input_size = input_size
        image_names = np.load(image_list_file)['x']
        labels = np.load(image_list_file)['y'].astype('int64') - 1
        self.bbox = np.load(image_list_file)['bbox']

        if data_len is not None:
            self.image_names = image_names[:data_len]
            self.labels = np.array(labels)[:data_len]
        else:
            self.image_names = image_names
            self.labels = np.array(labels)

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = os.path.join(self.img_dir, fix_name_with_split(self.image_names[index], '_', 3))
        img = np.array(Image.open(image_name)) - 32768  # load and convert to HU
        # now the desired information is in range [-1024, 3071]
        img = np.clip(img, self.min_hu, self.max_hu)
        # now let's normalize it to be in range [0, 255]
        img = 255. * ((img - self.min_hu) / (self.max_hu - self.min_hu))
        # now convert to an RGB PIL image
        img = Image.fromarray(img).convert('RGB')
        target = self.labels[index]

        if self.is_train:
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        else:
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
            # img = transforms.CenterCrop(options.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target, self.bbox[index].reshape(-1)

    def __len__(self):
        return len(self.image_names)
