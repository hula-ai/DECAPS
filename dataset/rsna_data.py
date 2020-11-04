from torch.utils.data import Dataset
import csv
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import pydicom


class RSNADataSet(Dataset):
    def __init__(self, mode, input_size=(448, 448), data_len=None):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        root = '/home/cougarnet.uh.edu/amobiny/Desktop/miccai_capsnet/dataset/rsna'
        if mode == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.input_size = input_size
        if self.is_train:
            image_list_file = os.path.join(root, 'train.npz')
        else:
            image_list_file = os.path.join(root, 'test.npz')

        image_names = np.load(image_list_file)['data'][:, -1]
        labels = np.load(image_list_file)['data'][:, 5].astype('int64')

        if data_len is not None:
            self.image_names = image_names[:data_len]
            self.labels = np.array(labels)[:data_len]

        else:
            self.image_names = image_names
            self.labels = np.array(labels)

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        if os.path.basename(image_name).endswith('jpg'):
            img = Image.open(image_name).convert('RGB')
        else:
            img = pydicom.read_file(image_name).pixel_array
            img = Image.fromarray(img).convert('RGB')
        target = self.labels[index]

        if self.is_train:
            img = transforms.Resize((500, 500), Image.BILINEAR)(img)
            img = transforms.RandomCrop(self.input_size)(img)
            # img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
            # img = transforms.CenterCrop(options.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        return len(self.image_names)
