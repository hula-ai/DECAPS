from torch.utils.data import Dataset
import csv
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import h5py


class MIASDataSet(Dataset):
    def __init__(self, mode, input_size=(448, 448), data_len=None):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        root = '/home/cougarnet.uh.edu/amobiny/Desktop/mias'
        if mode == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.input_size = input_size
        if self.is_train:
            file_path = os.path.join(root, 'mias_train.h5')
        else:
            file_path = os.path.join(root, 'mias_test.h5')

        h5f = h5py.File(file_path, 'r')
        images = h5f['x'][:]
        labels = h5f['y'][:].astype('int64')
        h5f.close()

        if data_len is not None:
            self.images = images[:data_len]
            self.labels = np.array(labels)[:data_len]

        else:
            self.images = images
            self.labels = np.array(labels)

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image = self.images[index]
        img = Image.fromarray(image).convert('RGB')
        target = self.labels[index]

        if self.is_train:
            img = transforms.Resize((500, 500), Image.BILINEAR)(img)
            img = transforms.RandomCrop(self.input_size)(img)
            # img = transforms.RandomHorizontalFlip()(img)
            # img = transforms.RandomRotation(30)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
            # img = transforms.CenterCrop(options.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        return len(self.images)
