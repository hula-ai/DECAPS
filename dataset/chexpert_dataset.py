from torch.utils.data import Dataset
import csv
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np


class CheXpertDataSet(Dataset):
    def __init__(self, mode, input_size=(448, 448), policy="ones", data_len=None):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        image_names = []
        root = '/home/cougarnet.uh.edu/amobiny/Desktop/CheXpert-v1.0-small'
        labels = []
        if mode == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.input_size = input_size
        if self.is_train:
            image_list_file = os.path.join(root, 'train.csv')
        else:
            image_list_file = os.path.join(root, 'valid.csv')

        with open(image_list_file, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            k = 0
            for line in csvReader:
                k += 1
                image_name = line[0]
                label = line[5:]

                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0

                image_names.append(os.path.join(os.path.dirname(root), image_name))
                labels.append(label)

        if data_len is not None:
            self.image_names = image_names[:data_len]
            self.labels = np.array(labels)[:, 4][:data_len]  # [2, 5, 6, 8, 10]

            # set no findings
            # col_sum = np.sum(self.labels[:, 1:], 1)
            # no_find_idx = np.where(col_sum == 0)[0]
            # self.labels[no_find_idx, 0] = 1

        else:
            self.image_names = image_names
            self.labels = np.array(labels)[:, 4]   # Opacity: 3

            # set no findings
            # col_sum = np.sum(self.labels[:, 1:], 1)
            # no_find_idx = np.where(col_sum == 0)[0]
            # self.labels[no_find_idx, 0] = 1

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        img = Image.open(image_name).convert('RGB')
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
