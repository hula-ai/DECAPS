import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
import imageio
from config import options
import PIL
import torch


class CUB:
    def __init__(self, mode='train', data_len=None):
        self.root = '/home/cougarnet.uh.edu/amobiny/Desktop/NTS_network/CUB_200_2011'
        self.mode = mode
        self.input_size = options.img_w
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.mode == 'train':
            self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if self.mode == 'test':
            self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            # img = transforms.RandomResizedCrop(self.input_size, scale=(0.5, 1.))(img)
            # img = transforms.RandomHorizontalFlip()(img)
            # img = transforms.RandomRotation(degrees=90, resample=PIL.Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if options.multicrop:
                img = transforms.Resize((self.input_size+100, self.input_size+100), Image.BILINEAR)(img)
                img = transforms.TenCrop(self.input_size, vertical_flip=False)(img)
                img = torch.stack([transforms.ToTensor()(im) for im in img])
            else:
                img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
                img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_label)
        else:
            return len(self.test_label)

