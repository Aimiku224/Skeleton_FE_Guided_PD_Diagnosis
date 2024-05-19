import torch.utils.data as data
import cv2
import PIL.Image as Image
import pandas as pd
import os
import random
import numpy as np


class FerDataSet(data.Dataset):
    def __init__(self, data_path, label_path, label_name, mode='train', transform=None, basic_aug=False):
        """
        :param data_path: path of dataset
        :param mode: train or eval
        """
        super().__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.basic_aug = basic_aug
        self.expr_dict = {'anger':0, 'disgust':1, 'fear':2, 'happy':3, 'sadness':4, 'surprise':5}

        if mode == 'train':
            with open(os.path.join(self.label_path, label_name), "r", encoding="utf-8") as f:
                self.info = f.readlines()
        else:
            with open(os.path.join(self.label_path, label_name), "r", encoding="utf-8") as f:
                self.info = f.readlines()

        for img_info in self.info:
            sub_id = img_info.strip()
            for expr in self.expr_dict.keys():
                img_name = sub_id + '_' + expr + '.jpg'
                self.img_paths.append(os.path.join(self.data_path, img_name))
                self.labels.append(self.expr_dict[expr])


    def __getitem__(self, idx):
        """
        :param index: 
        :return:
        """
        # open the image and get the corresponding label
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        label = self.labels[idx]
        label = np.array([label], dtype="int64")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def print_sample(self, index: int = 0):
        print("filename", self.img_paths[index], "\tlabel", self.labels[index])

    def __len__(self):
        return len(self.img_paths)
