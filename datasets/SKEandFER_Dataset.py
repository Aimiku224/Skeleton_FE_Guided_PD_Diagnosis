import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
from torch.utils.data import DataLoader
from stgcnplusplus.datasets import build_dataloader


class SKEandFERDataset(data.Dataset):
    def __init__(self, args, skdataset, dataloader_setting, pdfredataset):
        """
        :param data_path: path of dataset
        :param mode: train or eval
        """
        super().__init__()
        self.keypoints = []
        self.imgs = []
        self.labels = []
        
        self.ske_dataloader = build_dataloader(skdataset, **dataloader_setting)
        self.pdfer_dataloader = DataLoader(pdfredataset, batch_size=1, shuffle=False, num_workers=args.workers)

        for i, data in enumerate(self.ske_dataloader):
            skeleton = data['keypoint'][0]
            self.keypoints.append(skeleton)
        for i, (imgs, label) in enumerate(self.pdfer_dataloader):
            self.imgs.append(imgs)
            self.labels.append(label)

    def __getitem__(self, idx):
        """
        :param index: 
        :return:
        """
        keypoints = self.keypoints[idx]
        imgs = self.imgs[idx]
        label = self.labels[idx]
        
        return imgs, keypoints, label

    def print_sample(self, index:int = 0):
        print("imgs",self.imgs[index][0].shape, "\t keypoints", self.keypoints[index].shape, "\t label", self.labels[index].shape)
        print("imgs", self.imgs[index], "\t keypoints", self.keypoints[index], "\t label", self.labels[index])

    def __len__(self):
        return len(self.labels)
