import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np


class PdDataSet(data.Dataset):
    def __init__(self, data_path, label_path, label_name, mode='train', transform=None):
        """
        :param data_path: path of dataset
        :param mode: train or eval
        """
        super().__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.transform = transform
        self.sub_ids = []
        self.labels = []
        self.expr_dict = {'anger':0, 'disgust':1, 'fear':2, 'happy':3, 'sadness':4, 'surprise':5}

        if mode == 'train':
            with open(os.path.join(self.label_path, label_name), "r", encoding="utf-8") as f:
                self.info = f.readlines()
        else:
            with open(os.path.join(self.label_path, label_name), "r", encoding="utf-8") as f:
                self.info = f.readlines()

        for img_info in self.info:
            mp4 ,sub_id, pd_label = img_info.strip().split(' ')
            self.sub_ids.append(sub_id)
            self.labels.append(pd_label)


    def __getitem__(self, idx):
        """
        :param index: 
        :return:
        """
        # open the image and get the corresponding label
        imgs = []
        for expr in self.expr_dict.keys():
            img_name = self.sub_ids[idx] + '_' + expr + '.jpg'
            img_path = os.path.join(self.data_path, img_name)
            if not os.path.exists(img_path):
                print("Don't have"+ img_path)
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        
        label = self.labels[idx]
        label = np.array([label], dtype="int64")
        # print("PDDatasets"+str(np.array(imgs[0]).shape))
        return imgs, label

    def print_sample(self, index: int = 0):
        print("subject_id", self.sub_ids[index], "\tlabel", self.labels[index])

    def __len__(self):
        return len(self.sub_ids)
