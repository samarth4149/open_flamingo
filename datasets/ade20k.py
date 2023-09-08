import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
import xml.dom.minidom

from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor
from transformers import CLIPImageProcessor
import pickle as pkl
import utils_ade20k


def read_labels(path_labels):
    file = path_labels
    labels = []
    with open(file, 'r') as f:
        for line in f:
            tmp = list(map(int, line.strip().split(',')))
            labels.append(torch.tensor(tmp, dtype=torch.long))
    return labels


class ADE20k(data.Dataset):
    def __init__(self, root, data_split, transform=None, start_idx=0):
        # data_split = train / val
        self.root = root
        index_file = 'ADE20K_2021_17_01/index_ade20k_validation.pkl'
        with open('{}/{}'.format(root, index_file), 'rb') as f:
            index_ade20k = pkl.load(f)
        classnames = index_ade20k['objectnames']

        self.classnames = classnames

        image_list = []
        for folder, filename in zip(index_ade20k['folder'], index_ade20k['filename']):
            full_file_name = '{}/{}'.format(folder, filename)
            info = utils_ade20k.loadAde20K('{}/{}'.format(root, full_file_name))
            image_list.append(info['img_name'])

        self.image_list = image_list[start_idx: ]
        self.labels = index_ade20k['objectIsPart']
        self.labels[self.labels != 0] = 1
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = Image.open(img_path).convert('RGB')
        label_vector = torch.from_numpy(self.labels[:, index]).reshape(-1)

        targets = label_vector.long()
        target = targets[None,]

        if self.transform is not None:
            if isinstance(self.transform, CLIPImageProcessor):
                img = self.transform.preprocess(img, return_tensors='pt')['pixel_values'][0]
            elif isinstance(self.transform, transforms.Compose) or isinstance(self.transform, Blip2ImageEvalProcessor):
                img = self.transform(img)
            else:
                img = self.transform(img, return_tensors="pt")[
                        "pixel_values"
                    ]
                img = img.squeeze(0)

        return img, target, img_path

    def name(self):
        return 'ADE20k'


if __name__ == '__main__':
  root_path = '/projectnb/ivc-ml/sunxm/datasets/ADE20k'
  data_split = 'test'
  dataset = ADE20k(root_path,  data_split)
  batch = dataset[100]
  import pdb
  pdb.set_trace()
