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
# from . import utils_ade20k


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
        object_list = os.path.join(root, 'objectInfo150.txt')
        classnames = []
        with open(object_list, 'r') as f:
            lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            tokens = line.strip().split(' ')
            classnames.append(tokens[-1])

        self.classnames = classnames

        image_ann_file = os.path.join(root, 'annotations', '%s.txt' % data_split)
        with open(image_ann_file, 'r') as f:
            image_list = f.readlines()

        self.data_split = data_split
        self.image_list = image_list[start_idx:]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        line = self.image_list[index]
        tokens = line.split(',')
        img_path = os.path.join('images', self.data_split, tokens[0]).replace('png', 'jpg')
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        label_vector = torch.zeros(len(self.classnames)).reshape(-1)
        for token in tokens[1:]:
            label_vector[int(token)] = 1

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
  root_path = '/projectnb/ivc-ml/sunxm/datasets/ADEChallengeData2016'
  data_split = 'validation'
  dataset = ADE20k(root_path,  data_split)
  batch = dataset[100]
  import pdb
  pdb.set_trace()
