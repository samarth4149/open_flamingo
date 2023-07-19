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

def read_labels(path_labels):
    file = path_labels
    labels = []
    with open(file, 'r') as f:
        for line in f:
            tmp = list(map(int, line.strip().split(',')))
            labels.append(torch.tensor(tmp, dtype=torch.long))
    return labels


class voc2007(data.Dataset):
    def __init__(self, root, data_split, transform, start_idx=0):
        # data_split = train / val
        self.root = root
        self.classnames = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train', 'tvmonitor']

        self.annFile = os.path.join(self.root, 'Annotations')

        image_list_file = os.path.join(self.root, 'ImageSets', 'Main', '%s.txt' % data_split)

        with open(image_list_file) as f:
            image_list = f.readlines()
        self.image_list = [a.strip() for a in image_list]
        self.image_list = self.image_list[start_idx: ]

        self.data_split = data_split

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'JPEGImages', self.image_list[index] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        ann_path = os.path.join(self.annFile, self.image_list[index] + '.xml')
        label_vector = torch.zeros(20)
        DOMTree = xml.dom.minidom.parse(ann_path)
        root = DOMTree.documentElement
        objects = root.getElementsByTagName('object')
        for obj in objects:
            if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                continue
            tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
            label_vector[self.classnames.index(tag)] = 1.0
        targets = label_vector.long()
        target = targets[None, ]

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
        return 'voc2007'

if __name__ == '__main__':
  root_path = '/Users/sunxm/Desktop/research/datasets/pascal2007/VOCdevkit/VOC2007/'
  data_split = 'trainval'
  dataset = voc2007(root_path,  data_split)
  batch = dataset[100]
  import pdb
  pdb.set_trace()
  data_split = 'test'
  dataset = voc2007(root_path,  data_split)
  batch = dataset[100]
  pdb.set_trace()