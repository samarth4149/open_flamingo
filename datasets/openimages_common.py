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


class OpenImagesCommon(data.Dataset):
    def __init__(self, root, data_split, transform=None, start_idx=0, **kwargs):
        # data_split = train / val
        self.root = root
        classnames = []
        class_name_file = 'common/openimages_common_214_ram_taglist.txt'
        with open(os.path.join(self.root, class_name_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        self.classnames = classnames

        # self.annFile = os.path.join(self.root, 'common/openimages_common_214_ram_taglist.txt')

        image_list_file = os.path.join(self.root, 'common/openimages_common_214_ram_annots.txt')

        with open(image_list_file) as f:
            image_list = f.readlines()
        self.image_list = [a.strip() for a in image_list]
        self.image_list = self.image_list[start_idx: ]

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        tokens = self.image_list[index].split(',')
        img_path = os.path.join(self.root, tokens[0] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        label_vector = torch.zeros(len(self.classnames))
        for token in tokens[1:]:
            label_vector[self.classnames.index(token)] = 1

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
        return 'openimagesv6_common'


if __name__ == '__main__':
  root_path = '/projectnb/ivc-ml/sunxm/datasets/OpenImagesV6/'
  data_split = 'test'
  dataset = OpenImagesCommon(root_path,  data_split)
  batch = dataset[100]
  import pdb
  pdb.set_trace()
