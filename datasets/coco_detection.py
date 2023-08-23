import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
from torchvision import datasets as datasets
from pycocotools.coco import COCO
from PIL import Image
import torch
import os
from torchvision.transforms import transforms
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor
from transformers import CLIPImageProcessor

class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, data_split, transform, start_idx=0):
        # super(CocoDetection, self).__init__()
        self.classnames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                           "kite",
                           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                           "orange",
                           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]
        self.root = root
        annFile = os.path.join(self.root, 'annotations', 'instances_%s.json' % data_split)
        cls_id = list(range(len(self.classnames)))
        cls_id.sort()
        self.coco = COCO(annFile)
        self.data_split = data_split
        self.ids = list(self.coco.imgToAnns.keys())[start_idx:]

        # train_transform = transforms.Compose([
        #     # transforms.RandomResizedCrop(img_size)
        #     transforms.Resize((img_size, img_size)),
        #     CutoutPIL(cutout_factor=0.5),
        #     RandAugment(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])
        # test_transform = transforms.Compose([
        #     # transforms.CenterCrop(img_size),
        #     transforms.Resize((img_size, img_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])



        # if self.data_split == 'train2014':
        #     self.transform = train_transform
        # elif self.data_split == "val2014":
        #     self.transform = test_transform
        # else:
        #     raise ValueError('data split = %s is not supported in mscoco' % self.data_split)


        self.transform = transform

        self.cat2cat = dict()
        cats_keys = [*self.coco.cats.keys()]
        cats_keys.sort()
        for cat, cat2 in zip(cats_keys, cls_id):
            self.cat2cat[cat] = cat2
        self.cls_id = cls_id

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, len(self.classnames)), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, self.data_split, path)).convert('RGB')
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

        return img, target, path

    # def __len__(self):
    #     return 32

    def name(self):
        return 'coco'


if __name__ == '__main__':
    transform_clip = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    root_path = '/projectnb/ivc-ml/sunxm/datasets/mscoco_2014/'
    import pdb

    data_split = 'val2014'
    dataset = CocoDetection(root_path,  data_split,  transform=transform_clip)
    batch = dataset[100]
    pdb.set_trace()