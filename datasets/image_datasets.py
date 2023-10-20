import os
from torchvision.datasets import ImageFolder

from datasets.imagenet_utils import IMAGENET_1K_CLASS_ID_TO_LABEL, openai_imagenet_classnames
from datasets.elevater_utils import class_map_cls_id_to_class, class_names


class ImageDataset(ImageFolder):
    """Class to represent the datatset in ElEVATER."""

    def __init__(self, root, dataset_name, data_split, offset=0, **kwargs):
        super().__init__(root=os.path.join(root, data_split), **kwargs)
        self.offset = offset
        self.dataset_name = dataset_name

        if dataset_name == 'imagenet-1k':
            self.classnames = openai_imagenet_classnames
            self.cls_id_to_class = IMAGENET_1K_CLASS_ID_TO_LABEL
        else:
            self.classnames = class_names[dataset_name]
            self.cls_id_to_class = class_map_cls_id_to_class[dataset_name]

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = self.cls_id_to_class[target+self.offset]
        return {
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }
