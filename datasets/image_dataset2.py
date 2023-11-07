from vision_datasets import DatasetHub
from vision_datasets.pytorch import TorchDataset
from vision_datasets import ManifestDataset
import torchvision.datasets
import os
import pathlib
from vision_datasets import Usages
from torchvision import transforms
from PIL import Image
from datasets.imagenet_utils import openai_imagenet_classnames
from datasets.elevater_utils import class_names

# from imagenet_utils import openai_imagenet_classnames
# from elevater_utils import class_names

VISION_DATASET_STORAGE = 'https://cvinthewildeus.blob.core.windows.net/datasets'


def get_dataset_hub():
    vision_dataset_json = (pathlib.Path(__file__).resolve().parents[1] / 'datasets' / 'vision_datasets.json').read_text()
    hub = DatasetHub(vision_dataset_json)

    return hub


def image_dataset(root, transform, dataset_name, **kwargs):
    hub = get_dataset_hub()
    vision_dataset_storage = 'https://cvinthewildeus.blob.core.windows.net/datasets'
    results = hub.create_dataset_manifest(vision_dataset_storage, root, dataset_name, usage=Usages.TEST_PURPOSE)
    test_set, test_set_dataset_info, _ = results
    # if dataset_name != 'imagenet-1k':
    test_set = TorchDataset(ManifestDataset(test_set_dataset_info, test_set), transform=transform)
    # else:
    #     test_set = torchvision.datasets.ImageFolder(os.path.join(root, 'val'), transform=transform)

    if dataset_name == 'imagenet-1k':
        test_set.classnames = openai_imagenet_classnames
    else:
        test_set.classnames = class_names[dataset_name]
    return test_set


if __name__ == '__main__':
    transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    dataset = image_dataset('/projectnb/ivc-ml/sunxm/datasets', 'cifar-10', transform)
    import pdb
    pdb.set_trace()