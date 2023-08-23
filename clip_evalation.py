import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from tqdm import tqdm


from datasets.coco_detection import CocoDetection
from datasets.pascal_voc import voc2007
from torchvision import transforms
from PIL import Image
from datasets import template_map
import clip


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=8)


# Dataset arguments
parser.add_argument(
    "--dataset_name", type=str, default='coco', help="the name of the dataset to evaluate",
)


## COCO Dataset
parser.add_argument(
    "--dataroot",
    type=str,
    default=None,
)


parser.add_argument(
    "--prompt_key",
    type=str,
    help="Define which prompts we want to use for the evaluation",
    default="imagenet-1k",
)


parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="the output directory path",
    )


def get_model(feature_type):
    model, preprocess = clip.load("ViT-B/32", device='cuda:0')
    if feature_type == 'image':
        model.forward = model.encode_image
    elif feature_type == 'text':
        model.forward = model.encode_text
    else:
        raise Exception('Incorrect model type')
    return model


def extract_text_features(prompt_template, classnames):
    templates = prompt_template
    model = get_model(feature_type='text')
    model.eval()
    zeroshot_weights = []
    for classname in tqdm(classnames, 'Extracting text features with model CLIP-VIT-L/14.'):
        if type(classname) == list: classname = classname[0]
        texts = [template.format(classname) for template in templates]
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to('cuda:0')
    return zeroshot_weights


def extract_feature(data_loader):
    model = get_model(feature_type='image')
    model.to('cuda:0')
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, 'Extracting features with model CLIP-VIT-L/14.'):
            x, y = batch[:2]
            y = y.max(dim=1)[0]
            # compute output
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            outputs = model.encode_image(x)
            outputs /= outputs.norm(dim=-1, keepdim=True)
            all_features.append(outputs)
            all_labels.append(y)

    features = torch.concat(all_features)
    labels = torch.concat(all_labels)
    return torch.reshape(features, (features.shape[0], -1)), torch.reshape(labels, (labels.shape[0], -1))

def compute_map(y_true, y_pred):
    """
    Compute Mean Average Precision (mAP) for binary multi-label classification

    Args:
    y_true: ground-truth label array of size [batch_size, num_class]
    y_pred: prediction array of size [batch_size, num_class]

    Returns:
    mAP score
    """

    num_class = y_true.shape[1]
    APs = []

    for i in range(num_class):
        AP = average_precision_score(y_true[:, i: i+1], y_pred[:, i: i+1])
        APs.append(AP)

    mAP = np.mean(APs)
    return mAP

args, leftovers = parser.parse_known_args()

model_args = {
    leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
}

# define the dataset and dataloader
if args.dataset_name == 'coco':
    dataset_func = CocoDetection
    data_split = 'val2014'
else:
    dataset_func = voc2007
    data_split = 'val'

# define the data transform based on ELEVATER paper
transform_clip = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean= [0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

test_dataset = dataset_func(
                root=args.dataroot, data_split=data_split, transform=transform_clip
            )

test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

# define the prompts
prompt_template = template_map[args.prompt_key]
class_names = test_dataset.classnames


# extract the text feature
text_features = extract_text_features(prompt_template, class_names)

# iterate over the dataset,
image_features, targets = extract_feature(test_loader)

# make prediction based on threshold
cross_similarity = image_features @ text_features
predictions = (cross_similarity > args.threshold).int()

mAP = compute_map(y_true=targets.cpu().numpy(), y_pred=predictions.cpu().numpy())

print('mAP is %0.2f' % mAP)



