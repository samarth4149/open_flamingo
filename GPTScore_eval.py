import sys
sys.path.insert(0, './')

import argparse
import importlib
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from minigpt4.eval import MiniGPT4
from minigpt4_v2.eval_llama2 import MiniGPT4_llama2
from llava.eval import LLaVA
from llava_v1_5.eval import LLaVA_v1_5

from open_flamingo.eval.eval_model import BaseEvalModel

from datasets.coco_detection import CocoDetection
from datasets.pascal_voc import voc2007
from datasets.openimages_common import OpenImagesCommon
from datasets.openimages_rare import OpenImagesRare
from datasets.ade20k import ADE20k
from datasets.image_dataset2 import image_dataset
from datasets.open_clip_datasets import build_wd_dataset

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=8)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)

# Dataset arguments
parser.add_argument(
    "--dataset_name", type=str, default='coco', help="the name of the dataset to evaluate",
)


## COCO Dataset
parser.add_argument(
    "--coco_dataroot",
    type=str,
    default=None,
)

parser.add_argument(
    "--coco_prompts",
    type=str,
    default='caption:',
)


parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)

parser.add_argument(
    "--vqa",
    action='store_true',
    help="specify if evaluating in vqa form",
)

parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

# parser.add_argument("--model_path", type=str, help="path to model checkpoint.")

parser.add_argument(
        "--output_dir",
        type=str,
        help="the output directory path",
    )

parser.add_argument(
        "--save_freq",
        type=int,
        default=10,
        help="the output directory path",
    )

parser.add_argument(
        "--task",
        type=str,
        default='',
        help="the task identifier for minigpt-v"
    )


parser.add_argument(
        "--num_splits",
        type=int,
        default=None,
        help="the task identifier for minigpt-v"
    )

parser.add_argument(
        "--split_id",
        type=int,
        default=None,
        help="the task identifier for minigpt-v"
    )

parser.add_argument("--cfg-path",  help="path to configuration file.")


def similarity(nlp, phrase1, phrase2):
    # Load the medium English model. This model has word vectors.
    # nlp = spacy.load("en_core_web_md")

    # Process the phrases to get their vector representations
    phrase1 = nlp(phrase1)
    phrase2 = nlp(phrase2)

    # Compute the similarity between the two phrases
    similarity = phrase1.similarity(phrase2)
    return similarity

def main():
    args, leftovers = parser.parse_known_args()

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    if args.model == 'llava':
        eval_model = LLaVA(model_args['model_name'])
    elif args.model == 'llava_v1_5':
        eval_model = LLaVA_v1_5(model_name=model_args['model_name'], model_path=model_args['model_path'], model_base=model_args.get('model_base', None))
    elif args.model == 'minigpt4':
        from minigpt4.common.config import Config
        cfg = Config(args)
        eval_model = MiniGPT4(cfg.model_cfg, cfg.datasets_cfg.cc_sbu_align.vis_processor.train, int(model_args["device"]))
    elif args.model in ['minigpt4_llama2', 'minigpt_v']:
        from minigpt4_v2.common.config import Config
        cfg = Config(args)
        eval_model = MiniGPT4_llama2(cfg.model_cfg, cfg.datasets_cfg.cc_sbu_align.vis_processor.train, int(model_args["device"]))
    else:
        module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")
        eval_model = module.EvalModel(model_args)


    evaluate_captioning(args, eval_model=eval_model, dataset_name=args.dataset_name)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + query_set_size must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


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


def top_k_accuracy(pred, targets, k=1):
    # Get the top k indices of the predictions
    top_k_preds = torch.topk(pred, k, dim=1)[1]  # [n, k]

    # Expand targets to [n, k] for comparison
    targets = targets.view(-1, 1).expand_as(top_k_preds)

    # Check if targets are in top k predictions
    correct = torch.any(top_k_preds == targets, dim=1).float().sum()

    return (correct / len(targets)).cpu().item()

def get_query_set(train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, seed):
    np.random.seed(seed)
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    return torch.utils.data.Subset(test_dataset, random_indices)


def create_html_page(triplets):
    """
    Creates an HTML page to visualize the triplets.

    Args:
    triplets: A list of triplets. Each triplet is a tuple (image_path, predictions, labels), where
        - image_path is a string path to the image file
        - predictions is a list of predicted labels (strings)
        - labels is a list of ground truth labels (strings)

    Returns:
    An HTML string.
    """
    html = ["<html><body><table>"]

    # add a row for every 3 triplets
    for i in range(0, len(triplets), 3):
        html.append("<tr>")
        for j in range(3):
            if i + j < len(triplets):
                image_path, caption, predictions, labels = triplets[i + j]
                html.append("<td>")
                html.append(f'<img src="file://{image_path}" width="300" height="300"><br>')
                html.append("Predictions:<br>")
                html.append(caption + '----' +', '.join(predictions))
                html.append("<br>")
                html.append("Labels:<br>")
                html.append(', '.join(labels))
                html.append("</td>")
        html.append("</tr>")

    html.append("</table></body></html>")

    return "\n".join(html)

def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def evaluate_captioning(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    max_generation_length: int = 20,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    dataset_name: str = "coco",
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 20.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
    Returns:
        float: CIDEr score

    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if dataset_name in ["coco", "pascal_voc", "OpenImagesV6Common", "OpenImagesV6Rare", "ADE20k", 'imagenet-1k',
                        'cifar-10', 'cifar-100', 'oxford-flower-102', 'stanford-cars', 'rendered-sst2', 'eurosat_clip',
                        'mnist', 'oxford-iiit-pets', 'gtsrb', 'resisc45_clip', 'dtd', 'fgvc-aircraft-2013b-variants102',
                        'caltech-101', 'country211'] or dataset_name.startswith('wds/'):
        # build test dataset
        if dataset_name == 'coco':
            dataset_func = CocoDetection
            data_split = 'val2014'
        elif dataset_name == 'pascal_voc':
            dataset_func = voc2007
            data_split = 'test'
        elif dataset_name == 'OpenImagesV6Common':
            dataset_func = OpenImagesCommon
            data_split = 'test'
        elif args.dataset_name == 'OpenImagesV6Rare':
            dataset_func = OpenImagesRare
            data_split = 'test'
        elif args.dataset_name == 'ADE20k':
            dataset_func = ADE20k
            data_split = 'validation'
        elif args.dataset_name == 'imagenet-1k':
            # image classification datasets
            dataset_func = image_dataset
            data_split = None
        elif args.dataset_name.startswith('wds/'):
            dataset_func = build_wd_dataset
            data_split = None
        else:
            dataset_func = image_dataset
            data_split = None

        if args.model == 'open_flamingo':
            test_dataset = dataset_func(
                root=args.coco_dataroot, data_split=data_split, transform=eval_model.image_processor, dataset_name=args.dataset_name
            )
        elif args.model == 'blip':
            test_dataset = dataset_func(
                root=args.coco_dataroot, data_split=data_split, transform=eval_model.processor.image_processor, dataset_name=args.dataset_name
            )
        elif args.model in ['minigpt4', 'minigpt4_llama2', 'minigpt_v']:
            test_dataset = dataset_func(
                root=args.coco_dataroot, data_split=data_split, transform=eval_model.vis_processor, dataset_name=args.dataset_name
            )
        elif args.model in ['llava', 'llava_v1_5']:
            def image_transform(image, target):
                modified_image = eval_model.image_processor(image)
                return modified_image, target
            if args.dataset_name.startswith('wds/'):
                image_transform = eval_model.image_processor

            test_dataset = dataset_func(
                root=args.coco_dataroot, data_split=data_split, transform=image_transform, dataset_name=args.dataset_name
            )
        else:
            raise ValueError(f'model {args.model} is not supported')
    else:
        raise ValueError('Dataset %s is not supported' % dataset_name)


    if dataset_name.startswith('wds/'):
        class_names = test_dataset.classes
    else:
        class_names = test_dataset.classnames

    if args.split_id is not None and args.num_splits is not None:
        dataset_length = len(test_dataset)
        split_size = (dataset_length - 1) // args.num_splits + 1
        split_range = list(range(args.split_id * split_size, min([dataset_length, (args.split_id + 1)* split_size])))
        test_dataset = torch.utils.data.Subset(test_dataset, split_range)

    if isinstance(test_dataset, torch.utils.data.IterableDataset):
        test_dataloader = torch.utils.data.DataLoader(
                            test_dataset.batched(args.batch_size), batch_size=None,
                            shuffle=False)
    else:
        test_dataloader = DataLoader(test_dataset, args.batch_size,  shuffle=False, drop_last=False)

    targets = []
    preds = []
    count = 0
    for batch in tqdm(iter(test_dataloader)):
        if dataset_name in ["coco", "pascal_voc", "OpenImagesV6Common", "OpenImagesV6Rare", "ADE20k"]:
            batch_images, batch_target, batch_path = batch
            batch_target = batch_target.max(dim=1)[0]
        # elif dataset_name == 'imagenet-1k':
        #     import pdb
        #     pdb.set_trace()
        #     batch_images = batch['image']
        #     batch_target = batch['class_id']
        elif dataset_name.startswith('wds/'):
            batch_images, batch_target = batch
            if 'llava' in args.model:
                batch_images_ = []
                for image in batch_images:
                    batch_images_.append(image['pixel_values'][0])
                batch_images = {'pixel_values': [torch.stack(batch_images_, dim=0)]}
        else:
            batch_images = batch[0]
            batch_target = batch[1][0]

        if args.model == 'open_flamingo':
            batch_images = batch_images.unsqueeze(1).unsqueeze(1)

        prompt = args.coco_prompts
        batch_text = [f"<image>{prompt} "] * len(batch_images)

        if args.model in ['minigpt4', 'minigpt4_llama2', 'llava', 'llava_v1_5']:
            outputs = eval_model.get_GPTScore(batch_images=batch_images, class_names=class_names,  prompt=prompt)
        elif args.model in ['minigpt_v']:
            outputs = eval_model.get_GPTScore(batch_images=batch_images, class_names=class_names,  prompt=prompt,
                                              task=args.task)
        else:
            outputs = eval_model.get_outputs(
                batch_images=batch_images,
                batch_text=batch_text,
                max_generation_length=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        # compute mAP with the ground truth label
        targets.append(batch_target)
        preds.append(outputs)
        count += 1
        # if count >= 1:
        #     break

    # compute mAP with the ground truth label
    preds = torch.exp(torch.cat(preds, dim=0)).float()
    targets = torch.cat(targets, dim=0)
    if args.dataset_name in ["coco", "pascal_voc", "OpenImagesV6Common", "OpenImagesV6Rare", "ADE20k"]:
        mAP = compute_map(y_true=targets.cpu().numpy(), y_pred=preds.cpu().numpy())
        print('mAP is %0.2f' % mAP)
    else:
        acc = top_k_accuracy(preds.cpu(), targets, k=1)
        print('Top-1 is %0.2f' % (acc*100))

    if args.split_id is not None:
        np.save(os.path.join(args.output_dir, 'pred_%d.npy' % args.split_id), preds.cpu().numpy())
        np.save(os.path.join(args.output_dir, 'target_%d.npy' % args.split_id), targets.cpu().numpy())



if __name__ == "__main__":
    main()
