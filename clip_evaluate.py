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

from open_flamingo.eval.eval_model import BaseEvalModel

from datasets.coco_detection import CocoDetection
from datasets.pascal_voc import voc2007
from datasets.openimages_common import OpenImagesCommon
from datasets.openimages_rare import OpenImagesRare
from datasets.ade20k import ADE20k
from datasets.image_dataset2 import image_dataset
from datasets.open_clip_datasets import build_wd_dataset

from datasets.sugar_crepe import SugarCrepe
import json
import open_clip

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
    default="ViT-L-14",
)

parser.add_argument(
    '--pretrained',
    type=str,
    default='openai',
    help='open_clip pretrained argument'
)

parser.add_argument(
    "--vqa",
    action='store_true',
    help="specify if evaluating in vqa form",
)

parser.add_argument(
    '--save_scores',
    action='store_true',
    help='whether to save the scores'
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
        default='expts/tmp'
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

def main():
    args, leftovers = parser.parse_known_args()
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, args.pretrained)
    model.to('cuda')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
    results_dict = {}
    for dataset_name in args.dataset_name.split(','):
        print(f'Running eval for {dataset_name}...')
        metrics = evaluate(args, eval_model=model, tokenizer=tokenizer, dataset_name=dataset_name, transform=preprocess)
        results_dict[dataset_name] = metrics
        
    json.dump(results_dict, open(os.path.join(args.output_dir, 'results.json'), 'w'), indent=4)
    
            
def evaluate(args, eval_model, tokenizer, dataset_name, transform):
    if dataset_name.startswith('sugarcrepe'):
        dataset_func = SugarCrepe
        data_split = dataset_name.split('/')[1]
    else:
        raise NotImplementedError(f'Unknown dataset {dataset_name}')
    
    test_dataset = dataset_func(
        root=args.coco_dataroot, data_split=data_split, transform=transform, dataset_name=dataset_name)
    
    if isinstance(test_dataset, torch.utils.data.IterableDataset):
        test_dataloader = torch.utils.data.DataLoader(
                            test_dataset.batched(args.batch_size), batch_size=None,
                            shuffle=False)
    else:
        test_dataloader = DataLoader(test_dataset, args.batch_size,  shuffle=False, drop_last=False)
        
    preds = []
    targets = []
    for batch in tqdm(iter(test_dataloader)):
        if dataset_name.startswith('sugarcrepe'):
            batch_images, batch_captions = batch
            batch_target = torch.zeros(len(batch_images))
            batch_images = batch_images.to('cuda')
        else:
            raise NotImplementedError(f'Unknown dataset {dataset_name}')
        
        with torch.inference_mode():
            image_feats = eval_model.encode_image(batch_images)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            
            text_feats = []
            for _, captions in enumerate(batch_captions):
                caption_tok_ids = tokenizer(captions)
                caption_tok_ids = caption_tok_ids.to('cuda')
                curr_feats = eval_model.encode_text(caption_tok_ids)
                curr_feats = curr_feats / curr_feats.norm(dim=-1, keepdim=True)
                text_feats.append(curr_feats)
                
            text_feats = torch.stack(text_feats, dim=1)
            sims = torch.einsum('bd,bcd->bc', image_feats, text_feats)
            preds.append(sims.cpu())
            targets.append(batch_target)
            
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    
    acc = top_k_accuracy(preds, targets, k=1)
    print('Top-1 is %0.2f' % (acc*100))
    ret = {'top-1 acc': acc}
    
    if args.save_scores:
        dset_extension = '_'.join(dataset_name.split('/'))
        torch.save(preds, os.path.join(args.output_dir, f'preds_{dset_extension}.pt'))
    
    return ret
    
        
def top_k_accuracy(pred, targets, k=1):
    # Get the top k indices of the predictions
    top_k_preds = torch.topk(pred, k, dim=1)[1]  # [n, k]

    # Expand targets to [n, k] for comparison
    targets = targets.view(-1, 1).expand_as(top_k_preds)

    # Check if targets are in top k predictions
    correct = torch.any(top_k_preds == targets, dim=1).float().sum()

    return (correct / len(targets)).cpu().item()

if __name__ == "__main__":
    main()