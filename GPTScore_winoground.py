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
from datasets.winoground import Winoground
import json
from my_utils.ioutils import save_expt_info

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
    '--ptrain',
    default=None,
    help='What kind of p_train to get (will compute GPTScore if not specified)',
    choices=['gaussian', 'language_only']
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
        from llava.eval import LLaVA
        eval_model = LLaVA(model_args['model_name'])
    elif args.model == 'llava_v1_5':
        from llava_v1_5.eval import LLaVA_v1_5
        eval_model = LLaVA_v1_5(model_name=model_args['model_name'], model_path=model_args['model_path'], model_base=model_args.get('model_base', None))
    elif args.model == 'minigpt4':
        from minigpt4.eval import MiniGPT4
        from minigpt4.common.config import Config
        cfg = Config(args)
        eval_model = MiniGPT4(cfg.model_cfg, cfg.datasets_cfg.cc_sbu_align.vis_processor.train, int(model_args["device"]))
    elif args.model in ['minigpt4_llama2', 'minigpt_v']:
        from minigpt4_v2.eval_llama2 import MiniGPT4_llama2
        from minigpt4_v2.common.config import Config
        cfg = Config(args)
        eval_model = MiniGPT4_llama2(cfg.model_cfg, cfg.datasets_cfg.cc_sbu_align.vis_processor.train, int(model_args["device"]))
    else:
        module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")
        eval_model = module.EvalModel(model_args)
    
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    dataset = Winoground(
        root_dir='/projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/', 
        image_preprocess=eval_model.image_processor,
        return_image_paths=False)
    loader = DataLoader(dataset, args.batch_size,  shuffle=False, drop_last=False)
    
    # TODO : arrange images from winoground with captions appropriately to get GPTScore1
    all_scores = []
    for batch in tqdm(loader):
        # batch['images'] has shape B x 2 x (img_shape)
        # batch['captions'] has shape B x 2
        
        # fetching images is complex because of stupid datastructures in the model
        batch_imgs1 = batch['images'][0]['pixel_values'][0].cuda(non_blocking=True)
        batch_imgs2 = batch['images'][1]['pixel_values'][0].cuda(non_blocking=True)
        captions = batch['texts']
        
        scores1 = eval_model.get_GPTScore1({'pixel_values' : [batch_imgs1]}, captions, prompt=args.coco_prompts).cpu()
        scores2 = eval_model.get_GPTScore1({'pixel_values' : [batch_imgs2]}, captions, prompt=args.coco_prompts).cpu()
        
        scores = torch.stack([scores1, scores2], dim=1)
        all_scores.append(scores)
    
    all_scores = torch.cat(all_scores, dim=0)
    ret = dataset.evaluate_scores(all_scores)
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_expt_info(args, outdir=args.output_dir)
    torch.save(all_scores, os.path.join(args.output_dir, f'scores.pt'))
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(ret[0], f, indent=4)
    

if __name__ == "__main__":
    main()
