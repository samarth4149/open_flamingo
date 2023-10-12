import sys
sys.path.insert(0, './')

import argparse

from datasets.coco_detection import CocoDetection
from datasets.pascal_voc import voc2007
from datasets.openimages_common import OpenImagesCommon
from datasets.openimages_rare import OpenImagesRare
from datasets.ade20k import ADE20k

from minigpt4.common.config import Config
from minigpt4.eval import MiniGPT4
from llava.eval import LLaVA
import importlib

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

    # build the model
    if args.model == 'llava':
        eval_model = LLaVA(model_args['model_name'])
    elif args.model == 'minigpt4':
        cfg = Config(args)
        eval_model = MiniGPT4(cfg.model_cfg, cfg.datasets_cfg.cc_sbu_align.vis_processor.train, int(model_args["device"]))
    else:
        module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")
        eval_model = module.EvalModel(model_args)

    # build train and eval dataset, dataloader
    if args.dataset_name in ["coco", "pascal_voc", "OpenImagesV6Common", "OpenImagesV6Rare", "ADE20k"]:
        # build test dataset
        if args.dataset_name == 'coco':
            dataset_func = CocoDetection
            train_data_split = ''
            val_data_split = 'val2014'
        elif args.dataset_name == 'pascal_voc':
            dataset_func = voc2007
            train_data_split = 'trainval'
            val_data_split = 'test'
        elif args.dataset_name == 'OpenImagesV6Common':
            dataset_func = OpenImagesCommon
            val_data_split = 'test'
        elif args.dataset_name == 'OpenImagesV6Rare':
            dataset_func = OpenImagesRare
            val_data_split = 'test'
        elif args.dataset_name == 'ADE20k':
            dataset_func = ADE20k
            val_data_split = 'validation'
        else:
            raise ValueError

        if args.model == 'open_flamingo':
            image_transform = eval_model.image_processor
        elif args.model == 'blip':
            image_transform = eval_model.processor.image_processor
        elif args.model == 'minigpt4':
            image_transform = eval_model.vis_processor
        elif args.model == 'llava':
            image_transform = eval_model.image_processor
        else:
            raise ValueError(f'model {args.model} is not supported')

        test_dataset = dataset_func(
            root=args.coco_dataroot, data_split=data_split, transform=image_transform
        )


        if args.model == 'open_flamingo':

            test_dataset = dataset_func(
                root=args.coco_dataroot, data_split=data_split, transform=eval_model.image_processor
            )
        elif args.model == 'blip':
            test_dataset = dataset_func(
                root=args.coco_dataroot, data_split=data_split, transform=eval_model.processor.image_processor
            )
        elif args.model == 'minigpt4':
            test_dataset = dataset_func(
                root=args.coco_dataroot, data_split=data_split, transform=eval_model.vis_processor
            )
        elif args.model == 'llava':
            test_dataset = dataset_func(
                root=args.coco_dataroot, data_split=data_split, transform=eval_model.image_processor
            )
        else:
            raise ValueError(f'model {args.model} is not supported')
    else:
        raise ValueError('Dataset %s is not supported' % dataset_name)





# auto-resuming

# in loop
# training
# eval
# save the best model
# save the latest model

