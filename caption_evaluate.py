import sys
sys.path.insert(0, './')

import argparse
import importlib
import json
import os
import random
import uuid
from collections import defaultdict

from einops import repeat
import more_itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
import sng_parser
from wordhoard import Synonyms
from sklearn.metrics import average_precision_score

import nltk
from nltk.corpus import wordnet
#
from open_flamingo.eval.coco_metric import compute_cider, postprocess_captioning_generation
# from eval_datasets import CaptionDataset, VQADataset, ImageNetDataset, ImageDataset
from tqdm import tqdm


# from eval_datasets import VQADataset, ImageNetDataset
# from open_flamingo.eval.imagenet_utils import (
#     openai_imagenet_classnames,
#     IMAGENET_1K_CLASS_ID_TO_LABEL,
# )
#
# from open_flamingo.eval.elevater_utils import (
#     class_map_cls_id_to_class,
#     class_map,
# )
#
from open_flamingo.eval.eval_model import BaseEvalModel

from open_flamingo.eval.ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo.src.flamingo import Flamingo

from datasets.coco_detection import CocoDetection

# from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

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


def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if args.eval_coco:
        print("Evaluating on COCO...")
        scores = []
        map_score = evaluate_captioning(
            args,
            eval_model=eval_model,
            dataset_name="coco",
        )



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
                image_path, predictions, labels = triplets[i + j]
                html.append("<td>")
                html.append(f'<img src="file://{image_path}" width="300" height="300"><br>')
                html.append("Predictions:<br>")
                html.append(', '.join(predictions))
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
    if dataset_name == "coco":
        # build test dataset
        test_dataset = CocoDetection(
            root=args.coco_dataroot, data_split='val2014', transform=eval_model.image_processor
        )
        class_synonyms = []
        class_names = test_dataset.classnames
        class_names_np = np.array(class_names)
        for class_name in class_names:
            synonyms = []
            for syn in wordnet.synsets(class_name):
                for l in syn.lemmas():
                    synonyms.append(l.name())
            class_synonyms.append(synonyms)
    else:
        raise ValueError('Dataset %s is not supported' % dataset_name)

    test_dataloader = DataLoader(test_dataset, args.batch_size,  shuffle=False, drop_last=False)

    targets = []
    preds = []
    triplets = []
    for batch in tqdm(iter(test_dataloader)):
        batch_images, batch_target, batch_path = batch
        batch_target = batch_target.max(dim=1)[0]

        batch_images = batch_images.unsqueeze(1).unsqueeze(1)

        prompt = args.coco_prompts
        batch_text = [f"<image>{prompt} "] * len(batch_images)

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        new_predictions = [
            postprocess_captioning_generation(out, split_words=['.', '\n', prompt, prompt.capitalize()]).replace('"', "") for out in outputs
        ]

        # extract the nouns based on parser,
        batch_words = []
        for pred in new_predictions:
            pred_graph = sng_parser.parse(pred)
            pred_entities = pred_graph['entities']
            words = []
            for entity in pred_entities:
                words.append(entity['lemma_head'])
            batch_words.append(words)

        # Generate predictions from captions
        predictions = np.zeros((len(new_predictions), len(class_names)), dtype=np.int32)
        for b_idx, words in enumerate(batch_words):
            for word in words:
                for c_idx, class_synonym in enumerate(class_synonyms):
                    if word in class_synonym:
                        predictions[b_idx, c_idx] = 1

        # compute mAP with the ground truth label
        targets.append(batch_target)
        preds.append(predictions)

        # generate triplets for visualization
        # # Example usage:
        visual_path = '/Users/sunxm/Documents/research/datasets/mscoco_2014/val2014'
        for path, prediction, target in zip(batch_path, predictions, batch_target):
            # image_path
            image_path = os.path.join(visual_path, path)
            # predict list
            pred_classes = class_names_np[prediction == 1]
            # ground-truth label list
            target_np = target.cpu().numpy()
            gt_classes = class_names_np[target_np == 1]
            triplet = (image_path, pred_classes, gt_classes)
            triplets.append(triplet)


    # compute mAP with the ground truth label
    preds = np.concatenate(preds, axis=0)
    targets = torch.cat(targets, dim=0)
    mAP = compute_map(y_true=targets.cpu().numpy(), y_pred=preds)
    print('mAP is %0.2f' % mAP)

    # visualize the prediction
    html = create_html_page(triplets)

    # write the html string to a file
    html_folder = 'html'
    if not os.path.isdir(html_folder):
        os.makedirs(html_folder)
    with open(os.path.join(html_folder, 'coco_flamingo_%s'% ('_'.join(args.coco_prompts))), 'w') as f:
        f.write(html)




def evaluate_vqa(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """

    if dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
    elif dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
    elif dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
    elif dataset_name == "textvqa":
        train_image_dir_path = args.textvqa_image_dir_path
        train_questions_json_path = args.textvqa_train_questions_json_path
        train_annotations_json_path = args.textvqa_train_annotations_json_path
        test_image_dir_path = args.textvqa_image_dir_path
        test_questions_json_path = args.textvqa_test_questions_json_path
        test_annotations_json_path = args.textvqa_test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataset = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        seed,
    )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)
    predictions = []

    for batch in more_itertools.chunked(
        tqdm(test_dataset, desc=f"Running inference {dataset_name.upper()}"),
        args.batch_size,
    ):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch)
        )

        batch_images = []
        batch_text = []
        for i in range(len(batch)):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch[i]["image"]])

            context_text = "".join(
                [
                    eval_model.get_vqa_prompt(
                        question=x["question"], answer=x["answers"][0]
                    )
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=batch[i]["question"])
            )

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        process_function = (
            postprocess_ok_vqa_generation
            if dataset_name == "ok_vqa"
            else postprocess_vqa_generation
        )

        new_predictions = map(process_function, outputs)

        predictions.extend(
            [
                {"answer": p, "question_id": sample["question_id"]}
                for p, sample in zip(new_predictions, batch)
            ]
        )
    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"{dataset_name}results_{random_uuid}.json",
        test_questions_json_path,
        test_annotations_json_path,
    )

    # delete the temporary file
    os.remove(f"{dataset_name}results_{random_uuid}.json")

    return acc

if __name__ == "__main__":
    main()
