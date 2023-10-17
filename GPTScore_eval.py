import sys
sys.path.insert(0, 'open_flamingo/')

import argparse
import importlib
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import sng_parser
import spacy
import more_itertools

from copy import deepcopy

from nltk.corpus import wordnet
from open_flamingo.eval.coco_metric import postprocess_captioning_generation
from tqdm import tqdm
from minigpt4.common.config import Config
from minigpt4.eval import MiniGPT4
from llava.eval import LLaVA

from open_flamingo.eval.eval_model import BaseEvalModel

from datasets.coco_detection import CocoDetection
from datasets.pascal_voc import voc2007
from datasets.openimages_common import OpenImagesCommon
from datasets.openimages_rare import OpenImagesRare
from datasets.ade20k import ADE20k
import pickle

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
    if args.model == 'llava':
        eval_model = LLaVA(model_args['model_name'])
    elif args.model == 'minigpt4':
        cfg = Config(args)
        eval_model = MiniGPT4(cfg.model_cfg, cfg.datasets_cfg.cc_sbu_align.vis_processor.train, int(model_args["device"]))
    else:
        module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")
        eval_model = module.EvalModel(model_args)

    if args.vqa:
        evaluate_vqa(args, eval_model=eval_model, dataset_name=args.dataset_name)

    else:
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
    nlp = spacy.load("en_core_web_md")
    if dataset_name in ["coco", "pascal_voc", "OpenImagesV6Common", "OpenImagesV6Rare", "ADE20k"]:
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
        else:
            raise ValueError

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

    class_synonyms = []
    class_names = test_dataset.classnames
    class_names_np = np.array(class_names)
    for class_name in class_names:
        synonyms = set()
        _class_names = class_name.split(',')
        for _class_name in _class_names:
            _class_name = _class_name.strip()
            synonyms.add(_class_name.replace('_', ' '))
            for syn in wordnet.synsets(_class_name):
                for l in syn.lemmas():
                    synonyms.add(l.name().replace('_', ' '))
        class_synonyms.append(list(synonyms))

    test_dataloader = DataLoader(test_dataset, args.batch_size,  shuffle=False, drop_last=False)

    targets = []
    preds = []
    triplets = []
    for batch in tqdm(iter(test_dataloader)):
        batch_images, batch_target, batch_path = batch
        batch_target = batch_target.max(dim=1)[0]

        if args.model == 'open_flamingo':
            batch_images = batch_images.unsqueeze(1).unsqueeze(1)

        prompt = args.coco_prompts
        batch_text = [f"<image>{prompt} "] * len(batch_images)

        if args.model in ['minigpt4', 'llava']:
            outputs = eval_model.get_outputs(batch_images=batch_images, prompt=prompt)
        else:
            outputs = eval_model.get_outputs(
                batch_images=batch_images,
                batch_text=batch_text,
                max_generation_length=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        if args.model == 'open_flamingo':
            new_predictions = [
                postprocess_captioning_generation(out, split_words=['.', '\n', prompt, prompt.capitalize()]).replace('"', "") for out in outputs
            ]
        else:
            new_predictions = outputs

        pred_words = []
        for pred in new_predictions:
            pred_graph = sng_parser.parse(pred)
            pred_entities = pred_graph['entities']
            # prediction = pred
            words = []
            for entity in pred_entities:
                # prediction.replace(entity['span'],  entity['lemma_span'])
                words.append(entity['lemma_span'])
            pred_words.append(words)


        # Generate predictions from captions
        predictions = np.zeros((len(new_predictions), len(class_names)), dtype=np.int32)
        for b_idx, words in enumerate(pred_words):
            for word in words:
                match_synonyms = []
                for c_idx, class_synonym in enumerate(class_synonyms):
                    for synonym in class_synonym:
                        if synonym in word:
                            match_synonyms.append((c_idx, synonym))
                best_score = 0
                best_c_idx = None
                for match_synonym in match_synonyms:
                    idx, synonym = match_synonym
                    score = similarity(nlp, word, synonym)
                    if score > best_score:
                        best_score = score
                        best_c_idx = idx

                if best_c_idx is not None:
                    # predictions[b_idx, best_c_idx] = best_score
                    predictions[b_idx, best_c_idx] = 1
                    # import pdb
                    # pdb.set_trace()

        # compute mAP with the ground truth label
        targets.append(batch_target)
        preds.append(predictions)

        # generate triplets for visualization
        # # Example usage:
        visual_path = '/Users/sunxm/Documents/research/datasets/mscoco_2014/val2014'
        for path, caption, prediction, target in zip(batch_path, new_predictions, predictions, batch_target):
            # image_path
            image_path = os.path.join(visual_path, path)
            # predict list
            pred_classes = class_names_np[prediction == 1]
            # ground-truth label list
            target_np = target.cpu().numpy()
            gt_classes = class_names_np[target_np == 1]
            triplet = (image_path, caption, pred_classes, gt_classes)
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
    with open(os.path.join(html_folder, '%s_%s_%s.html' % (dataset_name, args.model, args.coco_prompts)), 'w') as f:
        f.write(html)


def evaluate_image_cls(
    args: argparse.Namespace,
    eval_model,
    batch_size: int,
    seed: int = 42,
    image_cls_part: int = None,
    dataset_name: str = "coco",
):
    """
    Evaluate a model on Image dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        batch_size (int): batch size
        imagenet_root (str): path to imagenet root for the specified split.
        seed (int, optional): random seed. Defaults to 42.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        num_shots (int, optional): number of shots to use. Defaults to 8.

    Returns:
        float: accuracy score
    """
    if not hasattr(eval_model, "model") or not hasattr(eval_model, "tokenizer"):
        raise NotImplementedError(
            "evaluate_imagenet is currently only supported for OpenFlamingo " "models"
        )
    np.random.seed(seed)
    tokenizer = eval_model.tokenizer

    if dataset_name in ["coco", "pascal_voc", "OpenImagesV6Common", "OpenImagesV6Rare", "ADE20k"]:
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
        else:
            raise ValueError

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

    tokenizer.padding_side = (
        "left"  # For generation padding tokens should be on the left
    )

    acc1 = 0
    acc5 = 0
    prompt_text = "<image>A photo of "

    test_iterator = more_itertools.chunked(test_dataset, batch_size)
    for batch_idx, batch in enumerate(test_iterator):
        vision_x = batch

        eval_model._encode_vision_x(vision_x.cuda())

        # Cache the context text: tokenize context and prompt,
        # e.g. '<context> a picture of a '
        ctx_and_prompt_tokenized = tokenizer(
            [context_text + prompt_text + " " for context_text in batch_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        with torch.no_grad():
            precomputed = model(
                vision_x=None,
                lang_x=ctx_and_prompt_tokenized["input_ids"].cuda(),
                attention_mask=ctx_and_prompt_tokenized["attention_mask"].cuda(),
                clear_conditioned_layers=False,
                use_cached_vision_x=True,
                use_cache=True,
            )

        def _detach_pkvs(pkvs):
            """Detach a set of past key values."""
            return tuple([tuple([x.detach() for x in inner]) for inner in pkvs])

        precomputed_pkvs = _detach_pkvs(precomputed.past_key_values)

        precomputed_logits = precomputed.logits.detach()

        overall_probs = []
        for image_cls_class_name in tqdm(class_map[image_cls_dataset_name]):
            past_key_values = None
            # Tokenize only the class name and iteratively decode the model's
            # predictions for this class.
            classname_tokens = tokenizer(image_cls_class_name, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()

            if classname_tokens.ndim == 1:  # Case: classname is only 1 token
                classname_tokens = torch.unsqueeze(classname_tokens, 1)

            classname_tokens = repeat(classname_tokens, "b s -> (repeat b) s", repeat=vision_x.shape[0])

            # Compute the outputs one token at a time, using cached
            # activations.

            # Initialize the elementwise predictions with the last set of
            # logits from precomputed; this will correspond to the predicted
            # probability of the first position/token in the imagenet
            # classname. We will append the logits for each token to this
            # list (each element has shape [B, 1, vocab_size]).
            elementwise_logits = [precomputed_logits[:, -2:-1, :]]

            for token_idx in range(classname_tokens.shape[1]):
                _lang_x = classname_tokens[:, token_idx].reshape((-1, 1))
                with torch.no_grad():
                    outputs = model(
                        vision_x=None,
                        lang_x=_lang_x,
                        clear_conditioned_layers=False,
                        use_cached_vision_x=True,
                        past_key_values=(
                            past_key_values if token_idx > 0 else precomputed_pkvs
                        ),
                        use_cache=True,
                    )
                past_key_values = _detach_pkvs(outputs.past_key_values)
                elementwise_logits.append(outputs.logits.detach())

            # logits/probs has shape [B, classname_tokens + 1, vocab_size]
            logits = torch.concat(elementwise_logits, 1)
            probs = torch.softmax(logits, dim=-1).detach()
            # collect the probability of the generated token -- probability
            # at index 0 corresponds to the token at index 1.
            probs = probs[:, :-1, :]  # shape [B, classname_tokens, vocab_size]

            gen_probs = torch.gather(probs, 2, classname_tokens[:, :, None]).squeeze(-1)

            class_prob = torch.sum(torch.log(gen_probs), 1).detach().cpu().numpy()
            overall_probs.append(class_prob/classname_tokens.shape[1])

        overall_probs = np.row_stack(overall_probs).T  # shape [B, num_classes]

        def topk(probs_ary: np.ndarray, k: int) -> np.ndarray:
            """Return the indices of the top k elements in probs_ary."""
            return np.argsort(probs_ary)[::-1][:k]

        for i in range(vision_x.shape[0]):
            top5 = [
                class_map_cls_id_to_class[image_cls_dataset_name][pred]
                for pred in topk(overall_probs[i], 5)
            ]

            y_i = batch[i]["class_name"]
            acc5 += int(y_i in set(top5))
            acc1 += int(y_i == top5[0])

            print(
                f"DEBUG: batch {batch_idx} elem {i} of {vision_x.shape[0]}:"
                f"label {y_i} // top5 {top5}"
            )

        examples_seen = batch_idx * batch_size + vision_x.shape[0]
        print(
            "eval {}/{}: acc@1 ({}), acc@5 ({})".format(
                examples_seen, num_samples, acc1 / examples_seen, acc5 / examples_seen
            )
        )
        if examples_seen >= num_samples:
            break

    return float(acc1) / num_samples




if __name__ == "__main__":
    main()
