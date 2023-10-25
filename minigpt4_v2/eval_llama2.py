import sys
sys.path.insert(0, '../')

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from minigpt4_v2.common.config import Config
from minigpt4_v2.common.dist_utils import get_rank
from minigpt4_v2.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
from transformers import StoppingCriteria, StoppingCriteriaList
from models.prompt_learning import PromptLearner

import re

def split_string(s, delimiters):
    pattern = '|'.join(map(re.escape, delimiters))
    split_string = re.split(pattern, s)
    return split_string

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def _detach_pkvs(pkvs):
    """Detach a set of past key values."""
    return tuple([tuple([x.detach() for x in inner]) for inner in pkvs])



class MiniGPT4_llama2():
    def __init__(self, model_config, vis_processor_cfg, gpu_id, learnable_prompt=0):
        self.model_config = model_config
        self.model_config.device_8bit = gpu_id
        self.device = 'cuda:{}'.format(gpu_id)
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device)

        # vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        if learnable_prompt > 0:
            self.prompt_learner = PromptLearner(learnable_prompt, 5120, dtype = torch.float32)
        else:
            self.prompt_learner = None

    @staticmethod
    def create_prompt(system, sep, messages):
        prompt = system + sep
        for role, message in messages:
            if message:
                prompt += role + ": " + message + sep
            else:
                prompt += role + ":"
        return prompt

    @staticmethod
    def expand_emb(emb, batch_size):
        return emb.tile((batch_size, 1,  1))

    def process_output(self, input_text: str):
        if input_text[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            input_text = input_text[1:]
        if input_text[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            input_text = input_text[1:]

        output_text = self.model.llama_tokenizer.decode(input_text, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        return output_text

    def get_outputs(self, batch_images,
                    system = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.",
                    sep = "", prompt= 'describe the image as detailed as possible', task='',
                    max_new_tokens=200, num_beams=1, do_sample=True, min_length=1, top_p=0.9, repetition_penalty=1.0,
                    length_penalty=1, temperature=1.0):
        batch_images = batch_images.to(self.device)
        image_emb, _ = self.model.encode_img(batch_images)

        roles = ("<s>[INST] ", " [/INST] ")

        messages= [(roles[0], "<Img><ImageHere></Img> %s %s" % (task, prompt)), (roles[1], None)]
        sentence = self.create_prompt(system, sep, messages)

        sentence_segs = sentence.split('<ImageHere>')

        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(sentence_segs)
        ]

        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        embs = [self.expand_emb(seg_embs[0], batch_images.shape[0]), image_emb, self.expand_emb(seg_embs[1], batch_images.shape[0])]
        mixed_embs = torch.cat(embs, dim=1)

        outputs = self.model.llama_model.generate(
            inputs_embeds=mixed_embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature
        )

        new_predictions = [self.process_output(output) for output in outputs]

        return new_predictions

    def get_GPTScore(self, batch_images, class_names,
                    system = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.",
                    sep = "", prompt= 'a photo of ',  task='',):
        batch_images = batch_images.to(self.device)
        image_emb, _ = self.model.encode_img(batch_images)

        roles = ("<s>[INST] ", " [/INST] ")
        prefix_length = None
        prefix_2nd_token = None
        class_probs = []
        prefix_outputs = None
        # class_names = ['tench', 'sink']
        for class_name in class_names:
            if task != '':
                messages= [(roles[0], "<Img><ImageHere></Img> %s %s" % (task, prompt)), (roles[1], class_name)]
            else:
                messages= [(roles[0], "<Img><ImageHere></Img> %s" % (prompt)), (roles[1], class_name)]

            sentence = self.create_prompt(system, sep, messages)
            sentence_segs = sentence.split('<ImageHere>')

            seg_tokens = [
                self.model.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(sentence_segs)
            ]
            #
            seg_2nd_token = seg_tokens[1]
            import pdb
            pdb.set_trace()
            seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
            embs = [self.expand_emb(seg_embs[0], batch_images.shape[0]), image_emb, self.expand_emb(seg_embs[1], batch_images.shape[0])]
            mixed_embs = torch.cat(embs, dim=1)

            overall_length = mixed_embs.shape[1]
            if prefix_length is None or prefix_2nd_token is None or prefix_outputs is None:
                if task != '':
                    prefix_messages = [(roles[0], "<Img><ImageHere></Img> %s %s" % (task, prompt)), (roles[1], None)]
                else:
                    prefix_messages = [(roles[0], "<Img><ImageHere></Img> %s" % (prompt)), (roles[1], None)]
                prefix_sentence = self.create_prompt(system, sep, prefix_messages)

                prefix_sentence_segs = prefix_sentence.split('<ImageHere>')

                prefix_seg_tokens = [
                    self.model.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
                    # only add bos to the first seg
                    for i, seg in enumerate(prefix_sentence_segs)
                ]
                prefix_2nd_token = prefix_seg_tokens[1]
                prefix_seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in prefix_seg_tokens]
                prefix_embs = [self.expand_emb(prefix_seg_embs[0], batch_images.shape[0]), image_emb,
                        self.expand_emb(prefix_seg_embs[1], batch_images.shape[0])]
                prefix_mixed_embs = torch.cat(prefix_embs, dim=1)

                prefix_length = prefix_mixed_embs.shape[1]
                with torch.no_grad():
                    prefix_outputs = self.model.llama_model(
                        inputs_embeds=prefix_mixed_embs, use_cache=True
                    )
                    precomputed_pkvs = _detach_pkvs(prefix_outputs.past_key_values)

                    prefix_output_logits = prefix_outputs.logits.detach()
            assert torch.all(torch.eq(seg_2nd_token[: , : prefix_2nd_token.shape[1]], prefix_2nd_token))
            # compute the length before the grad_truth location
            with torch.no_grad():
                outputs = self.model.llama_model(
                    inputs_embeds=mixed_embs[:, prefix_length:], past_key_values=precomputed_pkvs
                )
                outputs_logits = torch.cat([prefix_output_logits, outputs.logits], dim=1)
                probs = torch.log_softmax(outputs_logits, dim=-1).detach()

            probs = probs[:, :-1, :]
            probs = probs[:, -(overall_length-prefix_length):, :]
            input_ids = seg_2nd_token[:, prefix_2nd_token.shape[1]:]
            input_ids = input_ids.tile((batch_images.shape[0],  1))
            assert probs.shape[1] == input_ids.shape[1]
            gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

            class_prob = gen_probs.mean(dim=-1)
            class_probs.append(class_prob)

        class_probs = torch.stack(class_probs, dim=-1)

        return class_probs


    def get_prompt_outputs(self, batch_images,
                    system="Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.",
                    sep="###",
                    max_new_tokens=200, num_beams=1, do_sample=True, min_length=1, top_p=0.9, repetition_penalty=1.0,
                    length_penalty=1, temperature=1.0):
        batch_images = batch_images.to(self.device)
        image_emb, _ = self.model.encode_img(batch_images)

        prompt = self.prompt_learner()

        messages = [("Human", "<Img><ImageHere></Img> %s" % '<PromptHere>'), ("Assistant", None)]
        sentence = self.create_prompt(system, sep, messages)

        sentence_segs = split_string(sentence, ['<ImageHere>', '<PromptHere>'])

        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(sentence_segs)
        ]

        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        embs = [self.expand_emb(seg_embs[0], batch_images.shape[0]), image_emb,
                self.expand_emb(seg_embs[1], batch_images.shape[0]), self.expand_emb(prompt, batch_images.shape[0]),
                self.expand_emb(seg_embs[2], batch_images.shape[0])]
        mixed_embs = torch.cat(embs, dim=1)

        outputs = self.model.llama_model.generate(
            inputs_embeds=mixed_embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature
        )

        new_predictions = [self.process_output(output) for output in outputs]

        return new_predictions


# def parse_args():
#     parser = argparse.ArgumentParser(description="Demo")
#     parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
#     parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override some settings in the used config, the key-value pair "
#         "in xxx=yyy format will be merged into config file (deprecate), "
#         "change to --cfg-options instead.",
#     )
#     args_list = ["--cfg-path", "minigpt4/eval_configs/minigpt4_eval.yaml", "--gpu-id", 0]
#     args = parser.parse_args(args_list)
#     return args


# args = parse_args()
# cfg = Config(args)
#
# model = MiniGPT4(cfg.model_cfg, args.gpu_id)
# vis_processor = model.vis_processor
#
#
# img_path = "/projectnb/ivc-ml/sunxm/code/MiniGPT-4/test_examples/test1.png"
#
# raw_image = Image.open(img_path).convert('RGB')
# image = vis_processor(raw_image).unsqueeze(0).to('cuda:{}'.format(args.gpu_id))
#
# outputs = model.get_outputs(batch_images=image)
#
# print(outputs)




