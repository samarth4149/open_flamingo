import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
from transformers import StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class MiniGPT4():
    def __init__(self, model_config, vis_processor_cfg, gpu_id):
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
                    sep = "###", prompt = 'describe the image as detailed as possible',
                    max_new_tokens=200, num_beams=1, do_sample=True, min_length=1, top_p=0.9, repetition_penalty=1.0,
                    length_penalty=1, temperature=1.0):
        batch_images = batch_images.to(self.device)
        image_emb, _ = self.model.encode_img(batch_images)

        messages= [("Human", "<Img><ImageHere></Img> %s" % prompt), ("Assistant", None)]
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




