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


def process_output(input_text: str):
    if input_text[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        input_text = input_text[1:]
    if input_text[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        input_text = input_text[1:]

    output_text = model.llama_tokenizer.decode(input_text, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    return output_text


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args_list = ["--cfg-path", "minigpt4/eval_configs/minigpt4_eval.yaml", "--gpu-id", 0]
    args = parser.parse_args(args_list)
    return args


args = parse_args()
cfg = Config(args)


model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [torch.tensor([835]).to('cuda:{}'.format(args.gpu_id)),
                          torch.tensor([2277, 29937]).to('cuda:{}'.format(args.gpu_id))]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

img_path = "/projectnb/ivc-ml/sunxm/code/MiniGPT-4/test_examples/test1.png"

raw_image = Image.open(img_path).convert('RGB')
image = vis_processor(raw_image).unsqueeze(0).to('cuda:{}'.format(args.gpu_id))

# prepare the input to llama
system = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions."
roles = ("Human", "Assistant")
messages = []
offset = 2
sep = "###"

image_emb, _ = model.encode_img(image)
image_list = [image_emb]
messages.append(("Human", "<Img><ImageHere></Img>"))
messages[0] = ("Human", "<Img><ImageHere></Img> %s" % 'describe the image as detailed as possible' )
messages.append(["Assistant", None])


prompt = system + sep
for role, message in messages:
    if message:
        prompt += role + ": " + message + sep
    else:
        prompt += role + ":"

prompt_segs = prompt.split('<ImageHere>')

seg_tokens = [
            model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to('cuda:{}'.format(args.gpu_id)).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]

seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
import pdb
pdb.set_trace()
embs = [seg_embs[0],  image_emb,  seg_embs[1]]
mixed_embs = torch.cat(embs, dim=1)

current_max_len = embs.shape[1] + 300

outputs = model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=200,
            stopping_criteria=stopping_criteria,
            num_beams=1,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
        )

# take the first example in the batch
output_token = outputs[0]
print(process_output(output_token))





