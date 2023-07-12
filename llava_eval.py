import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class LLaVA():
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                      use_cache=True).cuda()
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower, torch_dtype=torch.float16)
        import pdb
        pdb.set_trace()

        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        self.tokenizer .add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer .add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = self.model.get_model().vision_tower[0]
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16,
                                                       low_cpu_mem_usage=True).cuda()
        self.model.get_model().vision_tower[0] = vision_tower

        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        self.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    def encode_prompt(self, query):
        qs = query
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN

        # get system message
        conv = conv_templates["multimodal"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        return input_ids, stop_str, stopping_criteria

    def get_outputs(self, batch_images, prompt):
        input_ids, stop_str, stopping_criteria = self.encode_prompt(prompt)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=batch_images.half().cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        predictions = []
        for output in outputs:
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()
            predictions.append(output)

        return predictions

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model = LLaVA(os.path.expanduser(args.model_name))
    image_processor = model.image_processor


    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    predictions = model.get_outputs(image_tensor.unsqueeze(0).half().cuda(), args.query)
    print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
