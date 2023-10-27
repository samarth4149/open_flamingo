import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def _detach_pkvs(pkvs):
    """Detach a set of past key values."""
    return tuple([tuple([x.detach() for x in inner]) for inner in pkvs])


class LLaVA():
    def __init__(self, model_name):
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        self.model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                      use_cache=True).cuda()
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

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
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs

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
        input_ids = input_ids.tile((batch_images.shape[0], 1))
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

    def get_GPTScore(self, batch_images, prompt, class_names):
        batch_images = batch_images['pixel_values'][0]
        class_probs = []
        prefix_input_ids, stop_str, stopping_criteria = self.encode_prompt(prompt)
        prefix_input_ids = prefix_input_ids.tile((batch_images.shape[0], 1))
        with torch.inference_mode():
            precomputed_output = self.model(
                prefix_input_ids,
                images=batch_images.half().cuda(),
                use_cache=True
            )
            precomputed_pkvs = _detach_pkvs(precomputed_output.past_key_values)

            prefix_output_logits = precomputed_output.logits.detach()

        for class_name in class_names:
            inputs = self.tokenizer([class_name])
            input_ids = torch.as_tensor(inputs.input_ids)[:, 1:].cuda()
            input_ids = input_ids.tile((batch_images.shape[0], 1))
            class_name_token_len = len(input_ids)
            with torch.inference_mode():
                outputs = self.model(
                    input_ids,
                    past_key_values= precomputed_pkvs,
                )
                import pdb
                pdb.set_trace()
                outputs_logits = torch.cat([prefix_output_logits, outputs.logits], dim=1)
                probs = torch.log_softmax(outputs_logits, dim=-1).detach()

            probs = probs[:, :-1, :]
            probs = probs[:, -class_name_token_len:, :]
            assert probs.shape[1] == input_ids.shape[1]
            gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

            class_prob = gen_probs.mean(dim=-1)
            class_probs.append(class_prob)

        class_probs = torch.stack(class_probs, dim=-1)

        return class_probs
