import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
from llava_v1_5.conversation import conv_templates, SeparatorStyle
from llava_v1_5.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava_v1_5.model import *
from llava_v1_5.mm_utils import KeywordsStoppingCriteria
import warnings
import shutil
from llava_v1_5.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN



def _detach_pkvs(pkvs):
    """Detach a set of past key values."""
    return tuple([tuple([x.detach() for x in inner]) for inner in pkvs])


class LLaVA_v1_5():
    def __init__(self, model_path, model_base, model_name):
        disable_torch_init()

        self.tokenizer, self.model, self.image_processor, context_len = self.load_pretrained_model(model_path, model_base, model_name)


    def load_pretrained_model(self, model_path, model_base, model_name, device_map="auto",
                              device="cuda"):
        kwargs = {"device_map": device_map}
        kwargs['torch_dtype'] = torch.float16

        if 'llava' in model_name.lower():
            # Load LLaVA model
            if 'lora' in model_name.lower() and model_base is None:
                warnings.warn(
                    'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
            if 'lora' in model_name.lower() and model_base is not None:
                lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print('Loading LLaVA from base model...')
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                              config=lora_cfg_pretrained, **kwargs)
                token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
                if model.lm_head.weight.shape[0] != token_num:
                    model.lm_head.weight = torch.nn.Parameter(
                        torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                    model.model.embed_tokens.weight = torch.nn.Parameter(
                        torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

                print('Loading additional LLaVA weights...')
                if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'),
                                                     map_location='cpu')
                else:
                    # this is probably from HF Hub
                    from huggingface_hub import hf_hub_download
                    def load_from_hf(repo_id, filename, subfolder=None):
                        cache_file = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            subfolder=subfolder)
                        return torch.load(cache_file, map_location='cpu')

                    non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                                       non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                           non_lora_trainables.items()}
                model.load_state_dict(non_lora_trainables, strict=False)

                from peft import PeftModel
                print('Loading LoRA weights...')
                model = PeftModel.from_pretrained(model, model_path)
                print('Merging LoRA weights...')
                model = model.merge_and_unload()
                print('Model is loaded...')
            elif model_base is not None:
                # this may be mm projector only
                print('Loading LLaVA from base model...')
                if 'mpt' in model_name.lower():
                    if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                        shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'),
                                        os.path.join(model_path, 'configuration_mpt.py'))
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                                config=cfg_pretrained, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                                  config=cfg_pretrained, **kwargs)

                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            else:
                if 'mpt' in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        else:
            # Load language model
            if model_base is not None:
                # PEFT model
                from peft import PeftModel
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16,
                                                             low_cpu_mem_usage=True, device_map="auto")
                print(f"Loading LoRA weights from {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
                print(f"Merging weights")
                model = model.merge_and_unload()
                print('Convert to FP16...')
                model.to(torch.float16)
            else:
                use_fast = False
                if 'mpt' in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                                 trust_remote_code=True, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

        image_processor = None

        if 'llava' in model_name.lower():
            mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
            if mm_use_im_patch_token:
                tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=device, dtype=torch.float16)
            image_processor = vision_tower.image_processor

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        return tokenizer, model, image_processor, context_len

    def encode_prompt(self, query):
        qs = query
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs

        # get system message
        conv = conv_templates["vicuna_v1"].copy()
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
            class_name_token_len = input_ids.shape[1]
            with torch.inference_mode():
                outputs = self.model(
                    input_ids,
                    past_key_values= precomputed_pkvs,
                )
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
