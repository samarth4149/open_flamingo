import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
from llava_v1_5.conversation import conv_templates, SeparatorStyle
from llava_v1_5.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava_v1_5.model import *
from llava_v1_5.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
import warnings
import shutil
from llava_v1_5.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from tqdm import tqdm
import copy

IGNORE_INDEX = -100

def _detach_pkvs(pkvs):
    """Detach a set of past key values."""
    return tuple([tuple([x.detach() for x in inner]) for inner in pkvs])


class LLaVA_v1_5():
    def __init__(self, model_path, model_name,  model_base=None):
        disable_torch_init()

        self.tokenizer, self.model, self.image_processor, context_len = self.load_pretrained_model(model_path, model_base, model_name)
        self.device = self.model.device


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
                    model.tie_weights()
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
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # get system message
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # inputs = self.tokenizer([prompt])
        # input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        return input_ids, stop_str, stopping_criteria
    
    def get_prompt(self, query, response=None):
        qs = query
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # get system message
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], response)
        prompt = conv.get_prompt()

        return prompt

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

        for class_name in tqdm(class_names):
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
    
    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def get_GPTScore1(self, batch_images, batch_captions, 
                     prompt='Describe the image in detail.',):
        batch_images = batch_images['pixel_values'][0]
        class_probs = []
        
        # prefix_input_ids, stop_str, stopping_criteria = self.encode_prompt(prompt)
        # prefix_input_ids = prefix_input_ids.tile((batch_images.shape[0], 1))
        # with torch.inference_mode():
        #     precomputed_output = self.model(
        #         prefix_input_ids,
        #         images=batch_images.to(dtype=torch.float16).cuda(),
        #         use_cache=True
        #     )
        #     precomputed_pkvs = _detach_pkvs(precomputed_output.past_key_values)
            
        
        
        for captions in batch_captions:
            # input_ids = [self.encode_prompt(prompt)[0].squeeze() for c in captions]
            input_ids = [tokenizer_image_token(self.get_prompt(prompt, c), self.tokenizer, return_tensors='pt') for c in captions]
            labels = copy.deepcopy(input_ids)
            
            tokenized_len = len(tokenizer_image_token(self.get_prompt(prompt), self.tokenizer))
            if prompt[-1] == " ":
                tokenized_len -= 1 # because white space
            for label in labels:
                label[:tokenized_len] = IGNORE_INDEX
                label[-1] = IGNORE_INDEX # also ignore eos token
            # labels[:, :tokenized_len] = IGNORE_INDEX

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                     batch_first=True,
                                                     padding_value=IGNORE_INDEX)
            
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]   
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
            batch_images = batch_images.to(device=self.device, dtype=torch.float16)
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.model.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                None,
                labels,
                batch_images
            )
            
            assert input_ids is None, "input_ids should be None for LLaVA-1.5"
            assert past_key_values is None, "past_key_values should be None for LLaVA-1.5"
            model_input_kwargs = {
                'input_ids': input_ids, # None for LLaVA-1.5
                'attention_mask': attention_mask,
                # 'images': batch_images,
                # 'labels': labels,
                'past_key_values': past_key_values,
                'inputs_embeds': inputs_embeds,
                'use_cache': None,
                'output_attentions': None,
                'output_hidden_states': None,
                # 'return_dict': True,
                'return_dict': False,
            }
            
            outputs = self.model.model(
                **model_input_kwargs
            )

            # output_logits = outputs.logits
            # probs = torch.log_softmax(output_logits, dim=-1)
            # labels[labels < 0] = 0

            hidden_states = outputs[0]
            output_logits = self.model.lm_head(hidden_states)

            # # Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # # Flatten the tokens
            # loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            # shift_labels = shift_labels.to(shift_logits.device)
            # class_prob = torch.zeros(shift_logits.shape[0])
            # for k in range(class_prob.shape[0]):
            #     class_prob[k] = (-loss_fct(shift_logits[k], shift_labels[k]))
            # class_probs.append(class_prob)
            
            # inputs = self.tokenizer(captions, return_tensors="pt", padding='longest')
            # input_ids = inputs.input_ids[:, 1:].cuda() # remove bos token
            # with torch.inference_mode():
            #     outputs = self.model(
            #         input_ids,
            #         past_key_values= precomputed_pkvs,
            #     )
            #     # outputs_logits = torch.cat([prefix_output_logits, outputs.logits], dim=1)
            #     output_logits = outputs.logits
            #     probs = torch.log_softmax(output_logits, dim=-1).detach()

            # probs = probs[:, -labels.shape[1]:, :]
            output_logits = output_logits[:, -labels.shape[1]:, :]
            shifted_logits = output_logits[:, :-1, :] # removes last token
            shifted_labels = labels[:, 1:]
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            class_prob = torch.zeros(shifted_logits.shape[0])
            for k in range(class_prob.shape[0]):
                class_prob[k] = (-loss_fct(shifted_logits[k], shifted_labels[k]))
            
            # labels = input_ids[:, 1:]
            # # probs = probs[:, -class_name_token_len:, :]
            # assert probs.shape[1] == labels.shape[1]
            # gen_probs = torch.gather(shifted_probs, 2, shifted_labels[:, :, None]).squeeze(-1)
            
            # # remove padding tokens
            # # non_pad_locs = (input_ids != self.tokenizer.pad_token_id) & (input_ids != self.tokenizer._convert_token_to_id('.'))
            # non_pad_locs = (shifted_labels != self.tokenizer.pad_token_id)
            # class_prob = (gen_probs * non_pad_locs).sum(dim=-1).div(non_pad_locs.sum(dim=-1))
            class_probs.append(class_prob)
            
        class_probs = torch.stack(class_probs, dim=-1)

        return class_probs
    
