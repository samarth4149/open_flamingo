from typing import List

from PIL import Image
import torch

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from open_flamingo.eval.eval_model import BaseEvalModel


class EvalModel(BaseEvalModel):
    """BLIP-2 model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "cpu"
    """

    def __init__(self, model_args):
        assert (
            "processor_path" in model_args
            and "lm_path" in model_args
            and "device" in model_args
        ), "BLIP-2 requires processor_path, lm_path, and device arguments to be specified"

        model_args["device"] = int(model_args["device"])

        self.device = model_args["device"] if model_args["device"] >= 0 else "cpu"
        self.processor = Blip2Processor.from_pretrained(model_args["processor_path"])
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_args["lm_path"]
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor.tokenizer.padding_side = "left"

    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, channels, height, width).
        """
        batch_images = None
        assert all(
            len(example) == 1 for example in batch
        ), "BLIP-2 only supports one image per example"

        for example in batch:
            assert len(example) == 1, "BLIP-2 only supports one image per example"
            batch_images = torch.cat(
                [
                    batch_images,
                    self.processor.image_processor(example, return_tensors="pt")[
                        "pixel_values"
                    ],
                ]
                if batch_images is not None
                else [
                    self.processor.image_processor(example, return_tensors="pt")[
                        "pixel_values"
                    ]
                ],
                dim=0,
            )
        return batch_images

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: torch.tensor,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        batch_images = batch_images['pixel_values'][0]
        encodings = self.processor.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        with torch.inference_mode():
            outputs = self.model.generate(
                # self._prepare_images(batch_images).to(self.device),
                batch_images.to(self.device),
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        return self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_GPTScore1(self, batch_images, batch_captions, prompt):
        batch_images = batch_images['pixel_values'][0]
        
        # TODO : implement without caching first
        class_probs = []
        for captions in batch_captions:
            inp_captions = [f"<image>{prompt} " for c in captions]
            
            encodings = self.processor.tokenizer(
                inp_captions,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=2000,
            )
            
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            
            encodings = self.processor.tokenizer(
                captions,
                padding="longest",
                # truncation=True,
                return_tensors="pt",
                max_length=2000,
            )
            decoder_input_ids = encodings["input_ids"].to(self.device)
            decoder_attention_mask = encodings["attention_mask"]
            
            with torch.inference_mode():
                outputs = self.model(
                    pixel_values=batch_images.to(self.device),
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask.to(self.device),
                    return_dict=True,
                )
                output_logits = outputs.logits
                probs = torch.log_softmax(output_logits, dim=-1).detach()
                
            gen_probs = torch.gather(probs, 2, decoder_input_ids[:, :, None]).squeeze(-1)
            
            # remove padding tokens
            # non_pad_locs = (input_ids != self.tokenizer.pad_token_id) & (input_ids != self.tokenizer._convert_token_to_id('.'))
            non_pad_locs = (decoder_input_ids != self.processor.tokenizer.pad_token_id)
            class_prob = (gen_probs * non_pad_locs).sum(dim=-1).div(non_pad_locs.sum(dim=-1))
            class_probs.append(class_prob)
        
        class_probs = torch.stack(class_probs, dim=-1)
        return class_probs

    def get_vqa_prompt(self, question, answer=None) -> str:
        return (
            f"Question:{question} Short answer:{answer if answer is not None else ''}"
        )

    def get_caption_prompt(self, caption=None) -> str:
        return f"A photo of {caption if caption is not None else ''}"

    def get_classification_prompt(self, class_str=None) -> str:
        raise NotImplementedError
