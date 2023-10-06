import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F

_tokenizer = _Tokenizer()

__all__ = ['dualcoop', 'DualCoop']



class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP_MLC.N_CTX
        ctx_init = cfg.TRAINER.COOP_MLC.PROMPT_INIT.strip()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init .split(" "))
            prompt = clip.tokenize(ctx_init )
            with torch.no_grad():
                embedding  = clip_model.token_embedding(prompt ).type(dtype)
            ctx_vectors = embedding [0, 1: 1 + n_ctx , :]
            prompt_prefix = ctx_init
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors , std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx )

        print(f'Initial positive context: "{prompt_prefix }"')
        print(f"Number of positive context words (tokens): {n_ctx }")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = []
        for p  in prompts :
            tokenized_prompts .append(clip.tokenize(p))
        tokenized_prompts = torch.cat(tokenized_prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix ", embedding [:, :1, :] )
        self.register_buffer("token_suffix ", embedding [:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        tokenized_prompts = tokenized_prompts 
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        # shape [len_of_token, dim_of_each_token]
        ctx = ctx.unsqueeze(0)

        prefix  = self.token_prefix
        suffix  = self.token_suffix

        prompts  = torch.cat(
            [
                prefix ,  # (n_cls, 1, dim)
                ctx ,  # (n_cls, n_ctx, dim)
                suffix ,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts = torch.cat([prompts_neg, prompts ], dim=0)

        if cls_id is not None:
            tokenized_prompts  = self.tokenized_prompts[self.n_cls:][cls_id]
            tokenized_prompts_neg = self.tokenized_prompts[:self.n_cls][cls_id]
            tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts ], dim=0)
        else:
            tokenized_prompts = self.tokenized_prompts


        return prompts, tokenized_prompts


class DualCoop(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.visual_encoder_type = cfg.MODEL.BACKBONE.NAME
        self.prompt_learner = MLCPromptLearner(cfg, classnames, clip_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = cfg.TRAINER.COOP_MLC.LS
        self.dtype = clip_model.dtype
        self.cfg = cfg

    def forward(self, image, cls_id=None):
        # get image and text features
        image_features, attn_weights = self.image_encoder(image.type(self.dtype))
        prompts, tokenized_prompts = self.prompt_learner(cls_id)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)

        # Class-Specific Region Feature Aggregation
        output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])
        b, c, _ = output.shape
        output_half = output[:,  c // 2:]
        w_half = F.softmax(output_half, dim=-1)
        w = torch.cat([w_half, w_half], dim=1)
        output = 5 * (output * w).sum(-1)

        b, c = output.shape

        # convert the shape of logits to [b, 2, num_class]
        logits = output.resize(b, 2, c//2)

        return logits

    @property
    def network_name(self):
        name = ''
        name += 'DualCoop-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params


def dualcoop(cfg, classnames, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building dualcoop")
    model = DualCoop(cfg, classnames, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    return model