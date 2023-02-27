from contextlib import suppress

import torch
from einops import rearrange
from tqdm import tqdm
import time


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def train_one_epoch(
    args,
    model,
    epoch,
    laion_loader,
    pile_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    bloom_filter
):
    num_batches_per_epoch_laion = laion_loader.num_batches
    num_batches_per_epoch_pile = pile_loader.num_batches

    print(f"Number of batches per epoch in laion: {num_batches_per_epoch_laion}")
    print(f"Number of batches per epoch in pile: {num_batches_per_epoch_pile}")

    # assert num_batches_per_epoch_laion == num_batches_per_epoch_pile, "Number of batches in laion and pile datasets must be the same"
    num_batches_per_epoch = num_batches_per_epoch_pile

    print(f"Number of batches per epoch: {num_batches_per_epoch}")

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = AverageMeter() # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter() # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    num_collisions = 0

    # loop through dataloader
    for num_steps, (batch_laion, batch_pile) in tqdm(
        enumerate(zip(laion_loader, pile_loader)), disable=args.rank != 0
    ):
        data_time_m.update(time.time() - end) 

        global_step = num_steps + epoch * num_batches_per_epoch

        #### LAION FORWARD PASS ####
        images = (
            batch_laion[0]
            .to(device_id, dtype=cast_dtype, non_blocking=True)
            .unsqueeze(1)
            .unsqueeze(1)
        )

        input_ids = batch_laion[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_laion[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with autocast():
            loss_laion = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        divided_loss_laion = loss_laion / args.gradient_accumulation_steps

        #### C4 FORWARD PASS ####
        images = batch_pile[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2)
        input_ids = torch.stack([x[0] for x in batch_pile[1]]).squeeze(1)
        attention_mask = torch.stack([x[1] for x in batch_pile[1]]).squeeze(1)
        urls = batch_pile[2]
        
        # add urls to bloom filter if not already present
        for url in urls:
            if url in bloom_filter:
                num_collisions += 1
            else:
                bloom_filter.add(url)
        
        # NOTE: irena: expected shape of clip_text_input_ids / attention_mask is (N, I, max_seq_len)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100

        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1
            
            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while token_idx < labels.shape[1] and labels[i][token_idx] != media_token_id:
                    labels[i][token_idx] = -100
                    token_idx += 1

        # print("labels: ", labels[0])

        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with autocast():
            loss_pile = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
            
            # if loss is nan, skip this batch
            if torch.isnan(loss_pile):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad()
                continue
            
        divided_loss_pile = loss_pile / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_laion * args.loss_multiplier_laion
            + divided_loss_pile * args.loss_multiplier_pile
        )
        loss.backward()
        
        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                laion_samples_per_second = args.gradient_accumulation_steps * args.batch_size_laion * args.world_size / step_time_m.val
                laion_samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size_laion / step_time_m.val

                c4_samples_per_second = args.gradient_accumulation_steps * args.batch_size_c4 * args.world_size / step_time_m.val
                c4_samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size_c4 / step_time_m.val

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg, 
                        "laion_samples_per_second": laion_samples_per_second,
                        "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                        "c4_samples_per_second": c4_samples_per_second,
                        "c4_samples_per_second_per_gpu": c4_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]['lr'], 
                    }, 
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                colision_percentage = (num_collisions / (args.batch_size_c4 * args.gradient_accumulation_steps)) * 100
                wandb.log({"c4_batch_collision_percentage": colision_percentage}, commit=False)
                num_collisions = 0
                
                wandb.log(
                    {
                        "loss_laion": divided_loss_laion.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )
                wandb.log(
                    {"loss_pile": divided_loss_pile.item(), "global_step": global_step},
                    commit=True,
                )
                


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count