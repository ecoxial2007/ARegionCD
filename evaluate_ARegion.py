import os
import numpy as np
import argparse
import json
import torch
import torch.nn.functional as F
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig
)

from src.datasets.region_vqa import RegionVQADataset
from src.highlighter_modules.guidance import ProbCFGLogitsProcessor


def insert_separator(X, sep_list):
    if len(X) > len(sep_list):
        sep_list.append([])
    return [ele for sublist in zip(X, sep_list) for ele in sublist]


def generate_bbox_mask(mask_crop, n_rows, n_cols,
                       block_dims=(12, 12), tokens_per_block=None):
    """
    Generates a 1D mask sequence following a (local + separator + global) structure
    from a high-resolution binary mask. This version handles non-square aspect ratios.

    Args:
        mask_crop (torch.Tensor): A high-resolution binary mask of shape [B, H, W].
                                  For example, (2, 480, 640).
        n_rows (int): The number of grid rows for the local views.
        n_cols (int): The number of grid columns for the local views.
        block_dims (tuple, optional): The dimensions (height, width) of the feature map for each sub-patch.
                                      Defaults to (12, 12). This replaces sep_tokens to support non-square blocks.
        tokens_per_block (int, optional): The total number of tokens per block. If provided, it's used for validation.
                                          Defaults to None.

    Returns:
        torch.Tensor: The final 1D mask tensor of shape [B, total_tokens].
    """
    # --- Step 1: Parameter Interpretation and Validation ---
    if mask_crop.dim() != 3:
        raise ValueError(f"Input mask_crop should have dimensions [B, H, W], but received {mask_crop.dim()} dimensions")

    B, H, W = mask_crop.shape
    device = mask_crop.device

    # MODIFIED: Use separate height and width for blocks instead of a single side length.
    block_side_h, block_side_w = block_dims

    # MODIFIED: Validation now checks against the product of block dimensions.
    if tokens_per_block is not None and tokens_per_block != block_side_h * block_side_w:
        warnings.warn(
            f"tokens_per_block ({tokens_per_block}) does not match the product of block_dims {block_dims}."
            f"Calculations will be based on block_dims={block_dims} as the geometric benchmark."
        )

    # --- Step 2: Construct the Mask Sequence for the Local View ---
    # 2.1 Downsample the mask to the total feature map dimensions of the local view.
    # MODIFIED: Use block_side_h and block_side_w for calculation.
    target_h_local = n_rows * block_side_h
    target_w_local = n_cols * block_side_w

    # interpolate requires 4D input [B, C, H, W], so add a channel dimension first
    local_feature_mask = F.interpolate(
        mask_crop.unsqueeze(1).float(),
        size=(target_h_local, target_w_local),  # This works correctly with non-square sizes
        mode='nearest'
    ).squeeze(1)  # Remove the channel dimension after calculation

    # 2.2 Insert newlines (with value 0) into the local feature mask.
    # The logic remains the same, but the dimensions are now based on the new target sizes.
    newline_local = torch.zeros((B, target_h_local, 1), device=device, dtype=torch.long)
    local_with_newlines = torch.cat([local_feature_mask.long(), newline_local], dim=2)
    local_mask_sequence = local_with_newlines.view(B, -1)

    # --- Step 3: Construct the Mask Sequence for the Global View ---
    # 3.1 Downsample the mask to the feature map dimensions of the global view.
    # MODIFIED: The global view now also respects the block's aspect ratio.
    target_h_global = block_side_h
    target_w_global = block_side_w

    global_feature_mask = F.interpolate(
        mask_crop.unsqueeze(1).float(),
        size=(target_h_global, target_w_global),  # This resizes to the non-square block dimensions
        mode='nearest'
    ).squeeze(1)

    # 3.2 Insert newlines (with value 0) into the global feature mask.
    newline_global = torch.zeros((B, target_h_global, 1), device=device, dtype=torch.long)
    global_with_newlines = torch.cat([global_feature_mask.long(), newline_global], dim=2)
    global_mask_sequence = global_with_newlines.view(B, -1)

    # --- Step 4: Construct the Separator Mask with value 0 ---
    # (No changes needed here)
    separator_mask = torch.zeros((B, 1), device=device, dtype=torch.long)

    # --- Step 5: Concatenate to get the final mask sequence ---
    # (No changes needed here)
    final_mask = torch.cat(
        [local_mask_sequence, separator_mask, global_mask_sequence],
        dim=1
    )

    return final_mask


@torch.no_grad()
def evaluate(model, processor, eval_dataset, args, disable_tqdm=False):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f"cuda:{local_rank}")
    model.eval()

    sqrt_num = int(np.sqrt(args.num_crops))
    n_rows, n_cols = sqrt_num, sqrt_num
    if 'OBScan' in args.img_root and n_rows == 4:
        n_rows = 3

    save_info_list = []
    acc = []

    print(f"evaluating on {len(eval_dataset)} examples")
    for i in tqdm(range(len(eval_dataset)), disable=(rank != 0) or disable_tqdm):
        example = eval_dataset[i]
        image = example['image']
        image_mask = example['image_mask']
        mask_np = np.array(image_mask)
        question = example['question']
        qs_highlighted_parts = example['highlights']
        bbox = example.get("bbox", (0.0, 0.0, 0.0, 0.0))
        mask_value = example['mask_value']
        attribute = example.get("attribute", None)

        if args.use_text:
            add_info = ', '.join(qs_highlighted_parts)
            add_info = ' Object: ' + add_info
        else:
            add_info = ''

        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}{add_info}',
        }
        prompt = processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=prompt,
            images=[image],
            images_mask=[mask_np],
            bboxes=[bbox],
            mask_values=[mask_value],
            attribute=[attribute],
            qs_highlighted_parts=qs_highlighted_parts,
            return_tensors='pt'
        )

        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        prompt_chunks = inputs["prompt_chunks"]
        image_ids_pad = inputs["image_ids_pad"]

        bbox_attention_mask_crop = inputs["bbox_attention_mask"]

        bbox_attention_mask = generate_bbox_mask(bbox_attention_mask_crop, n_rows, n_cols).tolist()  ### for image
        highlight_attention_mask = inputs["highlight_attention_mask"]  ### for text

        offset = 0
        input_ids = []
        combined_highlight_mask = []

        for tokens, mask in zip(insert_separator(prompt_chunks, image_ids_pad),
                                insert_separator(highlight_attention_mask,
                                                 bbox_attention_mask)):  # Use zero_mask_padding here
            input_ids.extend(tokens[offset:])
            combined_highlight_mask.extend(mask[offset:])

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = (input_ids > -1000000).to(torch.long)
        combined_highlight_mask = torch.tensor(combined_highlight_mask, dtype=torch.long, device=device).unsqueeze(0)

        inputs['input_ids'] = input_ids

        hl_mask_ = combined_highlight_mask
        hl_mask_[hl_mask_ == 1] = args.perturb_weight
        hl_mask_[hl_mask_ == 0] = args.attn

        cfg_batched_input = input_ids.repeat(2, 1)
        pixel_values = inputs['pixel_values'].repeat(2, 1, 1, 1, 1)
        image_sizes = inputs['image_sizes'].repeat(2, 1)

        generated_ids = model.generate(
            input_ids=cfg_batched_input,
            pixel_values=pixel_values,
            attention_mask=torch.cat([attention_mask, hl_mask_], dim=0),
            image_sizes=image_sizes,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            logits_processor=[ProbCFGLogitsProcessor(guidance_scale=args.cfg, use_log=True)],
            output_scores=True,
            return_dict_in_generate=True
        )

        batch_index = 1
        generated_texts = processor.batch_decode(
            generated_ids.sequences[:, inputs['input_ids'].size(1):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prediction = generated_texts[0].strip().strip('.')

        print('\n\n\n-------------------------------------------')
        print('Question:', example['question'], 'GT:', example['answer'])
        print('Prediction:', prediction)
        print('-------------------------------------------\n\n\n')

        token_probs = []
        generated_tokens = []

        for i, scores in enumerate(generated_ids.scores):
            probs = torch.softmax(scores, dim=-1)
            generated_token_id = generated_ids.sequences[batch_index, inputs['input_ids'].size(1) + len(token_probs)]
            token_prob = probs[batch_index, generated_token_id].item()
            token_probs.append(token_prob)

        for idx, prob in enumerate(token_probs):
            token = processor.decode(generated_ids.sequences[batch_index, inputs['input_ids'].size(1) + idx])
            generated_tokens.append(token)

        answer = example['answer'][0] if type(example['answer']) is list else example['answer']
        generated_dict = {
            "image_id": example['image_id'],
            'question': example['question'],
            'answer': example['answer'],
            'prediction': prediction,
            'token_probs': token_probs,
            'token_preds': generated_tokens,
        }

        if answer.lower() in prediction.lower():
            generated_dict["acc"] = 1
            acc.append(1)
        else:
            generated_dict["acc"] = 0
            acc.append(0)

        save_info_list.append(generated_dict)

    save_info_list = gather_object(save_info_list)
    acc = gather_object(acc)

    if rank == 0 and args.save_path:
        with open(args.save_path, 'w') as f:
            save_dict = {
                'sample info': save_info_list,
                'acc': np.mean(acc)
            }
            json.dump(save_dict, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='./Phi-3.5-vision-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--use_text', action='store_true', help='Use Text')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--use_qlora', action='store_true', help='Use QLora')

    parser.add_argument('--lora_path', type=str, help='LoRA directory')
    parser.add_argument('--input_path', type=str, help='Question and Answer json path')
    parser.add_argument('--save_path', type=str, help='Save result json path')
    parser.add_argument('--img_root', type=str, help='Image Folder')

    parser.add_argument('--num_crops', type=int, default=16, help='Number of maximum image crops')
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')

    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--attn", type=float, default=3.0)
    parser.add_argument("--perturb_weight", type=float, default=0.01)

    args = parser.parse_args()
    args.attention_weight = args.attn

    assert args.num_crops <= 16, 'num_crops must be less than or equal to 16'
    if args.use_qlora:
        args.use_lora = True

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, num_crops=args.num_crops
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if args.use_flash_attention else 'eager',
    )

    if args.use_lora:
        model.load_adapter(args.lora_path)

    eval_dataset = RegionVQADataset(
        annotation_file=args.input_path,
        vis_root=args.img_root
    )

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = model.to(f'cuda:{local_rank}')
    evaluate(
        model,
        processor,
        eval_dataset,
        args,
        disable_tqdm=not args.tqdm,
    )


if __name__ == '__main__':
    main()