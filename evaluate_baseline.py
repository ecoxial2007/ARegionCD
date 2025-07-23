import os
import json
import numpy as np
import argparse
import json
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate import DataLoaderConfiguration
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig
)

from src.datasets.region_vqa import RegionVQADataset


@torch.no_grad()
def evaluate(model, processor, eval_dataset, args, disable_tqdm=False):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f"cuda:{local_rank}")

    model.eval()
    save_info_list = []
    acc = []

    print(f"evaluating on {len(eval_dataset)} examples")
    for i in tqdm(range(len(eval_dataset)), disable=(rank != 0) or disable_tqdm):
        example = eval_dataset[i]
        print(example)

        image = example['image']
        question = example['question']

        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}',
        }

        prompt = processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors='pt'
        ).to(f'cuda:{local_rank}')

        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )

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
            generated_token_id = generated_ids.sequences[:, inputs['input_ids'].size(1) + len(token_probs)]
            token_prob = probs[:, generated_token_id].item()
            token_probs.append(token_prob)

        for idx, prob in enumerate(token_probs):
            token = processor.decode(generated_ids.sequences[:, inputs['input_ids'].size(1) + idx])
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

    args = parser.parse_args()

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