"""
LoRA fine-tuning of Qwen2.5-VL on the sensor activity-identification JSONL from
``data/sensor/export_sft.py``. Only the language-model layers get adapters; the
vision tower stays frozen. Loss is computed on the assistant answer only.

Usage (on a GPU node, from the repo root):
    python -m data.sensor.train_lora --train sensor_work/sft/training_image.jsonl \
        --output_dir sensor_work/lora/qwen2_5_vl_7b_image
"""

import argparse
import json
import math

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments

from data.sensor.export_sft import IMAGE_TOKEN

# Language-model blocks are ``layers.N``; the vision tower uses ``blocks.N``, so it is excluded.
LM_LORA_TARGETS = r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_chat(record: dict, min_pixels: int, max_pixels: int, with_answer: bool) -> list[dict]:
    user, answer = record["messages"][0]["content"], record["messages"][1]["content"]
    content = [{"type": "image", "image": p, "min_pixels": min_pixels, "max_pixels": max_pixels} for p in record["images"]]
    content.append({"type": "text", "text": user.replace(IMAGE_TOKEN, "")})
    chat = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
    if with_answer:
        chat.append({"role": "assistant", "content": answer})
    return chat


class Collator:
    def __init__(self, processor, min_pixels: int, max_pixels: int):
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __call__(self, batch: list[dict]) -> dict:
        from qwen_vl_utils import process_vision_info

        assert len(batch) == 1, "use per_device_train_batch_size=1 with gradient accumulation"
        record = batch[0]
        full = to_chat(record, self.min_pixels, self.max_pixels, with_answer=True)
        prompt = to_chat(record, self.min_pixels, self.max_pixels, with_answer=False)
        full_text = self.processor.apply_chat_template(full, tokenize=False)
        prompt_text = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        images = process_vision_info(full)[0] if record["images"] else None

        inputs = self.processor(text=[full_text], images=images, return_tensors="pt")
        prompt_len = self.processor(text=[prompt_text], images=images, return_tensors="pt").input_ids.shape[1]
        labels = inputs.input_ids.clone()
        labels[:, :prompt_len] = -100
        inputs["labels"] = labels
        return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1605632)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=LM_LORA_TARGETS, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(args.base, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    train_data = load_jsonl(args.train)
    total_steps = math.ceil(len(train_data) / args.grad_accum * args.epochs)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=max(1, int(0.05 * total_steps)),
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=train_data,
        data_collator=Collator(processor, args.min_pixels, args.max_pixels),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
