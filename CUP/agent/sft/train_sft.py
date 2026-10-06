"""
阶段 B：LoRA SFT 训练（服务器跑）。

数据：make_trajectories.py 产出的 JSONL（messages 格式与 rollout.py 推理一致，
图像以 base64 单独存放，user content 用 {"type": "image"} 占位）。
训练：冻结主干 + peft LoRA（r=32, alpha=64, 全线性层），
只对 assistant 部分算 next-token loss（其余 label 置 -100），bf16。

用法（服务器）：
  python agent/sft/train_sft.py \
      --base_model /workspace/cup-debug/models/Qwen2.5-VL-7B-Instruct \
      --data trajectories.jsonl --out_dir /workspace/cup-debug/models/qwen25vl_lora \
      --epochs 1 --batch_size 1 --grad_accum 32 --lr 1e-4
"""
import argparse
import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

ASSISTANT_MARK = "<|im_start|>assistant\n"


def load_trajectories(jsonl_path):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _tokenize_len(processor, text):
    """纯文本 tokenize 后的 token 数（图像占位 token 在文本中，长度与带图一致）。"""
    return len(processor(text=[text], return_tensors="pt")["input_ids"][0])


def encode_sample(processor, sample, max_pixels):
    """messages + 图像 → input_ids/labels（对每一段 assistant 内容算 loss）。"""
    messages = sample["messages"]
    image_bytes = base64.b64decode(sample["image_base64"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = processor(text=[text], images=[image], return_tensors="pt",
                         min_pixels=64 * 64, max_pixels=max_pixels)["input_ids"][0]

    # 多段 assistant 掩码：遍历每段 "<|im_start|>assistant\n ... <|im_end|>"，
    # 只对该段【内容】（不含 im_start/im_end 标记）算 loss
    labels = torch.full_like(full_ids, -100)
    pos = 0
    while True:
        start = text.find(ASSISTANT_MARK, pos)
        if start < 0:
            break
        content_start = start + len(ASSISTANT_MARK)
        end = text.find("<|im_end|>", content_start)
        if end < 0:
            end = len(text)
        seg_start = _tokenize_len(processor, text[:content_start])
        seg_end = _tokenize_len(processor, text[:end])
        labels[seg_start:seg_end] = full_ids[seg_start:seg_end]
        pos = end
    return {"input_ids": full_ids, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--max_pixels", type=int, default=256 * 256,
                    help="图像最大像素数（控制显存）")
    args = ap.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(args.base_model)

    # Qwen2.5-VL 的视觉 patch embedding 是 Conv3d（peft 不支持），
    # 按名字只对 Linear 层打 LoRA（语言主干 + 视觉/投影中的同名注意力层）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    samples = load_trajectories(args.data)
    print(f"训练样本数: {len(samples)}")
    dataset = [encode_sample(processor, s, args.max_pixels) for s in samples]

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"LoRA 已保存 -> {args.out_dir}")


if __name__ == "__main__":
    main()
