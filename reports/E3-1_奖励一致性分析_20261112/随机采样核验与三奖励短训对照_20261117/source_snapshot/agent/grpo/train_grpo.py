"""
阶段 C：GRPO 检索奖励训练（手写轻量实现，无 veRL/TRL 依赖）。

设计（对应方案）：
  - 冷启动：从阶段 B 的 LoRA 检查点（lora_v2_all）初始化
  - 每步：抽 1 张训练图 → 组 G=8 采样（温度 1.0，单轮整段生成——
    SFT 后模型已学会输出含 tool_call/result/answer 的完整轨迹）
  - 奖励：R_format(轨迹文本) + 1/(1+rank)（rank 由冻结评分器计算，h_end = 最后 token）
  - 优势：组内标准化；loss = -mean(A * log π)，AdamW 更新 LoRA
  - 每步日志存（工具调用序列、R_retrieval、R_format）——阶段 D 四象限统计的原料

用法（服务器）：
  python agent/grpo/train_grpo.py \
      --base_model /workspace/cup-debug/models/Qwen2.5-VL-7B-Instruct \
      --lora_path /workspace/cup-debug/models/lora_v2_all \
      --train_arrow /workspace/cup-debug/data/ucm/ucm_caption_karpathy_label_train.arrow \
      --scorer_ckpt <RSP-CLIP 纯推理 checkpoint> \
      --val_arrow /workspace/cup-debug/data/ucm/ucm_caption_karpathy_label_val.arrow \
      --val_cache /workspace/cup-debug/grpo_val_cache.pt \
      --out_dir /workspace/cup-debug/models/lora_grpo \
      --steps 500 --group_size 8 --lr 5e-6 \
      --stats_log /workspace/cup-debug/grpo_stats.jsonl
"""
import argparse
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pyarrow as pa
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from agent.grpo.reward import total_reward
from agent.grpo.scorer import RetrievalScorer
from agent.rollout import SYSTEM_PROMPT

# 纯推理模式 prompt（阶段 A 原版：直接分析，无工具提示）
PLAIN_PROMPT = (
    "你是遥感图像分析助手。给定一张遥感图像和一条查询文本，"
    "请简要描述图像中与查询相关的内容。直接回答，不要输出思考过程。"
)

DEVICE = "cuda"


def sample_group(model, processor, image, caption, group_size, max_new_tokens=256,
                 sample_batch=None, few_shot=False):
    """单图组采样：G 条轨迹（无梯度）。返回 [(轨迹文本, h_end, 完整序列 ids)]。

    sample_batch：生成时每批条数（None = 一次生成全部）。显存不足时设 8，
    分批生成后拼接，采样结果与一次生成等价。
    few_shot：工具版 prompt 附加与评测 rollout 一致的 tool_call 协议示例，
    保证训练/评测 prompt 分布一致。
    带梯度的 logprob 由主循环逐条前向重算（逐条 backward 累加，控制显存峰值）。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if few_shot:
        # 与 agent/rollout.py rollout_one 的 few-shot 完全一致
        messages += [
            {"role": "user", "content": [{"type": "text", "text": "查询文本: There is a harbor with several ships."}]},
            {"role": "assistant", "content": "<tool_call>scene_classify</tool_call>"},
            {"role": "user", "content": "<result>harbor (0.61), port (0.24), ship (0.08)</result>"},
            {"role": "assistant", "content": "<answer>图中是一座港口，与查询文本相符。</answer>"},
        ]
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"查询文本: {caption}"},
        ],
    })
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = group_size if sample_batch is None else sample_batch

    seqs_list = []
    prompt_len = None
    with torch.no_grad():
        for start in range(0, group_size, batch):
            n = min(batch, group_size - start)
            inputs = processor(text=[text] * n, images=[image] * n,
                               return_tensors="pt").to(DEVICE)
            if prompt_len is None:
                prompt_len = inputs.input_ids.shape[1]
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=1.0, return_dict_in_generate=True,
            )
            seqs_list.append(outputs.sequences)
            del outputs, inputs
            torch.cuda.empty_cache()
    # 各批按批内最长 pad，跨批长度可能不同：统一 pad 到全局最大长度再拼接
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    max_len = max(s.shape[1] for s in seqs_list)
    seqs = torch.cat(
        [torch.nn.functional.pad(s, (0, max_len - s.shape[1]), value=pad_id)
         for s in seqs_list], dim=0)

    results = []
    for g in range(group_size):
        new_ids = seqs[g, prompt_len:]
        new_text = processor.decode(new_ids, skip_special_tokens=True)
        results.append((new_text, None, seqs[g]))
    return results, prompt_len, text


def forward_logprob(model, processor, image, text, seq, prompt_len):
    """单条带梯度前向：返回 (logprob_sum, h_end)。逐条调用以控制显存。"""
    single_inputs = processor(text=[text], images=[image], return_tensors="pt").to(DEVICE)
    full_ids = seq.unsqueeze(0)
    fwd_inputs = dict(single_inputs)
    fwd_inputs["input_ids"] = full_ids
    fwd_inputs["attention_mask"] = torch.ones_like(full_ids)
    fwd = model(**fwd_inputs, output_hidden_states=True, return_dict=True)

    new_ids = seq[prompt_len:]
    logits = fwd.logits[0, prompt_len - 1:-1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    logprob_sum = logp.gather(1, new_ids.unsqueeze(1)).sum()
    h_end = fwd.hidden_states[-1][0, -1, :].detach().cpu().float()
    return logprob_sum, h_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_path", default="",
                    help="SFT LoRA 检查点；为空 = 素人冷启动（随机初始化 LoRA）")
    ap.add_argument("--rl_arrow", required=True,
                    help="RL 训练 prompt 来源（val 集的 .arrow：图+查询，与评分缓存同表）")
    ap.add_argument("--scorer_ckpt", required=True)
    ap.add_argument("--val_arrow", required=True)
    ap.add_argument("--val_cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--sample_batch", type=int, default=0,
                    help="组采样每批条数（0 = 一次生成全部；显存不足时设 8）")
    ap.add_argument("--random_tool", action="store_true",
                    help="随机工具消融（与评测 rollout 命令保持一致；"
                         "训练阶段 sample_group 不执行真实工具，随机化在评测 rollout 生效）")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--stats_log", required=True)
    ap.add_argument("--prompt_mode", default="tool", choices=["tool", "plain"],
                    help="tool = 工具版 prompt；plain = 纯推理 prompt（带奖励函数的纯推理 CUP）")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="R_retrieval 权重 λ（0 = 只剩格式奖励）")
    ap.add_argument("--reward-mode", default="soft",
                    choices=["soft", "hard", "format-only"],
                    help="检索奖励模式：soft = 1/(1+rank)；hard = 命中 1/未命中 0；"
                         "format-only = 检索奖励不参与")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    GLOBAL_PROMPT = SYSTEM_PROMPT if args.prompt_mode == "tool" else PLAIN_PROMPT

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. 模型：base + LoRA。lora_path 为空 = 素人冷启动（随机初始化 LoRA）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map=DEVICE)
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05)
        model = get_peft_model(model, lora_config)
        print("素人冷启动：随机初始化 LoRA")
    # 梯度检查点：8 条带梯度前向的激活内存过大，checkpoint 化以显存换时间
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(args.base_model)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        if p.requires_grad is False and p.dtype == torch.float32:
            pass
    trainable = [p for n, p in model.named_parameters() if "lora" in n]
    for p in trainable:
        p.requires_grad_(True)
    print(f"可训 LoRA 参数: {sum(p.numel() for p in trainable)}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # 2. 评分器（冻结翻译层 + 图像特征缓存）
    scorer = RetrievalScorer(args.scorer_ckpt, args.val_arrow, args.val_cache)
    print(f"评分器就绪：val 图像 {scorer.image_features.shape[0]} 张")

    # 3. RL prompt 数据：val 集（图+查询），每步随机抽 1 张；
    #    评分目标 = 该 val 图自己（rank 对齐），测试集留作最终评估
    table = pa.ipc.RecordBatchFileReader(
        pa.memory_map(args.rl_arrow, "r")).read_all()
    n_prompts = len(table)

    stats_f = open(args.stats_log, "a")
    rl_mean_tracker = []

    # 4. GRPO 主循环
    for step in range(args.steps):
        idx = random.randrange(n_prompts)
        image = Image.open(io.BytesIO(table["image"][idx].as_py())).convert("RGB")
        caption = table["caption"][idx].as_py()[0]

        samples, prompt_len, prompt_text = sample_group(
            model, processor, image, caption, args.group_size,
            sample_batch=args.sample_batch or None,
            few_shot=(args.prompt_mode == "tool"))

        # 第一遍：no_grad 前向拿 h_end → 评分 → 组内优势
        rewards, ranks = [], []
        with torch.no_grad():
            for text, _, seq in samples:
                _, h_end = forward_logprob(model, processor, image, prompt_text,
                                           seq, prompt_len)
                rank, r_ret = scorer.score(h_end, idx)
                r = total_reward(text, rank, lam=args.lam, mode=args.reward_mode)
                rewards.append(r)
                ranks.append(rank)
                stats_f.write(json.dumps({"step": step, "rank": rank,
                                          "r_retrieval": r_ret,
                                          "r_format": r - r_ret,
                                          "r_total": r,
                                          "has_tool": "<tool_call>" in text}) + "\n")
        rewards = torch.tensor(rewards, device=DEVICE)
        mean_r, std_r = rewards.mean(), rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # 第二遍：逐条带梯度前向 → 逐条 backward 累加（显存峰值 = 单条计算图）
        optimizer.zero_grad()
        G = args.group_size
        for g, (text, _, seq) in enumerate(samples):
            logp, _ = forward_logprob(model, processor, image, prompt_text,
                                      seq, prompt_len)
            loss_g = -(advantages[g] * logp) / G
            loss_g.backward()
            del logp, loss_g
            torch.cuda.empty_cache()
        optimizer.step()

        rl_mean_tracker.append(mean_r.item())
        if (step + 1) % 10 == 0:
            print(f"step {step + 1}/{args.steps} | "
                  f"R_mean {mean_r.item():.3f} (最近10步 {sum(rl_mean_tracker[-10:]) / 10:.3f}) | "
                  f"rank_mean {sum(ranks) / len(ranks):.1f}")
            stats_f.flush()

        if (step + 1) % 100 == 0:
            model.save_pretrained(f"{args.out_dir}_step{step + 1}")
            print(f"检查点保存 -> {args.out_dir}_step{step + 1}")

    model.save_pretrained(args.out_dir)
    stats_f.close()
    print(f"GRPO 完成 -> {args.out_dir}，统计日志 {args.stats_log}")


if __name__ == "__main__":
    main()
