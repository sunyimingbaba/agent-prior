"""
CUP agent 轨迹生成主脚本（阶段 1.2）
====================================
离线阶段：对每个 (遥感图像, 查询 caption)，用 MLLM（Qwen2.5-VL-7B）推理，
保存模型"看过图+查询"后的最后一个 token 隐藏态 h_end 到磁盘缓存。

缓存协议（与训练侧数据集加载器约定）：
  路径：{cache_dir}/{table_name}/{img_index}.pt
        table_name 如 rsicd_caption_karpathy_label_train（天然含 split），
        img_index 是 .arrow 表中该图像的整数行号 —— 与 base_dataset 的
        index_mapper 图像索引严格对齐。
  内容：torch tensor，shape [3584]（Qwen2.5-VL-7B 隐藏维，已 squeeze 掉 batch 维）。
  训练侧读取后做 .float() 并送入 agent_proj。

第一版：纯推理 —— 模型直接生成一段简短分析，取生成序列最后 token 的
h_end，不涉及工具调用。第二阶段再升级为 <think>/<tool_call> 工具循环
（tools.py 已备好两个工具）。

用法（服务器）：
  python CUP/agent/rollout.py \
      --data_root /data2/dsets/dataset \
      --dataset rsicd \
      --model Qwen/Qwen2.5-VL-7B-Instruct \
      --cache_dir ./agent_cache
断点续跑：已存在的缓存文件自动跳过，可反复执行。
"""

import argparse
import io
import os
import random
import re
import sys

# 把 CUP/ 加入 sys.path，使工具版能 import prompt_clip 与同目录的 tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 每个遥感数据集对应的三个 .arrow 表（train/val/test karpathy 划分）
DATASET_TABLES = {
    "rsicd": [
        "rsicd_caption_karpathy_label_train",
        "rsicd_caption_karpathy_label_val",
        "rsicd_caption_karpathy_label_test",
    ],
    "ucm": [
        "ucm_caption_karpathy_label_train",
        "ucm_caption_karpathy_label_val",
        "ucm_caption_karpathy_label_test",
    ],
    "sydney": [
        "Sydney_caption_karpathy_label_train",
        "Sydney_caption_karpathy_label_val",
        "Sydney_caption_karpathy_label_test",
    ],
    # RSITMD 只有 train/test 划分（val 与 test 同表，见 rsitmd_caption_karpathy_dataset.py）
    "rsitmd": [
        "rsitmd_caption_karpathy_label_train",
        "rsitmd_caption_karpathy_label_test",
    ],
}

SYSTEM_PROMPT = (
    "你是遥感图像分析助手。给定一张遥感图像和一条查询文本，分析图像中与查询相关的内容。"
    "你可以使用工具 scene_classify 获取图像的场景分类信息（top-3 场景类别及概率）。"
    "使用工具时：只输出 <tool_call>scene_classify</tool_call>，然后立即停止生成，"
    "等待系统把工具结果以 <result>...</result> 的形式返回给你，再继续分析。"
    "千万不要自己编造或猜测工具结果。"
    "分析完成后以 <answer>...</answer> 输出最终答案。"
)

# 纯推理模式 prompt（--no_tools 时使用：阶段 A 原版，无工具提示）
PLAIN_PROMPT = (
    "你是遥感图像分析助手。给定一张遥感图像和一条查询文本，"
    "请简要描述图像中与查询相关的内容。直接回答，不要输出思考过程。"
)


def load_mllm(model_path, device, lora_path=""):
    """加载 Qwen2.5-VL-7B 和对应 processor（bf16，约需 16GB 显存）。
    lora_path 非空时用 peft 加载 LoRA 适配器（阶段 B 的 SFT 模型）。"""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"已加载 LoRA 适配器: {lora_path}")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# 闭合标签可选：素人 Qwen 学 few-shot 后常输出 "<tool_call>scene_classify" 后直接 EOS
TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?")


def parse_tool_call(text):
    """从生成文本中提取第一个工具调用。返回工具名或 None。"""
    m = TOOL_CALL_RE.search(text)
    if m:
        name = m.group(1).strip()
        return name if name == "scene_classify" else None  # 只认已注册的工具
    return None


@torch.no_grad()
def rollout_one(model, processor, image, caption, tools, max_rounds,
                max_new_tokens, device, verbose=False, injected_h_end=False,
                system_prompt=None, temperature=0.0):
    """
    工具循环版 rollout：生成 → 解析 tool_call → 执行工具 → <result> 回填 → 再生成。
    解析不到 tool_call 时正常结束循环（该图退化为纯推理，照常存 h_end）。

    返回 (h_end, tool_used, tool_filtered)：
      默认 h_end 口径与纯推理版一致——取最后一次 generate 的最后一步、
      最后一层、最后一个 token（shape [3584]）。
      injected_h_end=True 时（实验2：口径修正），若发生过工具注入，
      h_end 改为取「注入后状态」——回填后直接 forward 的最后 token 隐藏态，
      避免模型第二轮乱文生成污染状态。
      tool_used 为本张图实际成功调用工具的次数（统计用）。
    """
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
    ]
    if (system_prompt or SYSTEM_PROMPT) == SYSTEM_PROMPT:
        # few-shot 示例：教模型"发 tool_call 等结果"的正确协议（仅工具版 prompt 使用）
        messages += [
            {"role": "user", "content": [{"type": "text", "text": "查询文本: There is a harbor with several ships."}]},
            {"role": "assistant", "content": "<tool_call>scene_classify</tool_call>"},
            {"role": "user", "content": "<result>harbor (0.61), port (0.24), ship (0.08)</result>"},
            {"role": "assistant", "content": "<answer>图中是一座港口，与查询文本相符。</answer>"},
        ]
    # 真实查询
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"查询文本: {caption}"},
        ],
    })
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    tool_used = 0
    tool_filtered = 0
    outputs = None
    last_injected_hidden = None
    for _ in range(max_rounds):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),            # 0 = 贪心（确定）；>0 = 采样
            temperature=(temperature if temperature > 0 else 1.0),
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        new_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
        new_text = processor.decode(new_ids, skip_special_tokens=True)
        if verbose:
            print(f"  [round {_ + 1}] 模型输出: {new_text!r}")

        tool_name = parse_tool_call(new_text)
        if tool_name is None:
            break                      # 没调工具（<answer> 或格式失败）→ 循环结束

        result_text = tools[tool_name](image)     # 真实执行工具；低置信度时返回 None
        if result_text is None:                   # 置信度过滤：不注入，退化为纯推理
            tool_filtered += 1
            if verbose:
                print(f"  [round {_ + 1}] 工具调用: {tool_name} -> 置信度过低，跳过注入")
            break
        tool_used += 1
        if verbose:
            print(f"  [round {_ + 1}] 工具调用: {tool_name} -> 结果: {result_text!r}")

        # 回填：本轮输出 + <result> 拼进 input_ids（pixel_values 不动），继续下一轮
        feedback = new_text + f"\n<result>{result_text}</result>\n"
        feedback_ids = processor(text=[feedback], return_tensors="pt").input_ids.to(device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], feedback_ids], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        if injected_h_end:
            # 实验2口径：注入后状态 = 回填后单次 forward 的最后 token 隐藏态
            fwd = model(**inputs, output_hidden_states=True, return_dict=True)
            last_injected_hidden = fwd.hidden_states[-1][:, -1, :]

    if outputs is None:                # max_rounds<=0 的边界：不静默，直接报错
        raise RuntimeError("rollout_one 未执行任何 generate（max_rounds 必须 >= 1）")

    if injected_h_end and last_injected_hidden is not None:
        h_end = last_injected_hidden.squeeze(0).cpu()
    else:
        # hidden_states: (生成步数) 步 × (层数) 层，每层 [1, seq_len, 3584]
        # 取最后一步、最后一层、最后一个 token → [1, 3584] → squeeze → [3584]
        last_step_hidden = outputs.hidden_states[-1][-1]
        h_end = last_step_hidden[:, -1, :].squeeze(0).cpu()
    return h_end, tool_used, tool_filtered


def main():
    parser = argparse.ArgumentParser(description="CUP agent 先验 h_end 离线生成")
    parser.add_argument("--data_root", type=str, required=True,
                        help=".arrow 数据集根目录（与训练配置 data_root 一致）")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_TABLES.keys()))
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cache_dir", type=str, default="./agent_cache_tool")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度（0 = 贪心解码，确定性；>0 启用采样）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0,
                        help="每个 split 只处理前 N 张图（0 = 全部），调试用")
    parser.add_argument("--max_rounds", type=int, default=3,
                        help="工具循环最大轮数")
    parser.add_argument("--no_tools", action="store_true",
                        help="回退纯推理模式（供对照调试）")
    parser.add_argument("--verbose", action="store_true",
                        help="逐张打印轨迹（模型输出 + tool_call + result），调试用")
    parser.add_argument("--tool_conf_threshold", type=float, default=0.29,
                        help="工具置信度阈值（cosine）：低于阈值不注入结果，退化为纯推理")
    parser.add_argument("--domain_cls", type=str, default="",
                        help="域内场景分类头路径（train_scene_cls.py 产出）；"
                             "非空时工具从 CLIP 零样本切换为域内微调分类器")
    parser.add_argument("--random_tool", action="store_true",
                        help="随机工具消融：分类头输出类别名按固定 seed(0) 打乱，"
                             "置信度保持原值——工具照常调用回填，但信息无意义")
    parser.add_argument("--injected_h_end", action="store_true",
                        help="实验2：h_end 取注入后状态（回填后 forward），而非生成后状态")
    parser.add_argument("--lora_path", type=str, default="",
                        help="阶段 B：LoRA 适配器路径，非空时加载 SFT 版 MLLM")
    args = parser.parse_args()

    model, processor = load_mllm(args.model, args.device, lora_path=args.lora_path)

    if args.no_tools:
        tools = {}
    else:
        # 工具版：预加载一次 CLIP，避免每张图重复 load（约 2GB 模型）
        from prompt_clip import clip
        from tools import scene_classify_with_conf
        clip_model, preprocess = clip.load("ViT-L/14")
        clip_model = clip_model.float().eval().cuda()
        threshold = args.tool_conf_threshold

        cls_head = None
        cls_names = None
        if args.domain_cls:
            ckpt = torch.load(args.domain_cls, map_location="cpu")
            cls_head = torch.nn.Linear(ckpt["feat_dim"], len(ckpt["classes"]))
            cls_head.load_state_dict(ckpt["head"])
            cls_head = cls_head.float().cuda().eval()
            cls_names = list(ckpt["classes"])
            if args.random_tool:
                rng = random.Random(0)
                perm = list(range(len(cls_names)))
                rng.shuffle(perm)
                cls_names = [cls_names[perm[i]] for i in range(len(cls_names))]
                print("随机工具模式：类别名已按固定 seed(0) 打乱")
            print(f"域内分类头已加载：{len(cls_names)} 类 <- {args.domain_cls}")

        def scene_tool(image):
            text, conf = scene_classify_with_conf(
                image, model=clip_model, preprocess=preprocess,
                cls_head=cls_head, cls_names=cls_names)
            return text if conf >= threshold else None

        tools = {"scene_classify": scene_tool}

    tables = DATASET_TABLES[args.dataset]
    if args.split != "all":
        tables = [t for t in tables if t.endswith(args.split)]

    total_generated, total_skipped, total_tool_calls, total_filtered = 0, 0, 0, 0
    for table_name in tables:
        arrow_path = os.path.join(args.data_root, f"{table_name}.arrow")
        if not os.path.exists(arrow_path):      # 某些数据集无 val 表（如 RSITMD），跳过
            print(f"跳过：表文件不存在 {arrow_path}")
            continue
        table = pa.ipc.RecordBatchFileReader(
            pa.memory_map(arrow_path, "r")
        ).read_all()

        cache_dir = os.path.join(args.cache_dir, table_name)
        os.makedirs(cache_dir, exist_ok=True)

        n_rows = len(table)
        if args.limit > 0:
            n_rows = min(n_rows, args.limit)

        for img_index in tqdm(range(n_rows), desc=table_name):
            cache_path = os.path.join(cache_dir, f"{img_index}.pt")
            if os.path.exists(cache_path):      # 断点续跑：已有缓存直接跳过
                total_skipped += 1
                continue

            # 查询文本：每张图 5 个 caption，取第一个（与方案 1.3 约定一致）
            caption = table["caption"][img_index].as_py()[0]
            image = Image.open(io.BytesIO(table["image"][img_index].as_py())).convert("RGB")

            if args.verbose:
                print(f"\n=== 图 {img_index} | 查询: {caption} ===")
            h_end, tool_used, tool_filtered = rollout_one(
                model, processor, image, caption, tools, args.max_rounds,
                args.max_new_tokens, args.device, verbose=args.verbose,
                injected_h_end=args.injected_h_end,
                system_prompt=PLAIN_PROMPT if args.no_tools else None,
                temperature=args.temperature)
            torch.save(h_end, cache_path)
            total_generated += 1
            total_tool_calls += tool_used
            total_filtered += tool_filtered

    rate = total_tool_calls / total_generated if total_generated > 0 else 0
    print(f"完成：新生成 {total_generated} 张，跳过（已缓存）{total_skipped} 张，"
          f"缓存目录 {args.cache_dir}")
    print(f"工具调用统计：{total_tool_calls} 次注入 / {total_generated} 张图"
          f"（注入率 {rate:.1%}），低置信度过滤 {total_filtered} 次")


if __name__ == "__main__":
    main()
