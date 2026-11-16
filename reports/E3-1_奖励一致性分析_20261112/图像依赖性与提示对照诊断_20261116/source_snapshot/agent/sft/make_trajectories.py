"""
阶段 B：教师轨迹生成（本地 Mac 跑，调用 DeepSeek Vision API）。

数据流：
  1. 读本地 .arrow 的抽样图 + 服务器导出的真实工具结果 JSON（export_tool_results.py 产出）
  2. 教师（deepseek-v4-flash-vision-exp）看图 + 查询（+ 工具结果）→ 输出 <answer> 整合分析
  3. 组装轨迹（与 rollout.py 推理格式逐字符一致）：
     assistant = "<tool_call>scene_classify</tool_call>\\n<result>{真实结果}</result>\\n<answer>{教师分析}</answer>"
  4. 80% 调工具轨迹 + 20% 纯回答轨迹（教师直接 <answer>，不调工具）
  5. 输出 JSONL：{"messages": [system, user(图+查询), assistant]}

用法（本地 Mac）：
  DEEPSEEK_API_KEY=... python3 make_trajectories.py \
      --arrow ~/cup-deploy-assets/arrow/ucm/ucm_caption_karpathy_label_train.arrow \
      --tool_results tool_results.json \
      --out trajectories.jsonl --limit 50
"""
import argparse
import base64
import io
import json
import os
import random
import ssl
import sys
import time
import urllib.request

import pyarrow as pa
from tqdm import tqdm

# 与 rollout.py 的工具版 SYSTEM_PROMPT 保持一致（训练格式 = 推理格式）
SYSTEM_PROMPT = (
    "你是遥感图像分析助手。给定一张遥感图像和一条查询文本，分析图像中与查询相关的内容。"
    "你可以使用工具 scene_classify 获取图像的场景分类信息（top-3 场景类别及概率）。"
    "使用工具时：只输出 <tool_call>scene_classify</tool_call>，然后立即停止生成，"
    "等待系统把工具结果以 <result>...</result> 的形式返回给你，再继续分析。"
    "千万不要自己编造或猜测工具结果。"
    "分析完成后以 <answer>...</answer> 输出最终答案。"
)

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash-vision-exp"
SSL_CTX = ssl._create_unverified_context()   # 本机 Python 证书链缺失，绕开验证


def _ensure_answer_tag(answer):
    """教师偶尔漏 <answer> 标签，后处理强制包裹（保持 SFT 数据格式 100% 统一）。"""
    answer = answer.strip()
    if answer.startswith("<answer>") and answer.endswith("</answer>"):
        return answer
    return f"<answer>{answer}</answer>"


def call_api(messages, max_tokens=300, retries=3):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"],
        },
    )
    for attempt in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180, context=SSL_CTX))
            return r["choices"][0]["message"]["content"] or ""
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    [API 重试 {attempt + 1}/{retries}] {type(e).__name__}", flush=True)
            time.sleep(2 * (attempt + 1))


def build_trajectory(image_bytes, caption, tool_result, use_tool, rng):
    """组装一条 SFT 轨迹。use_tool=True 为调工具轨迹，False 为纯回答轨迹。

    输出格式：image_base64 单独存放；messages 的 user content 用
    {"type": "image"} 占位（与 rollout.py 推理时的 messages 格式一致）。

    注：图像统一转 JPEG 再传 API（RS 数据含 TIFF，API 只收 jpeg/png/webp/gif）；
    SFT 训练侧用原始 bytes（与 rollout 推理看到的图完全一致）。
    """
    from PIL import Image as _PILImage

    jpeg_bytes = io.BytesIO()
    _PILImage.open(io.BytesIO(image_bytes)).convert("RGB").save(jpeg_bytes, format="JPEG")
    api_user_content = [
        {"type": "image_url",
         "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes.getvalue()).decode()}},
        {"type": "text", "text": f"查询文本: {caption}"},
    ]
    sft_user_content = [
        {"type": "image"},
        {"type": "text", "text": f"查询文本: {caption}"},
    ]

    if use_tool:
        # 教师只负责"基于真实结果写分析"；tool_call 与 result 注入按
        # rollout 推理的真实交互组装成【多轮 messages】：
        #   assistant 发 tool_call → user 注入 result → assistant 作答
        # 训练格式与推理逐轮一致（修复单轮整段格式导致的"自编 result"错配）
        teacher_in = [
            {"role": "user", "content": api_user_content + [
                {"type": "text", "text":
                    f"系统已调用场景分类工具，返回结果：{tool_result}。"
                    "请基于该工具结果和图像，用 <answer>...</answer> 格式输出对查询的整合分析。"}]},
        ]
        answer = call_api(teacher_in)
        answer = _ensure_answer_tag(answer)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sft_user_content},
            {"role": "assistant", "content": "<tool_call>scene_classify</tool_call>"},
            {"role": "user", "content": f"<result>{tool_result}</result>"},
            {"role": "assistant", "content": answer},
        ]
    else:
        teacher_in = [
            {"role": "user", "content": api_user_content + [
                {"type": "text", "text":
                    "请直接分析这张图，用 <answer>...</answer> 格式输出对查询的回答，不要调用任何工具。"}]},
        ]
        answer = call_api(teacher_in)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sft_user_content},
            {"role": "assistant", "content": _ensure_answer_tag(answer)},
        ]

    return {
        "image_base64": base64.b64encode(image_bytes).decode(),
        "messages": messages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", required=True, help="本地 .arrow 文件")
    ap.add_argument("--tool_results", required=True, help="服务器导出的工具结果 JSON")
    ap.add_argument("--out", required=True, help="输出 JSONL")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tool_ratio", type=float, default=0.8,
                    help="调工具轨迹占比（其余为纯回答轨迹）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    table = pa.ipc.RecordBatchFileReader(
        pa.memory_map(args.arrow, "r")
    ).read_all()
    n_rows = len(table) if args.limit <= 0 else min(len(table), args.limit)

    with open(args.tool_results) as f:
        tool_results = json.load(f)
    # 只取与当前 arrow 对应表名的工具结果（表名 = arrow 文件名去掉 .arrow）
    table_name = os.path.basename(args.arrow).replace(".arrow", "")
    flat = {int(k): v for k, v in tool_results.get(table_name, {}).items()}

    # 断点续跑：已有输出文件则从已有行数继续（append 模式）
    start_idx = 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            start_idx = sum(1 for _ in f)
        print(f"断点续跑：已存在 {start_idx} 条，从第 {start_idx} 张继续")
    if start_idx >= n_rows:
        print("轨迹已完整，无需续跑")
        return

    with open(args.out, "a") as f_out:
        for img_index in tqdm(range(start_idx, n_rows), desc="轨迹生成"):
            image_bytes = table["image"][img_index].as_py()
            caption = table["caption"][img_index].as_py()[0]
            use_tool = rng.random() < args.tool_ratio
            if use_tool and img_index not in flat:
                use_tool = False       # 工具结果缺失则退化为纯回答轨迹
            tool_result = flat.get(img_index, "")
            traj = build_trajectory(image_bytes, caption, tool_result, use_tool, rng)
            f_out.write(json.dumps(traj, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"完成：共 {n_rows} 条轨迹 -> {args.out}")


if __name__ == "__main__":
    main()
