"""
阶段 B：批量导出 scene_classify 真实工具结果（供本地轨迹组装用）。

教师轨迹的 <result> 必须由真实工具执行生成（不能用教师模型编造），
本脚本在 GPU 服务器上对指定抽样图跑 scene_classify，输出 JSON：
  {"<table_name>": {"<img_index>": "<result 文本>", ...}, ...}

用法（服务器）：
  python agent/sft/export_tool_results.py \
      --data_root /workspace/cup-debug/data/ucm --dataset ucm \
      --limit 50 --out /workspace/cup-debug/agent_cache_tool/tool_results_ucm50.json
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pyarrow as pa
import torch
from PIL import Image
from tqdm import tqdm

from agent.tools import scene_classify
from agent.rollout import DATASET_TABLES
from prompt_clip import clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASET_TABLES.keys()))
    ap.add_argument("--limit", type=int, default=0, help="每个表前 N 张（0=全部）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, preprocess = clip.load("ViT-L/14", device="cuda" if torch.cuda.is_available() else "cpu")
    model = model.float().eval()

    results = {}
    for table_name in DATASET_TABLES[args.dataset]:
        arrow_path = os.path.join(args.data_root, f"{table_name}.arrow")
        if not os.path.exists(arrow_path):
            continue
        table = pa.ipc.RecordBatchFileReader(
            pa.memory_map(arrow_path, "r")
        ).read_all()
        n_rows = len(table) if args.limit <= 0 else min(len(table), args.limit)
        table_results = {}
        for img_index in tqdm(range(n_rows), desc=table_name):
            image = Image.open(io.BytesIO(table["image"][img_index].as_py())).convert("RGB")
            table_results[str(img_index)] = scene_classify(
                image, model=model, preprocess=preprocess)
        results[table_name] = table_results

    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"已导出 {sum(len(v) for v in results.values())} 条工具结果 -> {args.out}")


if __name__ == "__main__":
    main()
