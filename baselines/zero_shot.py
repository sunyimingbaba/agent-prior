"""
零样本基线：官方 CLIP ViT-L/14，不做任何训练，直接测试集评测。

口径（与 CUP 复现实验一致）：
- 数据划分：CUP 的 karpathy split json（ucm/rsicd/rsitmd.json），不用随机切分
- 图像：官方 ViT-L/14 预处理；每张图编码一个图像特征
- 文本：每张图的所有去重 caption 各编码一个文本特征（CUP 的 all_texts 口径）
- 指标：图→文 R@1/5/10、文→图 R@1/5/10、mR（6 指标平均）
- 图→文命中判定：top-k 文本中任一属于查询图即算对；
  文→图命中判定：top-k 图像中任一为该文本所属图即算对

用法：
  python zero_shot.py --json ucm.json --arrow ucm/ucm_caption_karpathy_label_test.arrow \
      --ckpt ViT-L-14.pt --bpe bpe_simple_vocab_16e6.txt.gz --dataset ucm
"""
import argparse
import io
import json
import os

import pyarrow as pa
import torch
from PIL import Image
from tqdm import tqdm

from clip_backbone import build_transform, get_tokenizer, load_clip


def load_data(json_path, arrow_path, split):
    """读 karpathy json（划分 + captions + filename）+ arrow（图像 bytes）。

    返回：images[(filename, image_bytes, [captions])]
    """
    with open(json_path) as f:
        data = json.load(f)
    table = pa.ipc.RecordBatchFileReader(
        pa.memory_map(arrow_path, "r")).read_all()

    # arrow 行 → (path, image bytes) 索引
    bytes_by_path = {}
    for i in range(len(table)):
        bytes_by_path[str(table["path"][i].as_py())] = table["image"][i].as_py()

    images = []
    for im in data["images"]:
        if im["split"] != split:
            continue
        filename = im["filename"]
        if filename not in bytes_by_path:
            raise FileNotFoundError(f"图像缺失于 arrow: {filename}")
        captions = [s["raw"] for s in sorted(im["sentences"], key=lambda s: s["sentid"])]
        images.append((filename, bytes_by_path[filename], captions))
    return images


@torch.no_grad()
def encode_all(model, preprocess, tokenizer, images, device, bs=64):
    """编码全部图像（每图 1 个特征）与全部文本（每图去重 captions）。"""
    img_feats = []
    for i in tqdm(range(0, len(images), bs), desc="图像编码"):
        batch = []
        for _, raw, _ in images[i:i + bs]:
            pil = Image.open(io.BytesIO(raw))
            batch.append(preprocess(pil))
        x = torch.stack(batch).to(device)
        f = model.encode_image(x)
        img_feats.append(f.cpu())
    img_feats = torch.cat(img_feats, dim=0)                      # [N_img, 768]

    # 文本：每图去重 captions（与 CUP all_texts 口径一致）
    all_txt, txt_owner = [], []
    for idx, (_, _, captions) in enumerate(images):
        for cap in set(captions):
            all_txt.append(cap)
            txt_owner.append(idx)
    txt_feats = []
    for i in tqdm(range(0, len(all_txt), bs), desc="文本编码"):
        tokens = tokenizer.tokenize(all_txt[i:i + bs]).to(device)
        f = model.encode_text(tokens)
        txt_feats.append(f.cpu())
    txt_feats = torch.cat(txt_feats, dim=0)                      # [M_txt, 768]
    txt_owner = torch.tensor(txt_owner)
    return img_feats, txt_feats, txt_owner


def evaluate(img_feats, txt_feats, txt_owner):
    """双向 R@1/5/10 + mR（与 CUP 论文口径一致）。"""
    img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
    txt_feats = txt_feats / txt_feats.norm(dim=1, keepdim=True)
    sim = img_feats @ txt_feats.t()                              # [N_img, M_txt]

    n_img = img_feats.shape[0]
    # 图→文：top-k 文本中任一属于查询图
    topk = sim.topk(max(10, min(10, sim.shape[1])), dim=1).indices
    owner_mat = txt_owner.unsqueeze(0).expand(n_img, -1)          # [N_img, M_txt]
    query_ids = torch.arange(n_img).unsqueeze(1)                  # [N_img, 1]
    hits = (owner_mat.gather(1, topk) == query_ids).float()       # [N_img, K]
    ir = {k: hits[:, :k].max(dim=1)[0].mean().item() for k in [1, 5, 10]}

    # 文→图：top-k 图像中任一为该文本所属图
    sim_t = sim.t()                                               # [M_txt, N_img]
    topk_t = sim_t.topk(max(10, min(10, sim_t.shape[1])), dim=1).indices
    hits_t = (topk_t == txt_owner.unsqueeze(1)).float()
    tr = {k: hits_t[:, :k].max(dim=1)[0].mean().item() for k in [1, 5, 10]}

    mR = (ir[1] + ir[5] + ir[10] + tr[1] + tr[5] + tr[10]) / 6
    return ir, tr, mR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--arrow", required=True)
    ap.add_argument("--ckpt", required=True, help="官方 ViT-L-14.pt 路径")
    ap.add_argument("--bpe", required=True, help="bpe_simple_vocab_16e6.txt.gz 路径")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_clip(args.ckpt, device)
    preprocess = build_transform()
    tokenizer = get_tokenizer(args.bpe)

    images = load_data(args.json, args.arrow, args.split)
    print(f"{args.split} 集：{len(images)} 张图")

    img_feats, txt_feats, txt_owner = encode_all(
        model, preprocess, tokenizer, images, device, args.bs)
    ir, tr, mR = evaluate(img_feats, txt_feats, txt_owner)

    print(f"\n=== 零样本结果 ===")
    print(f"图→文  R@1={ir[1]*100:.2f}  R@5={ir[5]*100:.2f}  R@10={ir[10]*100:.2f}")
    print(f"文→图  R@1={tr[1]*100:.2f}  R@5={tr[5]*100:.2f}  R@10={tr[10]*100:.2f}")
    print(f"mR = {mR*100:.2f}")


if __name__ == "__main__":
    main()
