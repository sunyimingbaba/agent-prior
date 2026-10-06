"""
域内场景分类器训练（工具升级：替换 CLIP 零样本场景分类）。

做法：冻结 prompt_clip ViT-L/14 图像塔，提取 train 全部图像特征，
训练线性分类头，保存 {out} 供 rollout 的工具调用。

两种标签模式（--label_mode）：
- ucm（默认）：UCM 21 类按字母序，文件名全局编号 1..2100，每类 100 张，
  label = (int(文件名) - 1) // 100。
- prefix：文件名形如 "airport_1.jpg"，类别 = 下划线前缀；无前缀的行跳过。
  类别表从训练集动态构建（排序），随 ckpt 保存。

用法（服务器）：
  python agent/train_scene_cls.py \
      --arrow /workspace/cup-debug/data/ucm/ucm_caption_karpathy_label_train.arrow \
      --val_arrow /workspace/cup-debug/data/ucm/ucm_caption_karpathy_label_val.arrow \
      --out /workspace/cup-debug/models/ucm_scene_cls.pt
  python agent/train_scene_cls.py \
      --arrow /workspace/cup-debug/data/rsitmd/rsitmd_caption_karpathy_label_train.arrow \
      --out /workspace/cup-debug/models/rsitmd_scene_cls.pt --label_mode prefix
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
import torch
from PIL import Image

UCM_CLASSES = [
    "agricultural", "airplane", "baseball diamond", "beach", "buildings",
    "chaparral", "dense residential", "forest", "freeway", "golf course",
    "harbor", "intersection", "medium residential", "mobile home park",
    "overpass", "parking lot", "river", "runway", "sparse residential",
    "storage tanks", "tennis court",
]


def get_label(path: str, label_mode: str, class_index: dict):
    """返回类别下标；prefix 模式下无类别前缀返回 None（跳过该行）。"""
    name = path.split(".")[0]
    if label_mode == "prefix":
        if "_" not in name:
            return None
        return class_index.get(name.split("_")[0])
    num = int(name)
    return (num - 1) // 100


def extract_features(model, preprocess, table, device, label_mode, class_index,
                     batch=64):
    """提取 arrow 表内图像特征 + 标签（跳过 label 为 None 的行）。"""
    feats, labels = [], []
    n = len(table)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        batch_items = []
        for i in range(start, end):
            lab = get_label(table["path"][i].as_py(), label_mode, class_index)
            if lab is None:
                continue
            batch_items.append(i)
        if not batch_items:
            continue
        imgs = [Image.open(io.BytesIO(table["image"][i].as_py())).convert("RGB")
                for i in batch_items]
        imgs = torch.stack([preprocess(im) for im in imgs]).to(device)
        width = model.visual.conv1.weight.shape[0]
        zero_prompt = torch.zeros(imgs.shape[0], 16, width,
                                  dtype=imgs.dtype, device=imgs.device)
        with torch.no_grad():
            f = model.encode_image(imgs, zero_prompt)
        f = f / f.norm(dim=1, keepdim=True)
        feats.append(f.cpu())
        labels.extend(get_label(table["path"][i].as_py(), label_mode,
                                class_index) for i in batch_items)
    return torch.cat(feats), torch.tensor(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", required=True)
    ap.add_argument("--val_arrow", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--label_mode", default="ucm", choices=["ucm", "prefix"],
                    help="ucm = 编号映射（UCM 专用）；prefix = 文件名类别前缀")
    args = ap.parse_args()

    device = "cuda"
    from prompt_clip import clip

    model, preprocess = clip.load("ViT-L/14", device=device)
    model = model.float().eval()

    train_t = pa.ipc.RecordBatchFileReader(
        pa.memory_map(args.arrow, "r")).read_all()
    if args.label_mode == "prefix":
        names = sorted({p.split(".")[0].split("_")[0]
                        for p in train_t["path"].to_pylist()
                        if "_" in p.split(".")[0]})
        classes = names
    else:
        classes = UCM_CLASSES
    class_index = {c: i for i, c in enumerate(classes)}
    print(f"类别表({args.label_mode}) {len(classes)} 类: {classes}")

    X, y = extract_features(model, preprocess, train_t, device,
                            args.label_mode, class_index, args.batch)
    print(f"训练特征 {X.shape}，标签分布 {torch.bincount(y).tolist()}")

    head = torch.nn.Linear(X.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(len(X))
        total, correct, loss_sum = 0, 0, 0.0
        for start in range(0, len(X), args.batch):
            idx = perm[start:start + args.batch]
            xb, yb = X[idx].to(device), y[idx].to(device)
            logits = head(xb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            loss_sum += loss.item() * len(yb)
        print(f"epoch {epoch + 1}/{args.epochs} "
              f"acc {correct / total:.4f} loss {loss_sum / total:.4f}")

    head.eval()
    if args.val_arrow:
        val_t = pa.ipc.RecordBatchFileReader(
            pa.memory_map(args.val_arrow, "r")).read_all()
        Xv, yv = extract_features(model, preprocess, val_t, device,
                                  args.label_mode, class_index, args.batch)
        with torch.no_grad():
            acc = (head(Xv.to(device)).argmax(1).cpu() == yv).float().mean()
        print(f"val acc {acc.item():.4f}")

    torch.save({"head": head.state_dict(),
                "classes": classes,
                "feat_dim": X.shape[1]}, args.out)
    print(f"分类头已保存 -> {args.out}")


if __name__ == "__main__":
    main()
