"""
线性探测基线：冻结 CLIP 双塔，图像侧加可训练线性投影层（1024→768，
对齐到文本特征空间），用训练集图像-caption 对 + InfoNCE 对比损失训练，
测试集评测（与零样本同一套口径）。

实现说明：
- 冻结骨干下，训练前先【预提取】训练集图像特征与文本特征（每图取第一个
  caption，即 CUP 查询文本口径），随后只训投影层——与在线冻结骨干逐 batch
  前向在数学上完全等价，且显存占用极小（适配 10G 配额）。
- 训练：InfoNCE 双向（图→文 + 文→图），30 epochs，lr 1e-3，batch 128，
  线性 warmup（前 2 epoch）+ 余弦退火，AdamW。
- 评测：test 图像特征过投影层 vs 全部 test 文本特征（每图去重 captions），
  与 zero_shot.py 的 evaluate 口径一致。

用法：
  python linear_probe.py --json ucm.json \
      --arrow_train ucm/ucm_caption_karpathy_label_train.arrow \
      --arrow_test  ucm/ucm_caption_karpathy_label_test.arrow \
      --ckpt ViT-L-14.pt --bpe bpe_simple_vocab_16e6.txt.gz --dataset ucm
"""
import argparse
import io
import json
import math

import pyarrow as pa
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from clip_backbone import build_transform, get_tokenizer, load_clip
from zero_shot import evaluate, load_data


@torch.no_grad()
def extract_features(model, preprocess, tokenizer, images, device, bs=64):
    """预提取：每张图 1 个图像特征 + 每图第一个 caption 的文本特征。"""
    img_feats, txt_feats = [], []
    for i in tqdm(range(0, len(images), bs), desc="特征预提取"):
        img_batch, cap_batch = [], []
        for _, raw, captions in images[i:i + bs]:
            img_batch.append(preprocess(Image.open(io.BytesIO(raw))))
            cap_batch.append(captions[0])          # 每图取第一个 caption（查询文本口径）
        x = torch.stack(img_batch).to(device)
        tokens = tokenizer.tokenize(cap_batch).to(device)
        img_feats.append(model.encode_image(x).cpu())
        txt_feats.append(model.encode_text(tokens).cpu())
    return torch.cat(img_feats, dim=0), torch.cat(txt_feats, dim=0)


def train_probe(img_feats, txt_feats, epochs=30, lr=1e-3, batch_size=128,
                warmup_epochs=2, device="cuda", seed=0):
    """InfoNCE 双向训练线性投影层（1024→768）。"""
    torch.manual_seed(seed)
    n = img_feats.shape[0]
    img_feats = img_feats.to(device)
    txt_feats = txt_feats.to(device)
    txt_feats_n = txt_feats / txt_feats.norm(dim=1, keepdim=True)

    probe = nn.Linear(768, 768, bias=False).to(device)
    nn.init.eye_(probe.weight)                     # 初始化为恒等（稳定起步）
    # 直接在目标设备上创建 Parameter（先建再 .to() 会产生非叶张量）
    logit_scale = nn.Parameter(
        torch.ones([], device=device) * math.log(1 / 0.07))

    params = list(probe.parameters()) + [logit_scale]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            labels = idx.to(device)
            # 图→文：batch 图像（投影后）vs 全库文本 → [B, N]
            p_img = probe(img_feats[idx])
            p_img = p_img / p_img.norm(dim=1, keepdim=True)
            logits = logit_scale.exp() * p_img @ txt_feats_n.t()   # [B, N]
            loss_i2t = nn.functional.cross_entropy(logits, labels)
            # 文→图：batch 文本 vs 全库图像（投影后）→ [B, N]，全局 InfoNCE 对称
            all_img_proj = probe(img_feats)
            all_img_proj = all_img_proj / all_img_proj.norm(dim=1, keepdim=True)
            logits_t2i = logit_scale.exp() * txt_feats_n[idx] @ all_img_proj.t()
            loss_t2i = nn.functional.cross_entropy(logits_t2i, labels)
            loss = (loss_i2t + loss_t2i) / 2

            opt.zero_grad()
            loss.backward()
            opt.step()
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch * steps_per_epoch + nb)
            total_loss += loss.item()
            nb += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{epochs} | loss {total_loss / max(nb, 1):.4f} "
                  f"| lr {lr_at((epoch + 1) * steps_per_epoch - 1):.2e}")
    return probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--arrow_train", required=True)
    ap.add_argument("--arrow_test", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_clip(args.ckpt, device)
    preprocess = build_transform()
    tokenizer = get_tokenizer(args.bpe)

    # 1. 预提取训练特征（图 + 每图第一个 caption）
    train_images = load_data(args.json, args.arrow_train, "train")
    print(f"train 集：{len(train_images)} 张图")
    tr_img, tr_txt = extract_features(model, preprocess, tokenizer, train_images, device, args.bs)

    # 2. 训练线性投影层
    probe = train_probe(tr_img, tr_txt, epochs=args.epochs, lr=args.lr, device=device)

    # 3. 测试集评测（图像特征过投影层）
    test_images = load_data(args.json, args.arrow_test, "test")
    print(f"test 集：{len(test_images)} 张图")
    te_img, te_txt, te_owner = [], [], []
    txt_list, owner_list = [], []
    for idx, (_, raw, captions) in enumerate(test_images):
        te_img.append(preprocess(Image.open(io.BytesIO(raw))))
        for cap in set(captions):
            txt_list.append(cap)
            owner_list.append(idx)
    img_feats = []
    with torch.no_grad():
        for i in range(0, len(te_img), args.bs):
            x = torch.stack(te_img[i:i + args.bs]).to(device)
            f = model.encode_image(x)
            img_feats.append(probe(f).cpu())
        img_feats = torch.cat(img_feats, dim=0)
        txt_feats = []
        for i in range(0, len(txt_list), args.bs):
            tokens = tokenizer.tokenize(txt_list[i:i + args.bs]).to(device)
            txt_feats.append(model.encode_text(tokens).cpu())
        txt_feats = torch.cat(txt_feats, dim=0)
    txt_owner = torch.tensor(owner_list)

    ir, tr, mR = evaluate(img_feats, txt_feats, txt_owner)
    print(f"\n=== 线性探测结果 ===")
    print(f"图→文  R@1={ir[1]*100:.2f}  R@5={ir[5]*100:.2f}  R@10={ir[10]*100:.2f}")
    print(f"文→图  R@1={tr[1]*100:.2f}  R@5={tr[5]*100:.2f}  R@10={tr[10]*100:.2f}")
    print(f"mR = {mR*100:.2f}")


if __name__ == "__main__":
    main()
