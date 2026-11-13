"""
阶段 C：检索评分器。

架构（方案 3.3）：
  离线预计算（一次性）：val 集全部图像 → RSP-CLIP 图像塔 → 图像特征缓存
  在线评分（每条轨迹）：h_end → 冻结翻译层 → prior 融合 → CLIP 文本塔
                        → 与 val 图像特征算相似度 → 真实图像的 rank → 1/(1+rank)

翻译层冻结是硬约束（否则奖励非平稳）。

实现要点：
  - 复用训练好的 RSP-CLIP checkpoint（纯推理版）里的组件：agent_proj、
    visual_prompt_proj_gather、CLIP 图像/文本塔、文本 tokenize
  - 图像特征缓存用同一个 checkpoint 的 img_embeds 路径编码
  - 评分时只动 CLIP 文本塔 + 冻结翻译层，整链 no_grad
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pyarrow as pa
import torch
from PIL import Image
from tqdm import tqdm


class _MyBackboneFinetuningStub:
    """checkpoint pickle 里引用了 run_frozen.MyBackboneFinetuning，加载前注册桩类。"""

    def __init__(self, *args, **kwargs):
        pass


import __main__  # noqa: E402
__main__.MyBackboneFinetuning = _MyBackboneFinetuningStub


class RetrievalScorer:
    """离线图像特征缓存 + 在线轨迹评分（全部冻结、no_grad）。"""

    def __init__(self, ckpt_path, val_arrow_path, cache_path, device="cuda"):
        from vilt.modules.vilt_module import ViLTransformerSS
        from vilt.modules import vilt_utils

        config = {
            "clip_model": "ViT-L/14", "prompt_length": 16,
            "txt_features_dim": 768, "img_features_dim": 1024,
            "image_size": [14, 14],
            "loss_scale": 1, "loss_scale_kl": 1, "loss_scale_un": 1,
            "load_path": ckpt_path, "test_only": True,
            "loss_names": {"clip": 1},
            "use_agent_prior": True, "agent_hidden_dim": 3584,
        }
        self.pl = ViLTransformerSS(config).to(device).eval()
        for p in self.pl.parameters():
            p.requires_grad_(False)          # 翻译层等全部冻结（硬约束）

        self.device = device
        self.val_arrow_path = val_arrow_path
        self.cache_path = cache_path
        self.image_features, self.image_indices = self._load_or_build_cache()

        # 评分链走视觉塔（dummy 图 + visual_prompt），positional 为训练同款
        # （文本塔的 positional 只有 77 维，插不进 16 长度的 prompt）

    def _load_or_build_cache(self):
        """离线预计算 val 图像特征（或从磁盘读缓存）。"""
        if os.path.exists(self.cache_path):
            data = torch.load(self.cache_path, map_location="cpu")
            return data["image_features"], data["image_indices"]

        from model import clip
        table = pa.ipc.RecordBatchFileReader(
            pa.memory_map(self.val_arrow_path, "r")
        ).read_all()
        feats, indices = [], []
        for idx in tqdm(range(len(table)), desc="val 图像特征预计算"):
            image = Image.open(io.BytesIO(table["image"][idx].as_py())).convert("RGB")
            clip_img = self.pl.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self._encode_image(clip_img)
            feats.append(feat.cpu())
            indices.append(idx)
        image_features = torch.cat(feats, dim=0)   # [N, D]
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        torch.save({"image_features": image_features, "image_indices": indices},
                   self.cache_path)
        return image_features, indices

    def _encode_image(self, img):
        """与训练同源的图像分支：agent_proj 用零 h_end 占位（图像编码不走先验路径）。"""
        from vilt.modules.vilt_module import ViLTransformerSS  # noqa
        with torch.no_grad():
            feat = self.pl.model.encode_image(img, torch.zeros(
                1, self.pl.prompt_length, 1024, dtype=img.dtype, device=img.device))
            feat = feat / feat.norm(dim=1, keepdim=True)
        return feat

    @torch.no_grad()
    def score(self, h_end, image_index):
        """
        在线评分：h_end → agent_proj → prior → CLIP 文本塔 → 与 val 图像特征相似度
        → image_index 对应图像的 rank → 返回 (rank, r_retrieval)。
        """
        h = torch.tensor(h_end, dtype=torch.float32).unsqueeze(0).to(self.device)
        prior = self.pl.agent_proj(h)                      # [1, 1024] 冻结翻译层
        prior = prior.unsqueeze(1).repeat(1, self.pl.prompt_length, 1)
        ratio = torch.clip(self.pl.cluster_mapping(prior), 0, 1)
        random_prompt = self.pl.visual_prompt_proj(
            self.pl.visual_prompt_embeddings).expand(1, -1, -1)
        visual_prompt = (1 - ratio) * prior + ratio * random_prompt
        visual_prompt = self.pl.visual_prompt_proj_gather(visual_prompt)

        # 评分走视觉塔：dummy 图 + visual_prompt（1024 维，训练同款路径）
        dummy_img = torch.zeros(1, 3, 224, 224, device=self.device)
        feat = self.pl.model.encode_image(dummy_img, visual_prompt)
        feat = feat / feat.norm(dim=1, keepdim=True)

        sims = feat @ self.image_features.to(self.device).t()   # [1, N]
        rank = int((sims > sims[0, image_index]).sum().item())  # 真实图像排第几（0-based）
        return rank, 1.0 / (1.0 + rank)


def main():
    ap = argparse.ArgumentParser(description="预计算 val 图像特征缓存")
    ap.add_argument("--ckpt", required=True, help="RSP-CLIP checkpoint（纯推理版）")
    ap.add_argument("--val_arrow", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    scorer = RetrievalScorer(args.ckpt, args.val_arrow, args.cache)
    print(f"缓存完成：{scorer.image_features.shape[0]} 张 val 图像 -> {args.cache}")


if __name__ == "__main__":
    main()
