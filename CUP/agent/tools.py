"""
CUP agent 工具集
================
供 MLLM agent 调用的工具函数（阶段 1.1）。

约定：
- 每个工具接收图像，返回**结构化文本**（JSON 字符串或可读句子）。
- 工具执行结果以文本形式回填 MLLM 上下文，用 <result>...</result> 标签包裹。

当前状态：第一版 rollout.py 是纯推理（图+查询直接过 MLLM 取 h_end），
不调用任何工具；这两个工具先定义好，第二阶段升级工具调用循环时接入。
"""

import json

import torch


def object_detection(image, category=None):
    """
    目标检测工具（伪工具占位）。

    返回 JSON 文本，格式如：
        {"detections": [{"label": "ship", "bbox": [x1, y1, x2, y2], "conf": 0.9}]}

    当前为占位实现：只返回空结果，保证工具接口可跑通。
    第二阶段可替换为真实检测器（如 YOLO / DETR），接口不变。
    """
    return json.dumps({"detections": []})


# 遥感常见场景类别（UCM 21 类 + 少量扩展），scene_classify 的候选集合
RS_SCENES = [
    "airplane", "airport", "baseball field", "basketball court", "beach",
    "bridge", "buildings", "church", "forest", "freeway", "golf course",
    "harbor", "intersection", "medium residential", "mobile home park",
    "overpass", "parking lot", "playground", "port", "river", "runway",
    "ship", "stadium", "storage tanks", "tennis court", "vegetation",
]


def scene_classify(image, clip_model_name="ViT-L/14", topk=3, model=None, preprocess=None):
    """
    场景分类工具：用冻结 CLIP 对遥感场景类别打相似度。

    做法：把图像编码成特征，与 "a remote sensing image of {场景}" 的
    文本特征算余弦相似度，softmax 后取 top-k。

    model/preprocess 为 None 时自行加载（保持独立可用）；
    传入预加载对象时复用（rollout 批量调用场景，避免每张图重新 load 2GB 模型）。

    注：项目的 CLIP 是魔改版（encode_image/encode_text 强制带 prompt 参数，
    positional embedding 也按 prompt 预留）。工具不需要 prompt，传入
    全零伪 prompt（attention 会自动忽略零 token）。

    返回文本，如：
        "harbor (0.61), port (0.24), ship (0.08)"
    """
    text, _ = scene_classify_with_conf(image, clip_model_name, topk, model, preprocess)
    return text


def scene_classify_with_conf(image, clip_model_name="ViT-L/14", topk=3,
                             model=None, preprocess=None,
                             cls_head=None, cls_names=None):
    """
    scene_classify 的带置信度版本：返回 (文本, top1 置信度)。

    两种模式：
    - 零样本（默认）：CLIP 对 26 个场景类名打 cosine 相似度，返回 top-k。
      置信度用 top1 cosine（max 约 0.19~0.30，有区分度）。
    - 域内分类头（cls_head 提供时）：冻结 CLIP 图像特征 -> 线性头 -> softmax，
      返回 top-k 概率。域内微调后概率有区分度，置信度即 top1 概率。
    """
    from prompt_clip import clip  # 项目魔改 CLIP（训练同款实现；tokenize 为标准 BPE）

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None or preprocess is None:
        model, preprocess = clip.load(clip_model_name, device=device)
        model = model.float().eval()            # 权重统一 fp32（与 vilt_module 一致）

    img = preprocess(image).unsqueeze(0).to(device)
    width = model.visual.conv1.weight.shape[0]     # conv1 输出通道数 = 特征宽度
    zero_prompt = torch.zeros(1, 16, width, dtype=img.dtype, device=img.device)

    with torch.no_grad():
        image_feats = model.encode_image(img, zero_prompt)   # 魔改版强制带 visual_prompt
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        if cls_head is not None:
            logits = cls_head(image_feats)[0]
            probs = logits.softmax(dim=-1)
            names = cls_names if cls_names is not None else RS_SCENES
            top_idx = probs.topk(min(topk, len(names))).indices.tolist()
            text = ", ".join(f"{names[i]} ({probs[i].item():.2f})" for i in top_idx)
            return text, probs[top_idx[0]].item()
        text_feats = model.encode_text(
            clip.tokenize([f"a remote sensing image of {s}" for s in RS_SCENES]).to(device)
        )  # prompt=None 时走无 prompt 路径
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
        logits = (image_feats @ text_feats.t()).softmax(dim=-1)[0]

    cos_sims = (image_feats @ text_feats.t())[0]
    top_idx = logits.topk(min(topk, len(RS_SCENES))).indices.tolist()
    # 文本里用 cosine 相似度（有区分度 0.19~0.36），softmax 概率被 26 类压扁（全 0.04）
    text = ", ".join(f"{RS_SCENES[i]} ({cos_sims[i].item():.2f})" for i in top_idx)
    top1_cos = cos_sims[top_idx[0]].item()
    return text, top1_cos
