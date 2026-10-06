"""
标准 CLIP ViT-L/14 推理实现（基线脚本共享模块）。

要点：
- 这是【标准 OpenAI CLIP】实现（非 CUP 魔改版），零样本/线性探测基线必须用官方行为。
- 权重文件为官方 ViT-L-14.pt，state_dict 键名与原版对齐，直接 load。
- 视觉编码返回 768 维（proj 后），文本编码返回 768 维——与官方一致，
  检索矩阵用二者的余弦相似度。
"""
import gzip
import html
import os
from collections import OrderedDict
from functools import lru_cache

import ftfy
import numpy as np
import regex as re
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)

# ---------------------------------------------------------------------------
# 官方预处理
# ---------------------------------------------------------------------------
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def build_transform():
    return Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


# ---------------------------------------------------------------------------
# 标准 BPE tokenizer（与 OpenAI CLIP simple_tokenizer 一致）
# ---------------------------------------------------------------------------
@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    """官方 CLIP BPE tokenizer（49408 词表）。"""

    def __init__(self, bpe_path):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def tokenize(self, texts, context_length=77):
        """文本列表 → [N, 77] token 张量（SOS/EOS 包裹，尾部补零）。"""
        if isinstance(texts, str):
            texts = [texts]
        sot = self.encoder["<|startoftext|>"]
        eot = self.encoder["<|endoftext|>"]
        all_tokens = [[sot] + self.encode(t)[:context_length - 2] + [eot] for t in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result


# ---------------------------------------------------------------------------
# 视觉 Transformer（ViT-L/14）
# ---------------------------------------------------------------------------
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """官方 ViT-L/14 视觉塔（patch 14, width 1024, 24 层, 16 头）。"""

    def __init__(self, input_resolution=224, patch_size=14, width=1024,
                 layers=24, heads=16, output_dim=768):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.conv1(x)                          # [N, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                     # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                     # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    """官方 OpenAI CLIP 结构：键名与 ViT-L-14.pt 完全对齐。

    文本塔成员直接挂在 CLIP 层（token_embedding / positional_embedding /
    ln_final / text_projection / transformer.resblocks.*），
    与官方 state_dict 键名一致，可 strict 加载。
    """

    def __init__(self, context_length=77, vocab_size=49408, width=768,
                 heads=12, layers=12):
        super().__init__()
        self.visual = VisionTransformer()

        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        self.transformer = Transformer(width, layers, heads, attn_mask=mask)
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, width))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)             # [N, 77, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return logits, image_features, text_features


def load_clip(ckpt_path, device):
    """加载官方 ViT-L-14.pt 权重到标准 CLIP 实现。

    官方权重是 JIT 存储的模型（torch.load 返回 RecursiveScriptModule），
    需取其 state_dict；键名前缀（如 "module."）做一次自适应剥离。
    """
    loaded = torch.load(ckpt_path, map_location="cpu")
    if isinstance(loaded, dict):
        state_dict = loaded
    else:
        state_dict = loaded.state_dict()
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    # jit 常量键（非参数），过滤
    state_dict = {k: v for k, v in state_dict.items()
                  if k not in ("input_resolution", "context_length", "vocab_size")}

    model = CLIP()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"权重缺少键: {missing[:5]} ...")
    if unexpected:
        raise RuntimeError(f"权重多余键: {unexpected[:5]} ...")
    model = model.float().to(device).eval()
    return model


def get_tokenizer(bpe_path):
    return SimpleTokenizer(bpe_path)
