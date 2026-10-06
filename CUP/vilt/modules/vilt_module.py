"""
ViLT (Vision-and-Language Transformer) 核心模块
==============================================
本文件实现了 CUP（Context and Uncertainty-aware Prompt）模型，
用于遥感图像的跨模态检索 —— 给定一张图找到最匹配的文字，或反之。

一句话概括核心思路：
  拿一个强大的预训练 CLIP（冻结不动），在它外面"贴"几层可学习的小组件
  （Prompt、Adapter、Analysis），让 CLIP 适配遥感数据。

为什么只贴组件不微调 CLIP？
  CLIP 有上亿参数，遥感数据只有几千张图，直接微调会立刻过拟合。
  只训练新增的几百万参数，既保留 CLIP 的通用能力，又适配了遥感领域。

各组件职责速览：
  ┌─────────────────┬───────────────────────────────────┐
  │ 组件              │ 一句话作用                         │
  ├─────────────────┼───────────────────────────────────┤
  │ CLIP (self.model) │ 冻结的预训练模型，负责编码图文     │
  │ prompt_embeddings │ 可学习的"虚拟词"，插入文本前面     │
  │ cluster_model    │ 对图像聚类，生成"这张图是什么"先验  │
  │ visual_prompt    │ 聚类先验 + 可学习 prompt 的融合     │
  │ adapter          │ 对 CLIP 输出特征做 ≤10% 的微调     │
  │ analysis         │ 估计每个特征维度的"不确定性"        │
  └─────────────────┴───────────────────────────────────┘

数据流（infer 方法）：
  图像 ──→ cluster_model ──→ visual_prior ──┐
                                            ├──→ 融合 → visual_prompt ──┐
   visual_prompt_embeddings（可学习）────────┘                          │
                                                                        ├──→ CLIP ──→ 特征
   text_prompt_embeddings（可学习）──→ text_prompt ─────────────────────┘
                                            │
   文本 ───────────────────────────────────┘
                                              │
                                    adapter → analysis → 输出
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from vilt.modules import heads, objectives, vilt_utils
import numpy as np
from PIL import Image
from model.context_cluster import cluster_model
from torch_ema import ExponentialMovingAverage
from prompt_clip import clip
from collections import OrderedDict
import math
from torch.nn import Conv2d, Dropout
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair


# ============================================================================
# 第一部分：基础组件
# 这些是模仿 OpenAI CLIP 写的 Transformer 积木块
# ============================================================================

class LayerNorm(nn.LayerNorm):
    """
    自定义 LayerNorm：解决 fp16 类型不一致的 bug。

    背景：PyTorch 原版 LayerNorm 内部用 fp32 计算，算完返回 fp32 结果。
    如果网络其他部分用 fp16（半精度），类型就会对不上，程序报错。
    这个子类做的事情：输入是什么类型，输出就强制转回什么类型。

    类比：你给计算器输入美元，它内部换成人民币算，
    但输出时自动换回美元，不让你感知到中间过程。
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype                        # 记住输入类型（fp16 还是 fp32）
        ret = super().forward(x)                    # 调用父类 nn.LayerNorm 完成真正的归一化计算
        return ret.type(orig_type)                  # 输出转回原始类型


class QuickGELU(nn.Module):
    """
    GELU 激活函数的快速近似版。

    什么是激活函数？神经网络如果全是 y=Wx+b 这种直线变换，
    堆 100 层也等于 1 层。必须插入"非线性"函数才能拟合复杂模式。

    GELU 精确版涉及误差函数 erf，计算慢。QuickGELU 用 sigmoid 近似：
        QuickGELU(x) = x × sigmoid(1.702 × x)

    效果：
      - x 很大正数 → sigmoid≈1 → 输出≈x（放行）
      - x 很大负数 → sigmoid≈0 → 输出≈0（阻断）
      - x 在中间  → sigmoid∈(0,1) → 输出部分通过

    类比：一扇自动门，你越靠近它开得越大，离远了就关上。
    1.702 是经验系数，让近似曲线最接近真实 GELU。
    这是 OpenAI CLIP 原版使用的激活函数。
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Transformer 的一个"积木块"，GPT 和 CLIP 都由这种块堆叠而成。

    结构（Pre-Norm 风格，LayerNorm 放在子层之前）：
        x → LayerNorm → Self-Attention → 和原来的x相加 → LayerNorm → MLP → 和原来的x相加 → 输出

    两个子层各做了什么：
      1. Self-Attention（自注意力）：每个词去"看"所有其他词，汇总相关信息
         例：句子"猫坐在垫子上"
            "坐"会重点关注"猫"（谁？）和"垫子"（哪？）
            "猫"会重点关注"坐"（在干嘛？）

      2. MLP（前馈网络）：每个词独立处理，先升维4倍再缩回来
         d_model → 4×d_model → GELU → d_model
         升维给网络更多"思考空间"，缩回来保持维度一致

    "残差连接"是什么？x_new = x_old + f(x_old)
    这是在说：新的输出 = 原来的 + 子层算出的"修正量"
    好处：如果子层学坏了（输出全0），x_old 还在，不会更差。
    类比：考试改答案——改对了加分，改错了至少还有原始答案兜底。
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)      # 多头自注意力层：同时从多个"角度"看信息
        self.ln_1 = LayerNorm(d_model)                           # 注意力前的归一化
        self.mlp = nn.Sequential(OrderedDict([                   # 前馈网络（两个 Linear 夹一个激活）
            ("c_fc", nn.Linear(d_model, d_model * 4)),           # 第1步：768 → 3072，升维扩大容量
            ("gelu", QuickGELU()),                                # 第2步：非线性激活，让信息过滤
            ("c_proj", nn.Linear(d_model * 4, d_model))          # 第3步：3072 → 768，降维回到原大小
        ]))
        self.ln_2 = LayerNorm(d_model)                           # MLP 前的归一化
        self.attn_mask = attn_mask                               # 注意力掩码：控制哪些位置可以互看

    def attention(self, x: torch.Tensor):
        """
        计算自注意力。Q（查询）、K（键）、V（值）都来自同一个 x，
        所以叫"自"注意力 —— 序列内部自己看自己。
        """
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        前向传播：Pre-Norm + 残差连接，两步走。
        注意顺序：先归一化，再计算，最后加回原值（不是先计算再加再归一化）。
        这种 Pre-Norm 方式训练更稳定，是 GPT-2/CLIP 的选择。
        """
        x = x + self.attention(self.ln_1(x))     # 子层1：归一化→注意力→残差加回
        x = x + self.mlp(self.ln_2(x))            # 子层2：归一化→MLP→残差加回
        return x


class Transformer(nn.Module):
    """
    完整的 Transformer：把 N 个 ResidualAttentionBlock 串起来。

    类比：12 个积木块首尾相连，数据依次流过，
    每过一个块就对输入的理解更"深"一层。
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width          # 特征维度，比如 768
        self.layers = layers        # 层数，比如 12
        # Sequential 把多个模块串成流水线：数据流过 Block1 → Block2 → ... → Block12
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# ============================================================================
# 第二部分：权重初始化
# ============================================================================

def init_weights(module):
    """
    给模型参数赋初始值。为什么不能全部初始化为 0？
    如果所有参数都是 0，每个神经元算出一样的结果，等于只有一个神经元。
    用随机小值初始化，打破对称性，不同神经元才能学到不同东西。

    规则：
      - Linear / Embedding：正态分布 N(0, 0.02)，均值0标准差0.02
      - LayerNorm：bias=0（不偏移），weight=1（不缩放），即初始什么都不改
      - Linear 的 bias：全部置零
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)   # 小随机数打破对称性
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()                           # bias 初始为 0
        module.weight.data.fill_(1.0)                      # weight 初始为 1（不缩放）

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()                           # Linear 的 bias 也清零


# ============================================================================
# 第三部分：核心模型 ViLTransformerSS
# ============================================================================
# 这是整个文件的心脏。它继承 PyTorch Lightning 的 LightningModule，
# 把模型定义、训练步骤、验证步骤、优化器配置全部封装在一起。
#
# 类名中的 "SS" = Single Stream，表示图文在同一个 Transformer 中处理
# （不过 CUP 实际使用的是 CLIP 双塔结构，这个名字是 ViLT 框架的历史遗留）
# ============================================================================

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        """
        模型初始化 —— 把所有的"积木"搭建起来。

        可以把整个过程想象成组装一台改装车：
          - CLIP 是原装发动机（冻结，不动它）
          - Prompt 是加装的进气系统（让输入更适合遥感）
          - Adapter 是加装的尾气调节（微调输出）
          - Analysis 是加装的诊断仪（告诉你哪里不确定）
        """
        super().__init__()
        self.save_hyperparameters()                    # Lightning 提供：保存所有超参数到 checkpoint
        self.test_only = False                         # 是否只做测试（不训练）
        vilt_utils.set_metrics(self)                   # 初始化评估指标（Accuracy、Scalar 等）
        self.current_tasks = list()                    # 当前激活的任务列表，由配置中的 loss_names 决定

        # 打印配置以供检查
        print("\n###########This is config ####################")
        print(self.hparams.config)
        print("###########This is config ####################\n")

        # ------ 损失权重超参数（从配置文件读取）------
        self.loss_scale = self.hparams.config["loss_scale"]          # 总损失的缩放系数
        self.loss_scale_kl = self.hparams.config["loss_scale_kl"]    # KL 散度损失的权重
        self.loss_scale_un = self.hparams.config["loss_scale_un"]    # 不确定性损失的权重

        # ================================
        # ① 加载 CLIP 作为骨干网络（这个不训练，冻结）
        # ================================
        self.name = config["clip_model"]               # 比如 "ViT-L/14" 或 "ViT-B/32"
        self.model, self.preprocess = clip.load(self.name)  # clip.load 返回 (模型, 图像预处理函数)
        self.model = self.model.float()                # 统一转 float32

        self.prompt_length = config['prompt_length']   # prompt 长度，如 16，即 16 个"虚拟 token"
        self.prompt_dropout = Dropout(0.1)             # Dropout：训练时随机把 10% 的值变 0，防过拟合

        # ================================
        # ② 文本 Prompt 分支
        # ================================
        # 一段文字送入 CLIP 前，在开头插入 16 个"虚拟词"。
        # 这 16 个虚拟词没有实际含义（不是 "hello"），但训练中自动学到
        # "这是遥感图像的文本，注意领域术语"这种任务上下文。
        #
        # 流程：prompt_embeddings(查表) → Linear投影(润色) → Dropout(防过拟合) → 拼到文本前面 → 送入 CLIP

        prompt_dim = config["txt_features_dim"]         # 文本维度：ViT-L 是 768，ViT-B 是 512
        self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)  # 一层线性投影，给 prompt 更多表达灵活性
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')

        val = math.sqrt(6. / float(1 + prompt_dim))     # Xavier 初始化的边界值

        # 这里是整个文本 prompt 分支最核心的参数：
        # 形状 (1, 16, 768)：1=占位batch, 16=prompt长度, 768=每个位置的维度
        # nn.Parameter 表示"这个张量需要在训练中被优化器更新"
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.prompt_length, prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)  # 用均匀分布随机初始化

        # ================================
        # ③ 视觉 Prompt 分支
        # ================================
        # 比文本 prompt 复杂：视觉 prompt 由两个来源融合而成。
        # 来源A（聚类先验）：cluster_model 根据图像内容动态生成，每张图不一样
        # 来源B（可学习）：visual_prompt_embeddings，和文本 prompt 类似，通用的
        #
        # 融合公式：visual_prompt = (1-ratio)×聚类先验 + ratio×可学习prompt
        # ratio 由 cluster_mapping 自动学习，限制在 [0,1]

        self.visual_prompt_dropout = Dropout(0.1)        # dropout 防过拟合
        visual_patch_size = config['image_size']          # 特征图尺寸，如 [14,14] 或 [32,32]

        visual_prompt_dim = config['img_features_dim']    # 图像维度：ViT-L 是 1024，ViT-B 是 768
        self.visual_prompt_proj = nn.Linear(visual_prompt_dim, visual_prompt_dim)
        nn.init.kaiming_normal_(self.visual_prompt_proj.weight, a=0, mode='fan_out')

        visual_val = math.sqrt(6. / float(3 * reduce(mul, visual_patch_size, 1) + visual_prompt_dim))

        # 可学习的视觉 prompt，类比文本的 prompt_embeddings
        self.visual_prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.prompt_length, visual_prompt_dim))
        nn.init.uniform_(self.visual_prompt_embeddings.data, -visual_val, visual_val)

        # 融合后的 prompt 再额外过一层处理（投影 + dropout），增强表达能力
        self.visual_prompt_gather_dropout = Dropout(0.1)
        self.visual_prompt_proj_gather = nn.Linear(visual_prompt_dim, visual_prompt_dim)
        nn.init.kaiming_normal_(self.visual_prompt_proj_gather.weight, a=0, mode='fan_out')

        nn.init.uniform_(self.visual_prompt_embeddings.data, -visual_val, visual_val)

        # ================================
        # ④ Analysis 模块 —— 不确定性估计
        # ================================
        # 只有一层 Linear。输入 CLIP 输出的特征（768维），输出同样维度。
        # 输出被 clamp(min=0) 强制非负，每维的值 = 该维度的"不确定性/标准差"。
        #
        # 在损失函数中，这个值被当作噪声幅度：不确定性大的维度被注入更多噪声，
        # 导致匹配不准→损失变大。模型为了降低损失，被迫让不确定性变小。
        # 本质是一个自我监督机制：不需要人工标注"哪里不确定"。

        self.analysis_img = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),           # 768 → 768
        )
        self.analysis_txt = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
        )

        # ================================
        # ⑤ Prior（先验）相关模块
        # ================================
        # prior_img：维度转换，视觉维度(1024) → 文本维度(768)
        # prior_analysis_img：输出被 clamp≥0，作为 KL 散度的目标分布
        #   意思是：可学习 prompt 的分布"应该像"聚类先验的分布

        self.prior_img = nn.Linear(visual_prompt_dim, prompt_dim)
        self.prior_analysis_img = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
        )

        # ================================
        # ⑥ Cluster Model（聚类模型）—— "上下文感知"的来源
        # ================================
        # 一个轻量级图像聚类网络（基于 ICLR'23 的 ContextCluster）。
        # 输入图像，输出每张图的"聚类特征"→ 这就是视觉先验 prompt 的原材料。
        #
        # cluster_mapping：Linear(visual_dim → 1)，输出一个标量，
        # 经 clamp(0,1) 后成为 ratio——决定先验和可学习 prompt 的融合比例。
        # ratio 是模型自己学的，不需要人工设定。

        self.cluster_model = cluster_model(num_classes=visual_prompt_dim)
        self.cluster_mapping = nn.Linear(visual_prompt_dim, 1)     # 高维 → 1 个标量

        # ================================
        # ⑥b Agent 先验投影（可选分支，默认关闭）
        # ================================
        # 实验开关：use_agent_prior=True 时，视觉先验不再来自 cluster_model，
        # 而是来自离线 MLLM（agent）生成的 h_end（图+查询文本的推理状态）。
        # agent_proj 把 MLLM 隐藏态（3584 维，Qwen2.5-VL-7B）投影到
        # visual_prompt_dim（1024/768），之后的 prior_img、repeat、融合、
        # KLD 蒸馏全部复用原代码，接口不变。
        # 默认 False = 原版 CCB 行为，完全不受影响。
        self.use_agent_prior = config.get("use_agent_prior", False)
        agent_hidden_dim = config.get("agent_hidden_dim", 3584)   # Qwen2.5-VL-7B 隐藏维
        self.agent_proj = nn.Sequential(
            nn.Linear(agent_hidden_dim, visual_prompt_dim),       # 3584 → 1024(ViT-L) / 768(ViT-B)
            LayerNorm(visual_prompt_dim),                          # 用本文件的 LayerNorm（fp16 兼容）
        )

        # ================================
        # ⑦ Adapter 模块 —— 特征微调器
        # ================================
        # 一个小 MLP：Linear → GELU → Linear，输入输出维度相同。
        # 当"残差"使用：final = ratio × adapter_out + (1-ratio) × original
        # ratio 被强制 clamp 在 [0, 0.1]，即 adapter 最多修正 10%。
        # 为什么要限制 10%？防止 adapter 过度修改 CLIP 的好特征。

        self.adapter_img = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),           # 768 → 768
            nn.GELU(),                                    # 非线性激活
            nn.Linear(prompt_dim, prompt_dim),           # 768 → 768
        )
        self.adapter_img_mapping = nn.Sequential(
            nn.Linear(prompt_dim, 1),                    # 768 → 1：自动学修正比例
        )

        self.adapter_txt = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.adapter_txt_mapping = nn.Sequential(
            nn.Linear(prompt_dim, 1),
        )

        # ================================
        # ⑧ 加载预训练权重（仅测试模式）
        # ================================
        # strict=False 允许只加载部分权重，新增的组件从随机初始化开始
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    # ========================================================================
    # infer()：模型前向传播的核心入口
    # ========================================================================
    # 每次训练/验证/测试都会调用这个方法。
    # 它完整实现了 CUP 的 12 步数据流：聚类→先验→融合→CLIP→Adapter→Analysis
    #
    # 简化版流程图：
    #   img ─→ cluster ─→ prior ─┐
    #                             ├─→ visual_prompt ─┐
    #   learnable ──→ random ────┘                   ├─→ CLIP ─→ feat ─→ adapter ─→ analysis
    #   learnable ──→ text_prompt ───────────────────┘
    #                                                txt ────────────┘
    # ========================================================================

    def infer(
            self,
            batch,
            agent_hidden=None,
    ):
        """
        核心前向传播。输入一个 batch 的 (图像, 文本)，输出特征和中间结果。

        参数 batch: 元组 (img, txt)
          img: (B, 3, 224, 224)，B 是 batch 大小
          txt: (B, 61)，每个文本被 tokenize 成 61 个 token
               （77 - 16 = 61，因为要给 prompt 留 16 个位置）

        参数 agent_hidden: [B, 3584] 离线 MLLM 先验（仅 use_agent_prior=True 时
          需要，由 compute_clip 从 batch 传入；原版路径恒为 None）
        """
        # --- 步骤 1：解包 ---
        (img, txt) = batch
        B = txt.shape[0]                                 # batch size，这一批有多少个样本

        # --- 步骤 2：生成视觉先验 ---
        # 原版：聚类模型，输入 (B,3,224,224) → 输出 (B, D_visual)
        # agent 版：h_end 经 agent_proj 投影，输出同样 (B, D_visual)，下游不变
        if self.use_agent_prior:
            visual_prior_prompt = self.agent_proj(agent_hidden.float().to(img.device))
        else:
            visual_prior_prompt = self.cluster_model(img)

        # --- 步骤 3：为 KL 散度生成先验分布 ---
        # prior_img：把视觉维度转成文本维度（维度对齐）
        visual_prior_prompt_out = self.prior_img(visual_prior_prompt)
        # prior_analysis_img → clamp(≥0)：非负约束，后面作为 KL 散度的目标分布
        # KL 散度要求输入是概率分布（非负），所以必须 clamp
        visual_prior_prompt_prior = torch.clamp(self.prior_analysis_img(visual_prior_prompt_out), min=0)

        # --- 步骤 4：把先验扩展到 prompt 长度 ---
        # unsqueeze: (B, D) → (B, 1, D)，插入一个维度
        # repeat:    (B, 1, D) → (B, 16, D)，复制 16 份
        # 为什么要 16 份？因为可学习 prompt 也是 16 个位置，要一一对应做加权
        visual_prior_prompt = visual_prior_prompt.unsqueeze(dim=1).repeat(1, self.prompt_length, 1)

        # --- 步骤 5：计算融合比例 visual_ratio ---
        # cluster_mapping 输入 (B, 16, D_visual)，输出 (B, 16, 1)
        # torch.clip 把值限制在 [0, 1]
        # ratio 的含义：ratio→0 = 更信先验，ratio→1 = 更信可学习 prompt
        # 每个样本、每个 prompt 位置都有独立的 ratio
        visual_ratio = torch.clip(self.cluster_mapping(visual_prior_prompt), min=0, max=1)

        # --- 步骤 6：构建可学习的视觉 prompt ---
        # prompt_embeddings (1, 16, D) → expand 到 (B, 16, D) → Linear 投影 → Dropout
        # expand 不复制数据只改视图，省显存；所有样本共享同一套可学习参数
        visual_random_prompt = self.visual_prompt_dropout(
            self.visual_prompt_proj(self.visual_prompt_embeddings).expand(B, -1, -1))

        # --- 步骤 7：融合！⭐ CUP 最核心的一行代码 ⭐ ---
        # 加权平均：final = (1-ratio)×先验 + ratio×可学习
        # 如果 ratio=0.2：80%来自图像自己，20%来自通用知识
        # 如果 ratio=0.8：20%来自图像自己，80%来自通用知识
        visual_prompt = (1 - visual_ratio) * visual_prior_prompt + visual_ratio * visual_random_prompt
        # 融合后再过一层投影，让融合后的 prompt 更协调
        visual_prompt = self.visual_prompt_gather_dropout(self.visual_prompt_proj_gather(visual_prompt))

        # --- 步骤 8：构建文本 prompt ---
        # 文本侧就简单了：可学习 embeddings → 投影 → Dropout，没有聚类先验
        text_prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))

        # --- 步骤 9：CLIP 前向传播 ---
        # CLIP 内部会把 visual_prompt 插入图像 token 序列最前面（CLS 之后）
        # 把 text_prompt 插入文本 token 序列最前面（SOS 之后）
        # 返回 4 个东西：
        #   logits_per_image: (B,B) 图-文相似度矩阵
        #   logits_per_text:  (B,B) 文-图相似度矩阵（转置）
        #   image_features:   (B,1024) 图像特征向量
        #   text_features:    (B,768)  文本特征向量
        logits_per_image, logits_per_text, image_features, text_features = self.model(img, txt, visual_prompt,
                                                                                      text_prompt)

        # --- 步骤 10：Adapter 残差微调（图像侧）---
        # adapter 输出一个"修正建议"，然后和原特征混合
        # ratio 被 clamp 在 [0, 0.1]：adapter 最多只能改 10%
        # 这保证了 CLIP 原始特征占主导，adapter 只做微小领域修正
        img_adapter = self.adapter_img(image_features)
        img_adapter_ratio = self.adapter_img_mapping(img_adapter)
        img_adapter_ratio = torch.clamp(img_adapter_ratio, min=0, max=0.1)   # 关键：上限 0.1！
        image_features = img_adapter_ratio * img_adapter + (1 - img_adapter_ratio) * image_features

        # --- 步骤 11：Adapter 残差微调（文本侧）---
        # 和图像侧完全一样的逻辑
        txt_adapter = self.adapter_txt(text_features)
        txt_adapter_scale = self.adapter_txt_mapping(txt_adapter)
        txt_adapter_scale = torch.clamp(txt_adapter_scale, min=0, max=0.1)
        text_features = txt_adapter_scale * txt_adapter + (1 - txt_adapter_scale) * text_features

        # --- 步骤 12：不确定性估计 ---
        # analysis 输出特征每个维度的"不确定性"，clamp 强制非负
        # 值越大 → 这一维越不确定 → 在损失函数中被注入更多噪声 → 损失变大
        # 模型为了降低损失，被迫让这些值变小（变得确定）
        img_analy = self.analysis_img(image_features)
        txt_analy = self.analysis_txt(text_features)
        img_analy = torch.clamp(img_analy, min=0)        # 标准差不能为负
        txt_analy = torch.clamp(txt_analy, min=0)

        # --- 汇总返回 ---
        # 返回这么多东西，是因为损失函数 compute_clip 需要这些中间结果
        # 来分别计算：对比损失、KL 损失、不确定性损失
        loss_scale = self.loss_scale

        ret = {
            "visual_prior_prompt": visual_prior_prompt_out,          # 聚类先验（维度转换后）
            "visual_prior_prompt_prior": visual_prior_prompt_prior,   # 非负先验（KL 的目标分布）
            "loss_scale": loss_scale,                                  # 总损失缩放系数
            "loss_scale_kl": self.loss_scale_kl,                      # KL 损失权重
            "loss_scale_un": self.loss_scale_un,                      # 不确定性损失权重
            "img_analysis": img_analy,                                 # 图像不确定性估计
            "txt_analysis": txt_analy,                                 # 文本不确定性估计
            "image_features": image_features,                          # 最终图像特征
            "text_features": text_features,                            # 最终文本特征
            "logits_per_image": logits_per_image,                      # 图像→文本相似度矩阵
            "logits_per_text": logits_per_text,                        # 文本→图像相似度矩阵
        }

        return ret

    def txt_embeds(
            self,
            batch
    ):
        """
        纯文本编码：只给文本，只出文本特征。不管图像。

        用于：离线建文本索引库。训练完后，把所有候选文本编码存起来，
        检索时直接算相似度，不用每次都重新编码文本。
        """
        txt = batch
        B = txt.shape[0]

        # 构建文本 prompt（和 infer 中一样）
        text_prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
        # 只调 CLIP 的文本编码器，省掉图像编码的计算
        text_features = self.model.encode_text(txt, text_prompt)

        # adapter 残差微调
        txt_adapter = self.adapter_txt(text_features)
        txt_adapter_scale = self.adapter_txt_mapping(txt_adapter)
        txt_adapter_scale = torch.clamp(txt_adapter_scale, min=0, max=0.1)
        text_features = txt_adapter_scale * txt_adapter + (1 - txt_adapter_scale) * text_features

        # L2 归一化：把向量长度变成 1
        # 归一化后，两个向量的内积 = 余弦相似度，计算检索排序很方便
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        ret = {
            "text_feats": text_features,
        }

        return ret

    def img_embeds(
            self,
            batch,
            agent_hidden=None,
    ):
        """
        纯图像编码：只给图像，只出图像特征。不管文本。

        用于：离线建图像索引库。和 txt_embeds 配套使用。

        参数 agent_hidden: 同 infer，use_agent_prior=True 时由调用方传入。
        """
        img = batch
        B = img.shape[0]

        # 构建视觉 prompt（和 infer 中完全一样的融合逻辑）
        if self.use_agent_prior:
            visual_prior_prompt = self.agent_proj(agent_hidden.float().to(img.device))
        else:
            visual_prior_prompt = self.cluster_model(img)
        visual_prior_prompt = visual_prior_prompt.unsqueeze(dim=1).repeat(1, self.prompt_length, 1)
        visual_ratio = torch.clamp(self.cluster_mapping(visual_prior_prompt), min=0, max=1)
        visual_random_prompt = self.visual_prompt_dropout(
            self.visual_prompt_proj(self.visual_prompt_embeddings).expand(B, -1, -1))
        visual_prompt = (1 - visual_ratio) * visual_prior_prompt + visual_ratio * visual_random_prompt
        visual_prompt = self.visual_prompt_gather_dropout(self.visual_prompt_proj_gather(visual_prompt))

        # 只调 CLIP 的图像编码器
        image_features = self.model.encode_image(img, visual_prompt)

        # adapter 残差微调
        img_adapter = self.adapter_img(image_features)
        img_adapter_ratio = self.adapter_img_mapping(img_adapter)
        img_adapter_ratio = torch.clamp(img_adapter_ratio, min=0, max=0.1)
        image_features = img_adapter_ratio * img_adapter + (1 - img_adapter_ratio) * image_features

        # L2 归一化
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        ret = {
            "image_feats": image_features,
        }

        return ret

    # ========================================================================
    # forward()：训练时的调度中心
    # ========================================================================
    # 根据当前激活的任务列表，把 batch 分发给对应的损失函数。
    # 对于 CUP 遥感检索：loss_names = {"clip": 1} → current_tasks = ["clip"]
    # 所以只会走 "clip" 分支 → 调用 objectives.compute_clip()
    #
    # 其他任务（mlm, mpp, itm 等）是 ViLT 框架支持的预训练任务，
    # CUP 实验中没有用到，保留是为了兼容原始 ViLT 代码。

    def forward(self, batch):
        ret = dict()

        # 没有指定任务 → 直接做推理，返回特征
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # CLIP 对比学习 —— CUP 的核心任务，遥感检索就走这条路
        if "clip" in self.current_tasks:
            ret.update(objectives.compute_clip(self, batch))

        # 以下任务 CUP 实验中不使用，保留是为了兼容 ViLT 框架
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    # ========================================================================
    # PyTorch Lightning 标准接口：训练/验证/测试
    # Lightning 自动处理 GPU 分配、梯度计算、日志、checkpoint 等
    # 你只需要写每个步骤做什么
    # ========================================================================

    def training_step(self, batch, batch_idx):
        """
        每个 batch 的训练步骤：
          1. 设置任务列表
          2. forward() → compute_clip() → 返回 {"clip_loss": ..., "prior_loss": ..., "uncertainty_loss": ...}
          3. 把所有带 "loss" 键的值加起来作为总损失
          4. Lightning 自动对 total_loss 做 backward() 计算梯度
          5. 优化器用梯度更新参数 —— 但 CLIP 被冻结了，只有新增组件被更新
        """
        vilt_utils.set_task(self)                      # 设置 current_tasks，比如 ["clip"]
        output = self(batch)                            # 前向传播 → 拿到所有损失
        total_loss = sum([v for k, v in output.items() if "loss" in k])  # 把所有子损失加在一起
        return total_loss

    def training_epoch_end(self, outs):
        """每个 epoch 训练结束后：记录指标到日志"""
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        """
        验证步骤：和训练类似，但 Lightning 自动关掉 dropout 和梯度计算。
        不需要返回 loss，Lightning 只关心指标。
        """
        self.hparams.config["current_epoch"] = self.current_epoch
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        """每个 epoch 验证结束后：计算 Recall@K 等检索指标"""
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()
        return ret

    def test_epoch_end(self, outs):
        """测试结束后打印最终检索结果"""
        self.test_only = True
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。具体逻辑在 vilt_utils.set_schedule() 中：
          - 优化器：AdamW
          - 参数分组：4 组（主体/head × 有/无 weight_decay）
          - 学习率：head 用 lr × lr_mult（更大），主体用 lr
          - 调度：warmup（预热） + 余弦退火（逐渐降低）
        """
        return vilt_utils.set_schedule(self)
