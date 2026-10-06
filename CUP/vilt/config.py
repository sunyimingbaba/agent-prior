"""
ViLT / CLIP 多模态实验配置文件
===============================
基于 Sacred 框架（https://github.com/IDSIA/sacred），用于管理实验的配置、日志与复现。

Sacred 核心概念：
- Experiment: 一个实验的顶层容器
- @ex.config: 定义默认配置（基础参数）
- @ex.named_config: 定义可插拔的"命名配置"，运行时可通过命令行组合叠加

使用方式示例：
    python run.py with task_finetune_irtr_rsicd_randaug_ViTB32 step25k
"""

from sacred import Experiment

# ============================================================================
# 创建 Experiment 实例
# ============================================================================
ex = Experiment("ViLT")


# ============================================================================
# 辅助函数：构建损失函数权重字典
# ============================================================================
def _loss_names(d):
    """
    返回一个包含所有可能损失项的字典，值为对应的权重（0 表示不使用该损失）。
    支持的损失类型：
        itm   - Image-Text Matching（图文匹配损失）
        mlm   - Masked Language Modeling（掩码语言建模损失）
        mpp   - Masked Patch Prediction（掩码图像块预测损失）
        vqa   - Visual Question Answering（视觉问答损失）
        nlvr2 - Natural Language for Visual Reasoning（自然语言视觉推理损失）
        irtr  - Image-Text Retrieval（图文检索损失，通常是对比损失）
        clip  - CLIP 风格的对比学习损失
    """
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "clip": 0,      # 新增：CLIP 对比损失，原版 ViLT 中没有
    }
    ret.update(d)        # 用传入的权重覆盖默认值
    return ret


# ============================================================================
# 默认配置（基础参数）
# ============================================================================
@ex.config
def config():
    # ---- 实验元信息 ----
    exp_name = "vilt"                            # 实验名称
    seed = 0                                     # 随机种子，保证可复现性
    datasets = ["coco", "vg", "sbu", "gcc"]      # 预训练数据集列表：COCO, Visual Genome, SBU, GCC
    loss_names = _loss_names({"itm": 1, "mlm": 1})  # 默认启用 ITM + MLM 两个预训练任务
    batch_size = 4096                            # 目标 batch size；实际 per-GPU batch 较小时，PL Trainer 会自动累积梯度
    clip_model = "ViT-L/14"                      # CLIP 图像编码器型号
    current_epoch = 0                            # 当前 epoch（用于断点续训）
    weight_margin = 0.015                        # 对比损失中的 margin 超参数
    loss_scale_kl = 1                            # KL 散度损失的缩放系数
    loss_scale_un = 1                            # 不确定性损失的缩放系数
    loss_scale = 1                               # 总损失的全局缩放系数

    # ---- 图像相关设置 ----
    max_image_len = -1                           # 图像 token 最大长度，-1 表示使用全部
    patch_size = 32                              # ViT 的图像分块大小（32×32 像素/块）
    draw_false_image = 1                         # ITM 任务中负样本图像的采样数量
    image_only = False                           # 是否仅使用图像编码器（纯视觉模式）

    # ---- 文本相关设置 ----
    vqav2_label_size = 3129                      # VQA v2 数据集的答案类别数（3129 个候选答案）
    max_text_len = 77                            # 文本 token 最大长度（BERT/CLIP 标准为 77）
    tokenizer = "bert-base-uncased"              # 使用的分词器
    vocab_size = 30522                           # 词汇表大小（BERT-base 为 30522）
    whole_word_masking = False                   # 是否启用全词掩码（Whole Word Masking）
    mlm_prob = 0.15                              # MLM 任务中 token 被掩码的概率（BERT 标准为 15%）
    draw_false_text = 0                          # ITM 任务中负样本文本的采样数量

    # ---- Transformer 结构设置（ViT-base 参数） ----
    vit = "vit_base_patch32_384"                 # ViT 变体名称：Base 版本，32×32 patch，384 输入分辨率
    hidden_size = 768                            # Transformer 隐藏层维度
    num_heads = 12                               # 多头注意力头数
    num_layers = 12                              # Transformer 层数
    mlp_ratio = 4                                # MLP 前馈网络的维度扩展倍数（4×768 = 3072）
    drop_rate = 0.1                              # Dropout 比率
    prompt_length = 16                           # 可学习 prompt token 的长度（用于 Prompt Learning）

    # ---- 优化器设置 ----
    optim_type = "adamw"                         # 优化器类型：AdamW
    learning_rate = 1e-4                         # 初始学习率
    weight_decay = 0.01                          # 权重衰减（L2 正则化系数）
    decay_power = 1                              # 学习率衰减的幂次（余弦退火调度参数）
    max_epoch = 100                              # 最大训练轮数
    max_steps = 25000                            # 最大训练步数（与 max_epoch 满足其一即停止）
    warmup_steps = 2500                          # 学习率预热步数
    end_lr = 0                                   # 学习率调度结束时的最终学习率
    lr_mult = 1                                  # 下游任务头部的学习率倍数（头部通常用更大学习率）
    clip_zero = True                             # 是否将 CLIP 特征的零向量位置进行特殊处理

    # ---- 下游任务设置 ----
    get_recall_metric = False                    # 是否计算 Recall@K 检索指标

    # ---- PyTorch Lightning Trainer 设置 ----
    resume_from = None                           # 断点续训的 checkpoint 路径
    fast_dev_run = False                         # 快速开发模式：只跑几个 batch 测试流程
    val_check_interval = 1.0                     # 验证间隔（1.0 = 每个 epoch 验证一次，0.5 = 半个 epoch 一次）
    test_only = False                            # 是否仅进行测试（跳过训练）

    # ---- 环境相关参数（随运行环境变化） ----
    data_root = ""                               # 数据集根目录
    log_dir = "/data/CLIP_wyj/result"            # 日志和模型保存目录
    per_gpu_batchsize = 0                        # 每 GPU 的 batch size，需手动指定
    num_gpus = 1                                 # GPU 数量
    num_nodes = 1                                # 节点数量（分布式训练）
    load_path = ""                               # 预训练权重加载路径
    num_workers = 8                              # DataLoader 的 worker 进程数
    precision = 16                               # 混合精度训练（16 表示 FP16）
    linear_projector = 0.1                       # 线性投影层的初始化标准差

    # ---- 超参数：CLIP 特征维度与图像处理 ----
    train_transform_keys = "ViT-L/14"            # 训练时的数据增强策略（与 CLIP 模型绑定）
    val_transform_keys = "ViT-L/14"              # 验证时的数据预处理策略
    image_size = [14, 14]                        # CLIP 输出的特征图尺寸（ViT-L/14 → 14×14 个 patch）
    img_features_dim = 1024                      # 图像特征维度（ViT-L 为 1024）
    txt_features_dim = 768                       # 文本特征维度（CLIP 文本编码器输出 768）

    # ---- 遥感数据集特有参数 ----
    cls_number = 7                               # 数据集类别数（默认 7，不同数据集会覆盖）
    save_image_path = '/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Matchloss-cliploss/images/RSICD'  # 图像保存路径
    json = "/home/amax/wyj/dataset/RSITMD/dataset_RSITMD.json"  # 数据集标注 JSON 文件路径

    # ---- agent 先验（MLLM 离线 h_end 替换 CCB 聚类先验，默认关闭 = 原版行为） ----
    use_agent_prior = False                      # True 时视觉先验来自 agent_proj(h_end) 而非 cluster_model
    agent_cache_dir = ""                         # rollout.py 生成的 h_end 缓存根目录
    agent_hidden_dim = 3584                      # MLLM 隐藏维（Qwen2.5-VL-7B = 3584）


# ============================================================================
# 命名配置 1：环境配置（env）
# ============================================================================
@ex.named_config
def env_dandelin():
    """Dandelin 服务器的环境配置：8 GPU，自定义数据路径"""
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/vilt/result"
    num_gpus = 8
    num_nodes = 1


# ============================================================================
# 命名配置 2：预训练任务（pretraining tasks）
# ============================================================================
@ex.named_config
def task_mlm_itm():
    """预训练：MLM + ITM（掩码语言建模 + 图文匹配）"""
    exp_name = "mlm_itm"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200          # 图像侧最多取 200 个 patch token


@ex.named_config
def task_mlm_itm_randaug():
    """预训练：MLM + ITM + RandAugment 数据增强"""
    exp_name = "mlm_itm_randaug"
    datasets = ["coco", "vg", "sbu", "gcc"]
    train_transform_keys = ["pixelbert_randaug"]   # 使用 Pixel-BERT 风格的 RandAug
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_mpp():
    """预训练：MLM + ITM + MPP（额外加入掩码图像块预测）"""
    exp_name = "mlm_itm_mpp"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mpp": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


# ============================================================================
# 命名配置 3：下游任务微调 — NLVR2（自然语言视觉推理）
# ============================================================================
@ex.named_config
def task_finetune_nlvr2():
    """微调 NLVR2：判断两张图片是否都匹配文字描述"""
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None             # 不按步数限制，只按 epoch
    warmup_steps = 0.1           # 0.1 表示 10% 的训练步数用于 warmup
    draw_false_image = 0         # NLVR2 不需要额外负样本
    learning_rate = 1e-4


@ex.named_config
def task_finetune_nlvr2_randaug():
    """微调 NLVR2 + RandAug"""
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


# ============================================================================
# 命名配置 4：下游任务微调 — VQA（视觉问答）
# ============================================================================
@ex.named_config
def task_finetune_vqa():
    """微调 VQA v2：给定图片和问题，预测答案"""
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1    # 每个 epoch 验证 10 次
    lr_mult = 10                # 分类头部学习率 ×10（头部需要更快收敛）


@ex.named_config
def task_finetune_vqa_randaug():
    """微调 VQA v2 + RandAug"""
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


# ============================================================================
# 命名配置 5：下游任务微调 — IRTR（图文检索）通用数据集
# ============================================================================
@ex.named_config
def task_finetune_irtr_coco():
    """微调图文检索：COCO 数据集"""
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})  # 检索损失为主，匹配损失为辅助（权重 0.5）
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True     # 图文检索需要计算 Recall@K
    draw_false_text = 15         # 负样本文本数量（用于 ITM 的 hard negative mining）
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_coco_randaug():
    """微调图文检索：COCO + RandAug"""
    exp_name = "finetune_irtr_coco_randaug"
    datasets = ["coco"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k():
    """微调图文检索：Flickr30K 数据集"""
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k_randaug():
    """微调图文检索：Flickr30K + RandAug"""
    exp_name = "finetune_irtr_f30k_randaug"
    datasets = ["f30k"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


# ============================================================================
# 命名配置 6：下游任务微调 — IRTR 遥感数据集（ViT-L/14 骨干）
#
# 遥感数据集说明：
#   - Sydney: 悉尼遥感图像数据集（cls_number=7）
#   - NWPU:   西北工业大学遥感数据集
#   - UCM:    UC Merced 土地利用数据集（cls_number=21）
#   - RSICD:  遥感图像字幕数据集（cls_number=30）
#   - RSITMD: 遥感图像文本匹配数据集（cls_number=32）
#
# 特点：使用 CLIP 损失（loss_names={"clip":1}），即纯对比学习
#      不使用 ITM/MLM，直接用 CLIP 双塔结构进行图文匹配
# ============================================================================

@ex.named_config
def task_finetune_irtr_sydney_randaug():
    """微调图文检索：Sydney 遥感数据集"""
    exp_name = "finetune_irtr_sydney_randaug"
    datasets = ["sydney"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"clip": 1})          # 纯 CLIP 对比损失
    batch_size = 256
    max_epoch = 50                                  # 遥感数据量小，训练更多 epoch
    max_steps = None
    warmup_steps = 0.3                              # 30% 步数用于 warmup
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 7                                  # Sydney 数据集有 7 个类别
    prompt_length = 16                              # 可学习 prompt 长度
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/Sydney"
    json = "/home/amax/wyj/dataset/Sydney_captions/karpathy/sydney.json"
    val_check_interval = 0.1


@ex.named_config
def task_finetune_irtr_nwpu_randaug():
    """微调图文检索：NWPU 遥感数据集"""
    exp_name = "finetune_irtr_nwpu_randaug"
    datasets = ["nwpu"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.3
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 7
    prompt_length = 16
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/NWPU"
    json = "/home/amax/wyj/dataset/NWPU-Captions-main/karpathy/nwpu_label.json"


@ex.named_config
def task_finetune_irtr_ucm_randaug():
    """微调图文检索：UCM 遥感数据集（21 类土地利用）"""
    exp_name = "finetune_irtr_ucm_randaug"
    datasets = ["ucm"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 21                                  # UCM 数据集有 21 类
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/UCM"
    json = "/home/amax/wyj/dataset/UCM_captions/karpathy/ucm.json"


@ex.named_config
def task_finetune_irtr_rsicd_randaug():
    """微调图文检索：RSICD 遥感图像字幕数据集"""
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["rsicd"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30                                    # RSICD 数据稍多，30 epoch
    max_steps = None
    clip_zero = True
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 30                                   # RSICD 有 30 类场景
    prompt_length = 16
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSICD"
    json = "/home/amax/wyj/dataset/RSICD_captions/karpathy/rsicd.json"


@ex.named_config
def task_finetune_irtr_rsitmd_randaug():
    """微调图文检索：RSITMD 遥感图像文本匹配数据集"""
    exp_name = "finetune_irtr_rsitmd_randaug"
    datasets = ["rsitmd"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 32                                   # RSITMD 有 32 类
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSITMD"
    json = "/home/amax/wyj/dataset/RSITMD/karpathy/rsitmd.json"
    loss_scale_kl = 1
    loss_scale_un = 1


# ============================================================================
# 命名配置 7：遥感数据集 IRTR — ViT-B/32 骨干
#
# ViT-B/32 与 ViT-L/14 的关键差异：
#   - image_size: [32, 32]   （patch 更大 → 特征图更小）
#   - img_features_dim: 768  （ViT-B 的特征维度）
#   - txt_features_dim: 512  （对应 CLIP ViT-B/32 的文本维度）
#   - clip_model = "ViT-B/32"
# ============================================================================

@ex.named_config
def task_finetune_irtr_ucm_randaug_ViTB32():
    """ViT-B/32 骨干：UCM 遥感数据集"""
    exp_name = "finetune_irtr_ucm_randaug"
    datasets = ["ucm"]
    clip_model = "ViT-B/32"
    train_transform_keys = "ViT-B/32"                # 数据增强与 CLIP 模型绑定
    val_transform_keys = "ViT-B/32"
    image_size = [32, 32]                            # ViT-B/32 的特征图：224/32 ≈ 7，但这里是 CLIP 输出的 grid
    img_features_dim = 768                           # ViT-B 的图像特征维度
    txt_features_dim = 512                           # ViT-B/32 的文本特征维度
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 21
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/UCM"
    json = "/home/amax/wyj/dataset/UCM_captions/karpathy/ucm.json"


@ex.named_config
def task_finetune_irtr_rsicd_randaug_ViTB32():
    """ViT-B/32 骨干：RSICD 遥感数据集"""
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["rsicd"]
    clip_model = "ViT-B/32"
    train_transform_keys = "ViT-B/32"
    val_transform_keys = "ViT-B/32"
    image_size = [32, 32]
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    clip_zero = True
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 30
    prompt_length = 16
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSICD"
    json = "/home/amax/wyj/dataset/RSICD_captions/karpathy/rsicd.json"


@ex.named_config
def task_finetune_irtr_rsitmd_randaug_ViTB32():
    """ViT-B/32 骨干：RSITMD 遥感数据集"""
    exp_name = "finetune_irtr_rsitmd_randaug"
    datasets = ["rsitmd"]
    clip_model = "ViT-B/32"
    train_transform_keys = "ViT-B/32"
    val_transform_keys = "ViT-B/32"
    image_size = [32, 32]
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 32
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSITMD"
    json = "/home/amax/wyj/dataset/RSITMD/karpathy/rsitmd.json"
    loss_scale_kl = 1
    loss_scale_un = 1


# ============================================================================
# 命名配置 8：遥感数据集 IRTR — ViT-B/16 骨干
#
# ViT-B/16 与 ViT-B/32 的差异：
#   - image_size: [16, 16]   （patch 更小 → 特征图分辨率更高）
#   - img_features_dim: 768  （相同，都是 ViT-B）
#   - txt_features_dim: 512  （相同）
#   - clip_model = "ViT-B/16"
# ============================================================================

@ex.named_config
def task_finetune_irtr_ucm_randaug_ViTB16():
    """ViT-B/16 骨干：UCM 遥感数据集"""
    exp_name = "finetune_irtr_ucm_randaug"
    datasets = ["ucm"]
    clip_model = "ViT-B/16"
    train_transform_keys = "ViT-B/16"
    val_transform_keys = "ViT-B/16"
    image_size = [16, 16]                            # ViT-B/16 的特征图：224/16 = 14×14
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 21
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/UCM"
    json = "/home/amax/wyj/dataset/UCM_captions/karpathy/ucm.json"


@ex.named_config
def task_finetune_irtr_rsicd_randaug_ViTB16():
    """ViT-B/16 骨干：RSICD 遥感数据集"""
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["rsicd"]
    clip_model = "ViT-B/16"
    train_transform_keys = "ViT-B/16"
    val_transform_keys = "ViT-B/16"
    image_size = [16, 16]
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    clip_zero = True
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 30
    prompt_length = 16
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSICD"
    json = "/home/amax/wyj/dataset/RSICD_captions/karpathy/rsicd.json"


@ex.named_config
def task_finetune_irtr_rsitmd_randaug_ViTB16():
    """ViT-B/16 骨干：RSITMD 遥感数据集"""
    exp_name = "finetune_irtr_rsitmd_randaug"
    datasets = ["rsitmd"]
    clip_model = "ViT-B/16"
    train_transform_keys = "ViT-B/16"
    val_transform_keys = "ViT-B/16"
    image_size = [16, 16]
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 32
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSITMD"
    json = "/home/amax/wyj/dataset/RSITMD/karpathy/rsitmd.json"
    loss_scale_kl = 1
    loss_scale_un = 1


# ============================================================================
# 命名配置 9：遥感数据集 IRTR — ViT-L/14 骨干
#
# ViT-L/14 与 ViT-B 的差异：
#   - image_size: [14, 14]   （最高分辨率特征图）
#   - img_features_dim: 1024 （ViT-L 更大的特征维度）
#   - txt_features_dim: 768  （对应 CLIP ViT-L/14 的文本维度）
#   - clip_model = "ViT-L/14"
# ============================================================================

@ex.named_config
def task_finetune_irtr_ucm_randaug_ViTL14():
    """ViT-L/14 骨干：UCM 遥感数据集"""
    exp_name = "finetune_irtr_ucm_randaug"
    datasets = ["ucm"]
    train_transform_keys = "ViT-L/14"
    val_transform_keys = "ViT-L/14"
    clip_model = "ViT-L/14"
    image_size = [14, 14]                            # ViT-L/14 的特征图：224/14 = 16，但 CLIP 输出 14×14
    img_features_dim = 1024                          # ViT-L 的图像特征维度
    txt_features_dim = 768                           # ViT-L/14 的文本特征维度
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 21
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/UCM"
    json = "/home/amax/wyj/dataset/UCM_captions/karpathy/ucm.json"


@ex.named_config
def task_finetune_irtr_rsicd_randaug_ViTL14():
    """ViT-L/14 骨干：RSICD 遥感数据集"""
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["rsicd"]
    train_transform_keys = "ViT-L/14"
    val_transform_keys = "ViT-L/14"
    clip_model = "ViT-L/14"
    image_size = [14, 14]
    img_features_dim = 1024
    txt_features_dim = 768
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    clip_zero = True
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 30
    prompt_length = 16
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSICD"
    json = "/home/amax/wyj/dataset/RSICD_captions/karpathy/rsicd.json"


@ex.named_config
def task_finetune_irtr_rsitmd_randaug_ViTL14():
    """ViT-L/14 骨干：RSITMD 遥感数据集"""
    exp_name = "finetune_irtr_rsitmd_randaug"
    datasets = ["rsitmd"]
    train_transform_keys = "ViT-L/14"
    val_transform_keys = "ViT-L/14"
    clip_model = "ViT-L/14"
    image_size = [14, 14]
    img_features_dim = 1024
    txt_features_dim = 768
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 32
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSITMD"
    json = "/home/amax/wyj/dataset/RSITMD/karpathy/rsitmd.json"


# ============================================================================
# 命名配置 10：训练步数预设（step schedules）
# 这些配置与 "env" 和 "task" 正交，在命令行末尾叠加使用
# 例如：python run.py with task_mlm_itm step50k
# ============================================================================

@ex.named_config
def step25k():
    """训练 25,000 步"""
    max_epoch = 100     # epoch 上限很大，确保由 max_steps 控制停止
    max_steps = 25000


@ex.named_config
def step50k():
    """训练 50,000 步"""
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    """训练 100,000 步"""
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    """训练 200,000 步"""
    max_epoch = 200
    max_steps = 200000


# ============================================================================
# 命名配置 11：ViT 架构预设
# ============================================================================

@ex.named_config
def vit32_base():
    """ViT-Base/32 的 Transformer 结构参数"""
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12
