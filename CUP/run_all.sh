#!/bin/bash
# =============================================================================
# CUP (Context and Uncertainty-aware Prompt) 训练脚本
# 论文: Cross-Modal Remote Sensing Image-Text Retrieval via CUP (TNNLS 2025)
# 作用: 在3个遥感数据集上，用3种CLIP骨架，测试5种prompt长度，全面评估CUP模型
# =============================================================================

# ---------------------------------------------------------------------------
# 环境初始化
# ---------------------------------------------------------------------------

# 指定使用的GPU编号（单卡训练，0号卡）
CARD=0

# 初始化conda（使其在脚本中可用），然后切换到指定环境
source /home/amax/anaconda3/etc/profile.d/conda.sh
conda deactivate

echo ""
echo ""
echo "****************** Activating conda environment  zzzz ******************"
# 激活名为 "zzzz" 的conda环境（包含PyTorch、CLIP等依赖）
conda activate zzzz

# ---------------------------------------------------------------------------
# 1. 测试阶段（加载已有checkpoint评估，不训练）
# ---------------------------------------------------------------------------

echo "" 
echo "****************** Testing ******************"
echo "****************** Testing ******************"
echo ""
echo ""

# test_only=True: 只做推理评估，不训练
# load_path=<ckpt_path>: 填入预训练权重路径
# prompt_length=8: 使用8个prompt token进行评估
CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=8 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1 \
    test_only=True load_path=<ckpt_path>


# =============================================================================
# 2. ViT-B/32 骨架训练
#    patch_size=32, 图像特征维度=768, 文本特征维度=512
#    batch_size=160（因为ViT-B/32显存占用最小，可以用更大的batch）
# =============================================================================

batch_size=160

# 切换到代码工作目录
cd /home/amax/wyj/CUP_Test

echo ""
echo "****************** Training******************"
echo "****************** Training ******************"
echo ""
echo ""

# ---------------------------------------------------------------------------
# 2.1 数据集: RSITMD（4743张遥感图像，32类，每图5句描述）
#     关键参数:
#       weight_margin=0.015  —— 论文Table II中RSITMD的最佳margin
#       loss_scale=1         —— 总损失中的clip_loss权重
#       linear_projector=0.1 —— 论文中的 δ (margin), MLP残差的动态上界
#       warmup_steps=0.1     —— 前10%的训练步数做lr warmup
#       learning_rate=5e-4   —— AdamW初始学习率
#     这里对 prompt_length=1,2,4,8,16 各训练一轮（论文Fig.5a分析最优长度）
# ---------------------------------------------------------------------------

echo ""
echo "****************** RSITMD_captions learning_rate=5e-4******************"
echo ""

# prompt_length=1: 视觉/文本各1个可学习的prompt token
CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=1 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

# prompt_length=2
CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=2 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

# prompt_length=4
CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=4 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

# prompt_length=8 —— 论文推荐的最优长度
CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=8 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

# prompt_length=16 —— 过长，性能略有下降
CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=16 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1


# ---------------------------------------------------------------------------
# 2.2 数据集: RSICD（10921张遥感图像，30类，每图5句描述）
#     文本相似度比RSITMD高，检索难度更大
#     max_epoch=30（论文中RSICD的设置）
# ---------------------------------------------------------------------------

echo ""
echo "****************** RSICD_captions learning_rate=5e-4******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=1 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=2 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=4 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=8 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=16 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1


# ---------------------------------------------------------------------------
# 2.3 数据集: UCM-Captions（2100张遥感图像，21类，每图5句描述）
#     数据量最小，文本相似度最高，难度最大
#     weight_margin=0.1 —— 比RSITMD/RSICD的0.015大，因为UCM更难
#     precision=32 —— 使用float32精度训练（其他数据集默认float16）
#     max_epoch=50（论文中UCM的设置）
# ---------------------------------------------------------------------------

echo ""
echo "****************** Prompt_length 16 learning_rate=5e-4 ******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=1 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=2 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=4 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=8 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTB32 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=16 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1


# =============================================================================
# 3. ViT-B/16 骨架训练
#    patch_size=16（比B/32更细粒度），图像特征维度=768，文本特征维度=512
#    batch_size=96（patch更小 → 序列更长 → 显存占用更大 → batch缩小）
# =============================================================================

batch_size=96

echo ""
echo "****************** RSITMD_captions learning_rate=5e-4******************"
echo ""

# --- 3.1 RSITMD, ViT-B/16 ---

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=1 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=2 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=4 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=8 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=16 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1


# --- 3.2 RSICD, ViT-B/16 ---

echo ""
echo "****************** RSICD_captions learning_rate=5e-4******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=1 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=2 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=4 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=8 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=16 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1


# --- 3.3 UCM-Captions, ViT-B/16 ---

echo ""
echo "****************** UCM ******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=1 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=2 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=4 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=8 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTB16 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=16 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1


# =============================================================================
# 4. ViT-L/14 骨架训练
#    patch_size=14（最大最细粒度），图像特征维度=1024，文本特征维度=768
#    batch_size=36（模型最大，显存占用最高，batch最小）
#    ViT-L/14 是论文中性能最强的骨架（Table III-V中的最佳结果）
# =============================================================================

batch_size=36

echo ""
echo "****************** RSITMD_captions learning_rate=5e-4******************"
echo ""

# --- 4.1 RSITMD, ViT-L/14 ---

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=1 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=2 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=4 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=8 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 \
    prompt_length=16 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1


# --- 4.2 RSICD, ViT-L/14 ---

echo ""
echo "****************** RSICD_captions learning_rate=5e-4******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=1 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=2 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=4 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=8 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSICD_captions \
    task_finetune_irtr_rsicd_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_nodes=1 \
    prompt_length=16 weight_margin=0.015 warmup_steps=0.1 \
    learning_rate=5e-4 loss_scale=1 linear_projector=0.1


# --- 4.3 UCM-Captions, ViT-L/14 ---

echo ""
echo "****************** UCM ******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=1 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=2 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=4 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=8 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/UCM_captions \
    task_finetune_irtr_ucm_randaug_ViTL14 \
    per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 \
    prompt_length=16 weight_margin=0.1 learning_rate=5e-4 \
    loss_scale=1 linear_projector=0.1
