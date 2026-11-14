#!/bin/bash
# 方案A：域内工具版主方法（few-shot GRPO + 域内微调场景分类工具）
# 对比目标：57.97（CLIP 零样本工具版）
PY=/opt/conda/envs/PyTorch-2.4.1/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/cup-debug/CUP
LOG=/tmp/pipeline_domain_tool.log
echo "[$(date '+%F %T')] domain-tool pipeline started" > $LOG

MODEL=/workspace/cup-debug/models/Qwen2.5-VL-7B-Instruct
ARROW=/workspace/cup-debug/data/ucm/ucm_caption_karpathy_label_val.arrow
LAST_CKPT=/workspace/cup-debug/logs/finetune_irtr_ucm_randaug_seed0_from_/version_0/checkpoints/last.ckpt
DATA_ROOT=/workspace/cup-debug/data/ucm
DOMAIN_CLS=/workspace/cup-debug/models/ucm_scene_cls.pt

# 1. GRPO 训练（few-shot 版，与 road1c 完全一致；训练阶段不执行工具）
$PY agent/grpo/train_grpo.py \
    --base_model $MODEL --rl_arrow $ARROW --scorer_ckpt $LAST_CKPT \
    --val_arrow $ARROW --val_cache /workspace/cup-debug/grpo_val_cache.pt \
    --steps 500 --group_size 8 --sample_batch 4 --lr 5e-6 --seed 0 \
    --reward-mode soft --lam 1.0 \
    --out_dir /workspace/cup-debug/models/lora_domain_tool \
    --stats_log /workspace/cup-debug/domain_tool_stats.jsonl \
    > /tmp/grpo_domain_tool.log 2>&1
[ -d /workspace/cup-debug/models/lora_domain_tool ] || { echo "RUN grpo FAILED" >> $LOG; exit 1; }
echo "[$(date '+%F %T')] domain-tool grpo done" >> $LOG

# 2. rollout：域内分类器工具
$PY agent/rollout.py --data_root $DATA_ROOT --dataset ucm \
    --model $MODEL --lora_path /workspace/cup-debug/models/lora_domain_tool \
    --cache_dir /workspace/cup-debug/agent_cache_domain_tool \
    --tool_conf_threshold 0 \
    --domain_cls $DOMAIN_CLS >> $LOG 2>&1
echo "[$(date '+%F %T')] domain-tool rollout done -> retrain" >> $LOG

# 3. 评测重训
$PY run_frozen.py with task_finetune_irtr_ucm_randaug_ViTL14 \
    data_root=$DATA_ROOT log_dir=/workspace/cup-debug/logs \
    per_gpu_batchsize=32 batch_size=256 num_gpus=1 seed=0 \
    save_image_path=/workspace/cup-debug/images \
    use_agent_prior=True agent_cache_dir=/workspace/cup-debug/agent_cache_domain_tool \
    > /tmp/train_domain_tool.log 2>&1
echo "[$(date '+%F %T')] domain-tool eval done" >> $LOG
echo "[$(date '+%F %T')] DOMAIN-TOOL ALL DONE" >> $LOG
