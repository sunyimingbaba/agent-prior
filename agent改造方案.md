# CUP 代码 Agent 改造方案（VSCode AI 执行版）

> 目标：把 CUP 的 CCB 聚类模块（ContextCluster）替换为"agent（MLLM）+ 工具"生成的先验提示，接口与下游 RSP-CLIP / UEM 完全兼容。
> 路线：Tool-Star 离线数据管线（轨迹数据生成与主模型训练解耦），训练循环零改动。

## 总体数据流

```
离线阶段（新脚本，一次跑完存盘）：
  图像 + 查询文本 → MLLM（agent）调工具推理 → 保存最后 token 隐藏态 h_end [B, 3584]
                                    ↓ 存到 agent_cache/ 目录

训练阶段（CUP 代码改造，4 处小改）：
  数据加载器读 h_end → agent_proj (MLP 3584→1024) → visual_prior_prompt
                                    ↓
  后续流程完全复用原代码：prior_img → prior_analysis_img → repeat L 次 → 公式1融合 → CLIP → UEM 损失
```

## 阶段 1：新建 agent 轨迹生成脚本（全新代码，不动 CUP）

### 新目录：`CUP/agent/`

**1.1 `agent/tools.py` —— 工具集（起步 2 个，先简单）**

```python
# 工具1：目标检测（先用伪工具占位，返回结构化文本）
def object_detection(image_path, category):
    """返回 JSON 文本，如 {"detections": [{"label": "ship", "bbox": [...], "conf": 0.9}]}"""

# 工具2：场景分类（可用冻结 CLIP 现成实现）
def scene_classify(image_path):
    """返回 top-3 场景类别文本"""
```

工具执行结果以**文本形式**回填 MLLM 上下文（`<result>...</result>` 标签包裹）。

**1.2 `agent/rollout.py` —— 轨迹生成主脚本**

```python
# 核心逻辑：
# 1. 加载 MLLM（Qwen2.5-VL-7B，transformers 加载）
# 2. 对 RSICD/UCM 训练集每张图：
#    prompt = 系统指令 + <image> + "查询文本: {caption}"
#    生成轨迹：模型输出 <think>...<tool_call>...，执行工具，<result> 回填，循环到 <answer>
# 3. 保存 h_end = 模型最后一步输出的 hidden_states[-1] 的最后一个 token，shape [1, 3584]
# 4. 存盘：agent_cache/{dataset}/{image_id}.pt
```

**第一版简化**：先不加工具，纯推理（图+查询直接过 MLLM 取 h_end），验证管线跑通后再加工具调用循环。

**1.3 查询文本来源**：RSICD/UCM 每张图 5 个 caption，取第一个 caption 作为 agent 的查询输入。
（注：原 CCB 只看图，agent 版看"图+查询"——先验对查询敏感，这是相对原 CUP 的天然增益点。）

## 阶段 2：CUP 代码改造（4 处改动，全部在现有文件）

### 改动 1：`vilt/modules/vilt_module.py` 的 `__init__`（第 343 行 `self.cluster_model = cluster_model(...)` 附近）

```python
# 新增：agent 先验投影 + 开关
self.use_agent_prior = config.get("use_agent_prior", False)  # 默认 False = 原版行为
agent_hidden_dim = config.get("agent_hidden_dim", 3584)      # Qwen2.5-VL-7B 隐藏维
self.agent_proj = nn.Sequential(
    nn.Linear(agent_hidden_dim, visual_prompt_dim),          # 3584 → 1024(ViT-L) 或 768(ViT-B)
    nn.LayerNorm(visual_prompt_dim),
)
# 原 cluster_model 保留不动（做对照实验用）
```

### 改动 2：训练路径（第 414 行 `visual_prior_prompt = self.cluster_model(img)`）

```python
if self.use_agent_prior:
    h_end = batch["agent_hidden"].float()          # [B, 3584]，数据加载器提供
    visual_prior_prompt = self.agent_proj(h_end)   # [B, visual_prompt_dim]
else:
    visual_prior_prompt = self.cluster_model(img)  # 原 CCB 路径，保持不变
# ↓ 以下 418-446 行全部不动：prior_img、prior_analysis_img、repeat、公式1融合
```

### 改动 3：测试/推理路径（第 558 行，同样的 if/else 替换）

### 改动 4：数据集读取 h_end 缓存

文件：`vilt/datasets/rsicd_caption_karpathy_dataset.py`（UCM/Sydney 版本同样改法）

在 `__getitem__` 中按图像 id 读缓存：

```python
# 从 agent_cache/{dataset}/{image_id}.pt 读取 h_end
agent_hidden = torch.load(cache_path) if use_agent_prior else torch.zeros(3584)
batch["agent_hidden"] = agent_hidden
```

缓存路径写死或从 config 读（建议 config 加 `agent_cache_dir`）。

### 配套：config 加两个键

- `use_agent_prior`: bool
- `agent_cache_dir`: str
- `agent_hidden_dim`: int（默认 3584）

## 阶段 3：验证步骤（按顺序执行，每步过了再走下一步）

1. **前向跑通**：`use_agent_prior=True`，UCM 小数据集跑 1 个 batch，确认无 shape 报错
2. **训练 1 epoch**：确认 loss 下降（USCE + CMR + KLD 三个损失都要能算，KLD 的蒸馏目标现在变成 agent_proj 的输出，梯度自动回传，无需改损失代码）
3. **对照实验**：`use_agent_prior=False`（原 CCB）vs `True`（agent 先验），比 R@1/R@5/R@10
4. **加工具**：把 rollout.py 从纯推理升级为工具调用循环，重新生成 h_end 缓存，再比一次

## 给 VSCode AI 的直接提示词

> 请按照 /Users/botlv/CUP-code/agent改造方案.md 执行 CUP 代码的 agent 改造：
> 1. 新建 CUP/agent/ 目录，实现 tools.py（2 个工具）和 rollout.py（Qwen2.5-VL-7B 生成 h_end 缓存）
> 2. 修改 vilt/modules/vilt_module.py：__init__ 加 agent_proj 和 use_agent_prior 开关；第 414 行和第 558 行的 cluster_model 调用改为 if/else 分支
> 3. 修改数据集加载器（rsicd/ucm/sydney 三个 karpathy dataset）支持读取 agent_hidden 缓存
> 4. 保持默认 use_agent_prior=False，确保原版代码行为完全不变
> 每改完一个文件告诉我改了什么，不要动方案以外的代码。

## 注意事项

- **MLLM 冻结**：第一版 agent_proj 可训练、MLLM 冻结（h_end 是预生成的缓存，MLLM 不参与训练图）。后期要做端到端再上 LoRA。
- **显存**：Qwen2.5-VL-7B 推理需 ~16GB 显存，服务器 A100 没问题；本地 Mac 跑不动就只做代码改造 + 服务器跑数据。
- **KLD 损失天然兼容**：它取 visual_prior_prompt 的第一个 token 做蒸馏，agent 分支下蒸馏目标自动变为 agent 先验，不需要改损失代码。
- 这个改造练的是通用管线（工具调用 + h_end 投影 + 离线数据），以后细粒度分类任务的代码骨架完全一样（把检索损失换成分类头即可）。
