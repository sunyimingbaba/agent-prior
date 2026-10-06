# CUP 阶段 B 方案：教师轨迹 + LoRA SFT（VSCode AI 执行版）

> 目标：用教师模型示范"工具协议"的标准轨迹，LoRA SFT 教素人 Qwen2.5-VL-7B 把 `tool_call → result 注入 → 继续分析` 当作正常对话模式。训练完用新 MLLM 重新 rollout 生成 h_end 缓存，重训翻译层，对照纯推理版。
> 背景（阶段 A 结论）：agent 纯推理先验全面优于 CCB（+6.4~+28 点）；工具版在素人档位全负（-1.2~-4.9 点），三个归因实验（过滤/换口径/全注入）全部否定局部归因，唯一统一解释：**素人模型没学过工具协议，任何 `<result>` 注入都是异常输入，污染隐藏态**。SFT 教协议后，工具的信息增益才有机会释放。

## 总体数据流

```
① 教师轨迹生成（离线，一次）：
   抽样图像 + 查询 → 教师模型输出"调工具意图" → 真实执行 scene_classify
   → 组装轨迹：<tool_call>scene_classify</tool_call>\n<result>真实结果</result>\n<answer>整合分析</answer>
                              ↓ 存为 SFT 训练数据（JSON/arrow）

② LoRA SFT（训 MLLM）：
   轨迹数据 → 冻结主干 + LoRA 适配器 → 标准 LM 损失（next token）→ 学会工具协议

③ 循环刷新（复用阶段 A 管线）：
   LoRA 版 MLLM 跑 rollout.py（工具循环）→ 新 h_end 缓存 → 重训翻译层 → 测试集三方对照
```

## 第一部分：教师轨迹数据生成

### 1.1 教师模型选择（二选一，按服务器条件）

- **首选：Qwen2.5-VL-72B-Instruct**（服务器本地跑，约需 80GB 显存，无 API 成本，数据隐私好）
- 备选：Gemini API（flash 版成本低，但需要 key 和网络；ToolFG 原文用的就是 Gemini）

### 1.2 轨迹格式（关键设计：与 rollout.py 推理格式逐字符一致）

rollout.py 推理时：模型输出 `new_text` → 我们拼 `<result>{真实结果}</result>\n` 继续生成。所以 SFT 训练数据的 assistant 部分必须是一整段连续文本：

```
assistant 文本 = "<tool_call>scene_classify</tool_call>\n<result>harbor (0.61), port (0.24), ship (0.08)</result>\n<answer>图中是港口场景，有多艘船停靠，与查询"有船的港口"相符。</answer>"
```

即：**tool_call 标签 + 真实工具结果 + answer 三段拼成一个 assistant 回复**。训练时 loss 对整段算。这样训练格式 = 推理格式，学生学到的协议就是推理时发生的协议。

### 1.3 result 必须真实执行工具生成

教师模型只负责输出"该调工具 + 最终分析"，**`<result>` 里的内容不是教师编的，是我们真实调用 `scene_classify(image)` 得到的结果**回填进去。保证学生学到的是"读真实工具输出"而不是"读教师编的输出"。

具体做法（对每张抽样图）：

1. 给教师模型：图 + 查询 + 提示"你要调用 scene_classify 工具，我会把结果给你，请基于结果给出最终分析"。让教师输出两部分：`<tool_call>scene_classify</tool_call>` 和 `<answer>分析</answer>`
2. 本地真实执行 `scene_classify(image)` 得到结果文本
3. 组装：`<tool_call>scene_classify</tool_call>\n<result>{结果}</result>\n<answer>{教师分析}</answer>`
4. 存为训练样本：messages = [system(工具版 SYSTEM_PROMPT), user(图+查询), assistant(上述整段)]

### 1.4 数据规模与混合

- 规模：**每数据集 2000~3000 条起步**（UCM 全量 2100；RSICD/RSITMD 各抽 2500）。协议学习样本效率高，不需要全量；先小后大
- 混合：建议 **80% 调工具轨迹 + 20% 纯回答轨迹**（教师直接 `<answer>`，不调工具）。防止模型学会"凡事都调工具"——纯推理版的成绩证明"不调工具也能好"，模型应学会按需选择

### 1.5 目录约定

```
CUP/agent/sft/
  make_trajectories.py    # 教师轨迹生成脚本（新文件）
  train_sft.py            # LoRA SFT 训练脚本（新文件）
  data/                   # 轨迹数据（JSONL）
```

## 第二部分：LoRA SFT 训练

### 2.1 训练配置（推荐起步值）

| 项 | 值 | 说明 |
|---|---|---|
| 基础模型 | Qwen2.5-VL-7B-Instruct | 与 rollout 一致 |
| LoRA r / alpha | 32 / 64 | 协议学习，r=16 也够；32 保险 |
| target_modules | 全部线性层（或 q_proj,k_proj,v_proj,o_proj + mlp 层） | peft 默认全线性即可 |
| 学习率 | 1e-4 | LoRA 常用区间 1e-4~2e-4 |
| epochs | 1~2 | 先 1 epoch 看效果，协议学习不需要多轮 |
| batch size / 梯度累积 | 按显存调，目标有效 batch 32~64 | |
| 精度 | bf16 | A100 原生 |
| 损失 | 标准 next-token LM loss，**只对 assistant 部分算**（user/system 部分 label 设 -100） | |
| 显存 | 约 25~30GB | A100-40G 可跑，80G 宽松 |

### 2.2 训练脚本要点

- 用 peft + transformers Trainer（或 trl SFTTrainer），不引入新框架依赖
- 图像输入：Qwen2.5-VL processor 处理，与 rollout.py 相同的 chat template
- 训练数据 messages 用 1.3 的格式；`<answer>` 前文全部参与 loss 计算
- 训练完 `save_pretrained` 存 LoRA 适配器（约几十 MB），**不合并主干**

### 2.3 rollout.py 小改（加载 LoRA）

新增 `--lora_path` 参数：非空时用 `peft.PeftModel.from_pretrained(base_model, lora_path)` 加载 LoRA 版 MLLM 跑工具循环 rollout。其余逻辑（工具循环、缓存协议、h_end 口径）**完全不动**。

## 第三部分：训练侧联动（零改动）

CUP 训练侧**一个文件都不动**。循环刷新流程：

```
LoRA 版 MLLM → rollout.py（--lora_path，工具循环）→ agent_cache_sft/ 新缓存
→ config 改 agent_cache_dir 指向新缓存 → 重训翻译层（阶段 A 那套）→ 测试集评估
```

对照三方：原版 CCB（已有）/ 纯推理 agent（已有）/ SFT 工具 agent（新）。

## 验证步骤（按顺序）

1. **轨迹质量抽检**：make_trajectories 先跑 50 条，人工/肉眼检查 5 条：tool_call 格式正确、result 是真实工具输出、answer 确实整合了 result 信息（不是无视结果自说自话）
2. **SFT 冒烟**：100 条轨迹、1 epoch、小 lr，确认 loss 下降、无 OOM、能正常 save LoRA
3. **全量 SFT**：每数据集 2000~3000 条，训练完检查点保存
4. **行为检查**（SFT 前后对比，工具循环版 rollout --limit 200）：调用率、乱文率（输出无意义文本的比例）——预期调用率维持高位、乱文率显著下降
5. **全量 rollout + 重训翻译层**：三数据集，新缓存目录 `agent_cache_sft`
6. **对照评估**：SFT 工具版 vs 纯推理版。**预期：工具增益转正（≥0，理想 +1~5 点）；若仍为负，说明单工具信息量不足以补偿协议成本，考虑加工具（真检测器）或调 SFT 数据配比**

## 给 VSCode AI 的直接提示词

> 请按照 /Users/botlv/CUP-code/agent阶段B方案.md 执行阶段 B：
> 1. 新建 CUP/agent/sft/make_trajectories.py：教师模型（Qwen2.5-VL-72B，路径可配置）生成轨迹；教师只输出 tool_call 意图和 answer，<result> 由本地真实执行 scene_classify 回填；80% 调工具 + 20% 纯回答混合；输出 JSONL 到 CUP/agent/sft/data/
> 2. 新建 CUP/agent/sft/train_sft.py：peft LoRA（r=32, alpha=64, lr=1e-4, bf16, 1~2 epochs），只对 assistant 部分算 loss，训练数据格式与 rollout.py 推理格式逐字符一致（tool_call + result + answer 拼成一个 assistant 文本）
> 3. 修改 rollout.py：加 --lora_path 参数支持加载 LoRA 版 MLLM，其余不动
> 4. CUP 训练侧任何文件都不要动
> 每改完一个文件告诉我改了什么，不要动方案以外的代码。

## 注意事项

- **格式一致性是生命线**：SFT 数据的 assistant 文本格式必须与 rollout.py 推理时"生成文本 + result 回填"的拼接方式逐字符一致，否则训练和推理存在分布偏移
- **教师轨迹里的 result 必须真实执行**：不能用教师模型自己编工具输出
- **对照口径不变**：h_end 仍取最后 generate 的最后 token，与阶段 A 全部实验同口径
- **先小后大**：轨迹 50 条抽检 → SFT 100 条冒烟 → 全量。每一步都别跳
- **行为检查数据要存档**：SFT 前后调用率/乱文率对比是论文"为什么 SFT 有效"的直接证据
