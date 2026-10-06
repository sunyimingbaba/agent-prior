# CUP 工具循环版 rollout 改造方案（VSCode AI 执行版）

> 目标：把 rollout.py 从"纯推理"升级为"工具调用循环"——MLLM 先看图和查询，可选调用 `scene_classify` 工具，工具结果以文本回填上下文，最终取整个循环最后一个 token 的隐藏态 h_end。训练侧**零改动**，只换缓存目录。
> 背景：纯推理版已在 UCM/RSICD/RSITMD 三数据集全面验证（agent 先验优于原版 CCB，增益 8.6~28 点）。本版验证"加了工具是否有增量"。

## 总体数据流（对比纯推理版）

```
纯推理版（现状）：
  图 + 查询 → MLLM 一次生成 → 直接取最后 token 的 h_end

工具版（本方案）：
  图 + 查询 → MLLM 生成
                │ 输出含 <tool_call>scene_classify</tool_call>？
                ├─ 是 → 真实执行工具 → <result>harbor (0.61), ...</result> 回填 → 继续生成（下一轮）
                └─ 否（输出 <answer> 或没按格式）→ 循环结束
  循环结束后（最多 max_rounds 轮）→ 取最后一次 generate 最后 token 的 h_end
```

## 改动清单（只动 CUP/agent/ 两个文件，训练侧零改动）

### 改动 1：`CUP/agent/tools.py` —— scene_classify 支持预加载 CLIP

**动机**：rollout 要跑上万张图，每张图调用一次工具；如果每次调用都重新 `clip.load()`（约 2GB 模型），会慢到不可用。改成 rollout 里全局加载一次、传入复用。

签名改为：

```python
def scene_classify(image, clip_model_name="ViT-L/14", topk=3, model=None, preprocess=None):
    """场景分类工具。model/preprocess 为 None 时自行加载（保持独立可用）；
    传入预加载对象时复用（rollout 批量调用场景）。返回文本不变。"""
    if model is None or preprocess is None:
        from prompt_clip import clip
        model, preprocess = clip.load(clip_model_name)
        model = model.eval().cuda()
    # 以下逻辑不变：文本特征、图像特征、softmax、top-k 拼接返回
```

其余逻辑（RS_SCENES、余弦相似度、返回格式 `"harbor (0.61), port (0.24), ship (0.08)"`）**完全不动**。

### 改动 2：`CUP/agent/rollout.py` —— 工具调用循环

**2.1 SYSTEM_PROMPT 改为工具版**：

```python
SYSTEM_PROMPT = (
    "你是遥感图像分析助手。给定一张遥感图像和一条查询文本，分析图像中与查询相关的内容。"
    "你可以调用工具：scene_classify（对图像做场景分类，返回 top-3 场景类别及概率）。"
    "调用格式：<tool_call>scene_classify</tool_call>，工具结果会以 <result>...</result> 返回给你。"
    "分析完成后以 <answer>...</answer> 输出最终答案。"
)
```

**2.2 新增解析函数**：

```python
import re

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>")

def parse_tool_call(text):
    """从生成文本中提取第一个工具调用。返回工具名或 None。"""
    m = TOOL_CALL_RE.search(text)
    if m:
        name = m.group(1).strip()
        return name if name == "scene_classify" else None  # 只认已注册的工具
    return None
```

**2.3 `rollout_one` 重写为多轮循环**（替换原函数）：

```python
@torch.no_grad()
def rollout_one(model, processor, image, caption, tools, max_rounds, max_new_tokens, device):
    """
    工具循环版 rollout：生成 → 解析 tool_call → 执行工具 → <result> 回填 → 再生成。
    返回 (h_end, tool_used)：tool_used 为本张图实际成功调用工具的次数（统计用）。
    h_end 口径与纯推理版一致：取最后一次 generate 的最后一步、最后一层、最后 token。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"查询文本: {caption}"},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    tool_used = 0
    outputs = None
    for _ in range(max_rounds):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        new_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
        new_text = processor.decode(new_ids, skip_special_tokens=True)

        tool_name = parse_tool_call(new_text)
        if tool_name is None:
            break                      # 没调工具（<answer> 或格式失败）→ 循环结束

        result_text = tools[tool_name](image)           # 真实执行工具
        tool_used += 1

        # 回填：本轮输出 + <result> 拼进 input_ids，图像输入（pixel_values）不动，继续下一轮
        feedback = new_text + f"\n<result>{result_text}</result>\n"
        feedback_ids = processor(text=[feedback], return_tensors="pt").input_ids.to(device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], feedback_ids], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    h_end = outputs.hidden_states[-1][-1][:, -1, :].squeeze(0).cpu()
    return h_end, tool_used
```

**关键点（必须遵守）**：
- `outputs` 若为 None（max_rounds=0 的边界，正常不会发生）直接报错即可，不要静默。
- 第二轮起的 `model.generate(**inputs)`，`inputs` 里**仍保留第一轮 processor 产出的 `pixel_values` 等图像字段**，只拼接 `input_ids` / `attention_mask`——图像 token 本来就在 input_ids 里，模型每轮都能"看到"图。
- h_end 取 `outputs.hidden_states[-1][-1][:, -1, :]`：与纯推理版口径一致（消融对比公平，唯一变量是"是否用工具"）。

**2.4 main() 修改**：

- 新增参数：
  - `--max_rounds`（int，默认 3）：工具循环最大轮数
  - `--no_tools`（flag）：回退纯推理模式（供对照调试，默认关闭）
  - `--cache_dir` 默认值改为 `./agent_cache_tool`（**不要覆盖纯推理版缓存**）
- `--no_tools` 时走纯推理路径（行为与现状完全一致）；否则加载一次 CLIP 并构造工具表：

```python
if not args.no_tools:
    from prompt_clip import clip
    clip_model, preprocess = clip.load("ViT-L/14")
    clip_model = clip_model.eval().cuda()
    tools = {"scene_classify": lambda img: scene_classify(img, model=clip_model, preprocess=preprocess)}
else:
    tools = {}
```

- 主循环里改为 `h_end, tool_used = rollout_one(model, processor, image, caption, tools, args.max_rounds, args.max_new_tokens, args.device)`，累计 `tool_used`。
- 结束时打印统计报告（新增）：

```python
print(f"工具调用统计：{total_tool_calls} 次调用 / {total_generated} 张图（调用率 {rate:.1%}）")
```

- 调试支持：`--limit` 小时（如 5）逐张打印轨迹文本（模型输出 + tool_call + result），确认循环行为正常。用 `--verbose` flag 控制，默认关闭。

### 不改的部分（明确边界）

- **训练侧零改动**：vilt_module.py / base_dataset.py / datamodule / objectives 全部不动，只靠换 `agent_cache_dir` 指向新缓存。
- **object_detection 不上场**：空壳返回 `{"detections": []}` 无信息量，本版工具集只有 scene_classify 一个（SYSTEM_PROMPT 也只介绍这一个）。
- **缓存协议不变**：`{cache_dir}/{table_name}/{img_index}.pt`，内容仍为 [3584] 张量。

## 验证步骤（按顺序，每步过了再走下一步）

1. **smoke test**：`--dataset ucm --limit 5 --verbose`，确认：每张图能跑完循环、tool_call 解析正确、`<result>` 正确回填、h_end shape 仍为 [3584]、统计报告正常打印。
2. **调用率观察**：`--limit 50` 看调用率。若调用率极低（<20%），先检查 SYSTEM_PROMPT 和解析正则，不要直接全量跑。
3. **全量生成**：三个数据集（建议顺序 RSICD → RSITMD → UCM），`--cache_dir ./agent_cache_tool`，断点续跑自动生效。
4. **训练对照**：config 的 `agent_cache_dir` 指向 `agent_cache_tool`，`use_agent_prior=True` 重训，对比纯推理版 mR（UCM 59.49 / RSICD 45.69 / RSITMD 68.17）。

## 给 VSCode AI 的直接提示词

> 请按照 /Users/botlv/CUP-code/agent工具循环改造方案.md 执行 CUP 的 rollout 工具循环升级：
> 1. 修改 CUP/agent/tools.py：scene_classify 增加 model/preprocess 可选参数（None 时自行加载，保持向后兼容），其余逻辑不动
> 2. 修改 CUP/agent/rollout.py：SYSTEM_PROMPT 换工具版；新增 parse_tool_call；rollout_one 重写为"生成→解析→执行工具→result 回填→再生成"的多轮循环（h_end 口径与纯推理版一致）；main 加 --max_rounds/--no_tools/--verbose 参数、cache_dir 默认改 agent_cache_tool、CLIP 预加载一次、结束打印工具调用统计
> 3. 训练侧任何文件都不要动
> 每改完一个文件告诉我改了什么，不要动方案以外的代码。

## 注意事项

- **消融公平性**：h_end 提取口径（最后 generate 的最后 token）与纯推理版完全一致，对比时唯一变量是"是否用工具"。
- **调用失败不是错误**：素人 Qwen2.5-VL 没经过 SFT，不一定按格式调工具。解析不到 tool_call 就正常结束循环、照常存 h_end（相当于这张图退化为纯推理）。调用率数据本身就是阶段 B（SFT 教工具调用）的动机证据，跑完如实记录。
- **显存**：MLLM（约 16GB）+ CLIP ViT-L/14（约 2GB）同卡没问题；A100 单卡够。
- **预期**：工具版增益大概率在 +0.5~3 点，细粒度数据集（RSICD/RSITMD）空间更大；即使增益小，消融条目 + 调用率统计也是论文实验表的有效增量。
