# CUP 论文代码：逐行全流程解读

> 适用读者：研0，有基础 Python 知识，了解一点点 PyTorch，但不了解多模态/CLIP/Prompt Learning。
> 阅读方式：建议边看本文档边在 IDE 中打开对应文件，对照着读。

---

## 目录

1. [基础知识速览](#1-基础知识速览)
2. [训练启动：从命令行到 main 函数](#2-训练启动从命令行到-main-函数)
3. [数据加载：图片和文本怎么变成模型输入](#3-数据加载图片和文本怎么变成模型输入)
4. [模型初始化：ViLTransformerSS 的每个模块](#4-模型初始化viltransformerss-的每个模块)
5. [一次训练 Step 的完整前向传播](#5-一次训练-step-的完整前向传播)
6. [损失计算：Monte Carlo 不确定性建模](#6-损失计算monte-carlo-不确定性建模)
7. [反向传播和参数更新](#7-反向传播和参数更新)
8. [验证和评估](#8-验证和评估)
9. [完整调用链速查表](#9-完整调用链速查表)

---

## 1. 基础知识速览

### 1.1 多模态检索在干什么

**任务**：给你一张遥感图片（比如机场的卫星图），在数据库里找到描述这张图片的文本（比如 "an airport with several planes on the runway"）。反过来也一样，给一句话，找出对应的图片。

**核心难题**：图片是像素（数字矩阵），文字是单词（离散符号），它们不在同一个"空间"里，没法直接比较。需要一个模型把图片和文字都映射到**同一个向量空间**，让匹配的图文对向量距离近，不匹配的距离远。

### 1.2 CLIP 是什么

CLIP 是 OpenAI 在 2021 年发布的模型，在海量图文对上训练。它有两个"编码器"：

```
图片 → 图像编码器（ViT/ResNet）→ 图像向量（如 512 维浮点数数组）
文字 → 文本编码器（Transformer）→ 文本向量（如 512 维浮点数数组）
```

训练好的 CLIP，猫的图片向量和 "a photo of a cat" 的文字向量余弦相似度很高。

**CUP 论文的出发点**：CLIP 是在自然图像（猫、狗、汽车）上训练的，直接用在遥感图像上效果不好。怎么办？

### 1.3 Prompt Learning 是什么

传统微调（Fine-tuning）：修改 CLIP 内部所有参数 → 参数量大，容易过拟合。

Prompt Learning：**不改 CLIP 的内部参数，而是在输入前面拼接一些可学习的向量**。这些向量叫 "prompt"。

**类比**：CLIP 是一个只会英文的翻译。你不想让他重新学中文，而是教他几个"中文暗号"。他听到暗号就自动调整翻译策略。这个"暗号"就是 prompt。

CUP 的创新：同时给**图像端**和**文本端**都加 prompt：

```
原始文本输入：   "an airport with planes" → token 序列 [SOT, an, airport, ..., EOT]
CUP 文本输入：  [SOT, p1, p2, ..., p8, an, airport, ..., EOT]  ← 前面拼了 8 个可学习 token

原始图像输入：   [CLS, patch1, patch2, ..., patch49]
CUP 图像输入：  [CLS, v1, v2, ..., v8, patch1, ..., patch49]  ← 前面拼了 8 个可学习向量
```

### 1.4 对比学习是什么

一个 batch 有 B 对（图片，文本）。对于第 i 张图片：
- 第 i 句文本是 **正样本**（正确的配对）
- 其他 B-1 句文本是 **负样本**（错误的配对）

损失函数的目标：让正样本对的相似度尽可能大，负样本对的相似度尽可能小。

```
相似度矩阵（B=4 为例）：
         text_0  text_1  text_2  text_3
img_0   [ 0.9     0.1     0.2     0.05 ]  ← 希望 img_0 和 text_0 的 0.9 最大
img_1   [ 0.15    0.85    0.1     0.2  ]  ← 希望 img_1 和 text_1 的 0.85 最大
img_2   [ 0.1     0.2     0.88    0.15 ]  ← ...
img_3   [ 0.05    0.1     0.15    0.92 ]  ← 对角线应该是最大值
```

### 1.5 PyTorch Lightning 是什么

PyTorch Lightning 是对 PyTorch 的封装。它帮你处理了训练循环、GPU 分配、日志记录等繁琐的事，你只需要：
1. 定义模型（`LightningModule`）
2. 定义数据（`LightningDataModule`）
3. 调用 `trainer.fit(model, data)`

代码中的 `ViLTransformerSS` 继承自 `pl.LightningModule`，所以它自带 `training_step`、`validation_step`、`configure_optimizers` 等方法。

### 1.6 Sacred 是什么

Sacred 是一个配置管理库。代码中的 `@ex.config`、`@ex.named_config`、`@ex.automain` 都是它的装饰器。

```bash
# 命令行中有 "with" 关键字，这是 Sacred 的语法
python run_frozen.py with task_finetune_irtr_rsitmd_randaug_ViTB32 prompt_length=8
```

`task_finetune_irtr_rsitmd_randaug_ViTB32` 对应 [config.py](CUP/vilt/config.py) 中第 429-455 行定义的命名配置。`prompt_length=8` 覆盖了默认值 16。

---

## 2. 训练启动：从命令行到 main 函数

### 2.1 命令行

```bash
# 文件：run_all.sh 第 40-41 行
CUDA_VISIBLE_DEVICES=0 python run_frozen.py with \
    data_root=/data/cross-dataset/dataset/RSITMD \
    task_finetune_irtr_rsitmd_randaug_ViTB32 \
    per_gpu_batchsize=160 num_gpus=1 num_nodes=1 prompt_length=8
```

**逐词解释**：
| 部分 | 含义 |
|------|------|
| `CUDA_VISIBLE_DEVICES=0` | 只让程序看到编号为 0 的 GPU |
| `python run_frozen.py` | 用 Python 执行训练入口脚本 |
| `with` | Sacred 库的关键字，表示后面的参数注入配置 |
| `data_root=...` | 数据存放的根目录 |
| `task_finetune_irtr_rsitmd_randaug_ViTB32` | 引用 config.py 第 429 行的预设配置组 |
| `per_gpu_batchsize=160` | 每张 GPU 每次处理 160 对图文 |
| `prompt_length=8` | 用 8 个可学习 prompt token |

### 2.2 配置加载：config.py 是怎么工作的

**文件**：[config.py](CUP/vilt/config.py)

```python
# 第 1 行
from sacred import Experiment
ex = Experiment("ViLT")      # 创建一个名为 "ViLT" 的实验
```

```python
# 第 20-102 行
@ex.config
def config():
    # 这是"默认配置"，所有值都可以被命令行和命名配置覆盖
    exp_name = "vilt"
    seed = 0
    loss_names = _loss_names({"itm": 1, "mlm": 1})  # 默认训练 itm+mlm 任务
    batch_size = 4096
    clip_model = "ViT-L/14"
    prompt_length = 16              # 默认 16 个 prompt
    max_epoch = 100
    learning_rate = 1e-4
    # ... 还有很多参数
```

```python
# 第 429-455 行
@ex.named_config
def task_finetune_irtr_rsitmd_randaug_ViTB32():
    # 这个命名配置覆盖了默认配置中的很多值
    clip_model = "ViT-B/32"                       # ← 覆盖：换成 ViT-B/32
    train_transform_keys = "ViT-B/32"             # ← 覆盖
    image_size = [32, 32]                         # ← 覆盖：ViT patch 大小
    img_features_dim = 768                        # ← 覆盖：图像特征维度
    txt_features_dim = 512                        # ← 覆盖：文本特征维度
    loss_names = _loss_names({"clip": 1})         # ← 覆盖：只算 clip loss
    max_epoch = 30                                # ← 覆盖：最多 30 轮
    cls_number = 32                               # RSITMD 的类别数
    prompt_length = 16                            # prompt 长度
    # ...
```

**Sacred 的合并逻辑**：
1. 先加载 `@ex.config` 的默认配置
2. 再加载 `@ex.named_config` 的命名配置（覆盖同名的值）
3. 最后用命令行 `with` 后面的参数覆盖（优先级最高）

最终生效的配置 = 默认 + 命名配置 + 命令行参数。比如 `prompt_length`：
- 默认 = 16
- `task_finetune_irtr_rsitmd_randaug_ViTB32` = 16
- 命令行 `prompt_length=8` = **8（最终生效）**

### 2.3 run_frozen.py：训练入口

**文件**：[run_frozen.py](CUP/run_frozen.py)

```python
# 第 41-119 行
@ex.automain                      # Sacred 的入口装饰器
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])    # 设置随机种子，保证可复现

    # ====== 步骤1：创建数据模块 ======
    dm = MTDataModule(_config, dist=False)
    # MTDataModule 根据 _config["datasets"]（值为 ["rsitmd"]）
    # 自动选择合适的 DataModule 子类（RSITMDCaptionKarpathyDataModule）

    # ====== 步骤2：创建模型 ======
    model = ViLTransformerSS(_config)
    # 这是整个项目的核心类！构造函数在 vilt_module.py 第 214 行

    # ====== 步骤3：设置 checkpoint 回调 ======
    exp_name = f'{_config["exp_name"]}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,                           # 只保留最好的 1 个模型
        monitor="val/the_metric",               # 监控验证集上的 the_metric
        mode="max",                             # 越大越好
        save_last=True,                         # 也保存最后一个 epoch 的模型
    )

    # ====== 步骤4：设置日志 ======
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{...}',
    )

    # ====== 步骤5：设置回调 ======
    callbacks = [MyBackboneFinetuning(), checkpoint_callback, lr_callback]
    # MyBackboneFinetuning 是自定义回调（第 14 行）
    # 它在训练开始前冻结 CLIP 的所有参数，保证只有 prompt/adapter 等模块被训练

    # ====== 步骤6：创建 Trainer 并开始训练 ======
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],               # GPU 数量
        num_nodes=_config["num_nodes"],         # 节点数量
        accelerator="ddp",                      # 分布式数据并行
        max_epochs=_config["max_epoch"],        # 最大训练轮数
        callbacks=callbacks,                    # 回调列表
        accumulate_grad_batches=grad_steps,     # 梯度累积步数
        val_check_interval=...,                 # 多久验证一次
        # ...
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)       # ← 开始训练！
        trainer.test(model, test_dataloaders=dm)  # 训练完测试
    else:
        trainer.test(model, datamodule=dm)      # 纯测试模式
```

### 2.4 MyBackboneFinetuning：冻结 CLIP 的机制

**文件**：[run_frozen.py](CUP/run_frozen.py) 第 14-37 行

```python
class MyBackboneFinetuning(BaseFinetuning):
    def freeze_before_training(self, pl_module):
        # 在训练开始前被 Lightning 自动调用
        self.freeze_clip(pl_module.model)
        # pl_module.model 就是 clip.load() 返回的 CLIP 模型
        # 这会把 CLIP 所有参数的 requires_grad 设为 False

    def freeze_module(self, module):
        for param in module.parameters(recurse=False):
            param.requires_grad = False      # ← 冻结！这些参数不更新
```

**什么是冻结？**
PyTorch 中每个参数都有一个 `requires_grad` 属性。设为 `False` 后，反向传播时不会计算这个参数的梯度，优化器也不会更新它。

CUP 中**被冻结**的参数（不更新）：
- CLIP 的图像编码器（ViT/ResNet）
- CLIP 的文本编码器（Transformer）
- CLIP 的 token embedding 和位置编码

CUP 中**可训练**的参数（会更新）：
- `prompt_embeddings`：文本端 prompt
- `prompt_proj`：文本 prompt 投影层
- `visual_prompt_embeddings`：视觉端 prompt
- `visual_prompt_proj`：视觉 prompt 投影层
- `visual_prompt_proj_gather`：视觉 prompt 融合后的投影层
- `cluster_model`：ContextCluster（视觉先验提取器）
- `adapter_img` / `adapter_txt`：图像/文本 adapter
- `adapter_img_mapping` / `adapter_txt_mapping`：adapter 门控标量
- `analysis_img` / `analysis_txt`：不确定性分析模块
- `prior_img` / `prior_analysis_img`：先验相关模块

**参数量对比**：CLIP ViT-B/32 约 150M 参数，CUP 新增的可训练参数只有约 5M。这就是 Prompt Learning 的优势——训练负担极轻。

---

## 3. 数据加载：图片和文本怎么变成模型输入

### 3.1 数据存储格式

数据以 Apache Arrow 格式存储。每个数据集有几个 `.arrow` 文件：

```
/data/cross-dataset/dataset/RSITMD/
├── rsitmd_caption_karpathy_label_train.arrow   ← 训练集（图片+多句描述）
├── rsitmd_caption_karpathy_label_val.arrow     ← 验证集
└── rsitmd_caption_karpathy_label_test.arrow    ← 测试集
```

Arrow 是列式存储，高效读写。每个文件是一个表格，列包括：
- `image`：JPEG 格式的图片二进制数据
- `caption`：文本描述列表 `["text1", "text2", "text3", "text4", "text5"]`
- `path`：原始图片的文件路径

### 3.2 DataModule 初始化

**文件**：[rsitmd_caption_karpathy_datamodule.py](CUP/vilt/datamodules/rsitmd_caption_karpathy_datamodule.py)

```python
class RSITMDCaptionKarpathyDataModule(BaseDataModule):
    @property
    def dataset_cls(self):
        return RSITMDCaptionKarpathyDataset    # 告诉 BaseDataModule 用这个 Dataset
```

`BaseDataModule`（继承自 `pl.LightningDataModule`）会自动调用 `train_dataloader()`、`val_dataloader()`、`test_dataloader()` 创建对应的 DataLoader。

### 3.3 Dataset 初始化：建立索引

**文件**：[base_dataset.py](CUP/vilt/datasets/base_dataset.py) `BaseDataset.__init__()` 第 12-91 行

#### 步骤 3.3.1：读取 arrow 文件

```python
# 第 49-61 行
tables = [
    pa.ipc.RecordBatchFileReader(
        pa.memory_map(f"{data_dir}/{name}.arrow", "r")
    ).read_all()
    for name in names   # 训练时 names = ["rsitmd_caption_karpathy_label_train",
                        #                    "rsitmd_caption_karpathy_label_val"]
]
self.table = pa.concat_tables(tables, promote=True)
# 把训练集和验证集的 arrow 合并成一个大表
```

#### 步骤 3.3.2：提取所有文本并去重

```python
# 第 62-69 行
self.all_texts = self.table[text_column_name].to_pandas().tolist()
# all_texts[0] = ["an airport with several planes", "aerial view of runway", ...] （第0张图的5句描述）
# all_texts[1] = ["a bridge over the river", ...]
# ...

self.all_texts = [list(set(texts)) for texts in self.all_texts]
# set() 去重：同一张图的重复描述只保留一份
```

#### 步骤 3.3.3：建立索引映射

这是整个数据加载中最关键的概念。

**问题**：RSITMD 有约 5000 张图，每张图有 5 句描述。训练时需要把所有 25000 个（图, 描述）对都遍历到。

**方案**：建立一个映射表 `index_mapper`，把 "数据集索引"（0~24999）映射到 "(图片编号, 描述编号)"。

```python
# 第 81-87 行
j = 0
for i, texts in enumerate(self.all_texts):     # i = 图片编号，texts = 第i张图的描述列表
    for _j in range(len(texts)):               # _j = 描述在列表中的位置
        self.index_mapper[j] = (i, _j)         # j = 数据集中的唯一编号
        j += 1

# 具体例子（假设每张图有 5 句描述）：
# index_mapper = {
#     0:  (0, 0),   # 数据集样本0 = 第0张图的第0句描述
#     1:  (0, 1),   # 数据集样本1 = 第0张图的第1句描述
#     ...
#     4:  (0, 4),   # 数据集样本4 = 第0张图的第4句描述
#     5:  (1, 0),   # 数据集样本5 = 第1张图的第0句描述
#     ...
# }
```

所以 `len(dataset)` = `len(index_mapper)` = 所有图片的描述总数（而非图片数）。

#### 步骤 3.3.4：创建 CLIP 预处理函数

```python
# 第 93-94 行
model_name = str(self.transforms)    # "ViT-B/32"
_, self.preprocess = clip.load(model_name, device='cpu')
# clip.load 返回 (model, preprocess)
# 这里只需要 preprocess，它是一个 torchvision 变换管道：
#   Resize(224) → CenterCrop(224) → 转Tensor → Normalize(特殊均值和方差)
```

### 3.4 取一个样本：get_suite 逐行详解

**文件**：[base_dataset.py](CUP/vilt/datasets/base_dataset.py) `get_suite()` 第 174-189 行

```python
def get_suite(self, index):          # index 是 PyTorch DataLoader 给的，范围 0~24999
    result = None
    while result is None:            # 循环直到成功（处理偶尔的读取错误）
        ret = dict()

        ret.update(self.get_image(index))   # ← 获取图片
        if not self.image_only:
            txt = self.get_text(index)      # ← 获取文本
            ret.update(txt)

        result = True
    return ret
```

#### 3.4.1 get_image 详解

```python
# 第 120-134 行
def get_image(self, index, image_key="image"):
    # 步骤A：读取原始图片
    image = self.get_raw_image(index, image_key=image_key)
    # → 从 arrow 文件读取 JPEG 二进制 → PIL.Image.open() → 转 RGB

    # 步骤B：CLIP 预处理
    clip_img = self.preprocess(image)
    # → Resize(224) → CenterCrop(224) → ToTensor → Normalize
    # 输出 shape: [3, 224, 224]，值范围约 [-2, 2]（归一化后）

    # 步骤C：获取元信息
    _index, caption_index = self.index_mapper[index]
    # _index = 图片编号，caption_index = 描述编号

    image_path = self.table["path"][_index]   # 图片路径（用于调试）

    return {
        "clip_img": clip_img,                # Tensor [3, 224, 224]
        "img_index": _index,                 # 图片编号（同一张图的不同描述共享）
        "cap_index": caption_index,          # 描述编号
        "raw_index": index,                  # 数据集中的原始索引
        "path": image_path                   # 文件路径
    }
```

#### 3.4.2 get_text 详解

```python
# 第 142-152 行
def get_text(self, raw_index):
    # 步骤A：找到对应的文本
    index, caption_index = self.index_mapper[raw_index]
    text = self.all_texts[index][caption_index]
    # text = "an airport with several planes on the runway"

    # 步骤B：CLIP tokenization
    clip_text = clip.tokenize(text, context_length=77-self.prompt_length)
    # context_length = 77 - 8 = 69（给 prompt 预留了 8 个位置！）

    return {
        "clip_text": (text, clip_text),      # (原始字符串, token Tensor [1, 69])
        "img_index": index,
        "cap_index": caption_index,
        "raw_index": raw_index,
    }
```

**tokenize 做了什么**（[clip.py](CUP/prompt_clip/clip.py) 第 197-237 行）：

对于输入 `"an airport with several planes"`：

1. 转为小写并分词：`["an", "airport", "with", "several", "planes"]`
2. 每个词映射为数字 ID（通过 BPE 词典）：`[550, 1234, 591, 2533, 3541]`
3. 加上起始符和结束符：`[49406, 550, 1234, 591, 2533, 3541, 49407]`
4. 填充到固定长度 69：`[49406, 550, 1234, ..., 49407, 0, 0, 0, ...]`

最终返回的 `clip_text` 是一个 `[1, 69]` 的 LongTensor，大多数位置是 0（padding）。

**为什么 context_length = 77 - prompt_length？** 因为 prompt 会拼接在文本前面，总共占用 77 个位置：
```
[SOT] [p1] [p2] ... [p8] [an] [airport] ... [planes] [EOT] [PAD] ... [PAD]
 ← 1 → ←───── 8 ──────→ ←────────────── 69-2=67 ──────────────→
                            实际文本占 67，但只用了 5+2=7 个
总计 = 1 + 8 + 69 = 78？不对，再看：context_length=77-self.prompt_length=69
                                                                     = 77-8=69
序列 = [SOT(token0)] [p1..p8(8个)] [an..EOT(最多67)] = 1 + 8 + 67 = 76 ... 
不对，1+8+68=77。EOT 在最末尾。text.argmax(dim=-1) 会返回 EOT 的位置（最大值 49407）。

理解：tokenizer 产生的序列最长 69，其中：
- 第一个是 SOT
- 最后一个是 EOT（如果文本不够长，EOT 就是最后一个非0位置）
- 中间是最多 67 个实际 token
加上 prompt 的 8 个，总共 69+8=77，刚好等于 CLIP 的上下文长度。
```

### 3.5 Batch 拼接：collate 函数

**文件**：[base_dataset.py](CUP/vilt/datasets/base_dataset.py) `collate()` 第 192-218 行

DataLoader 攒够 160 个样本后，调用 `collate()` 把它们拼成一个 batch。

```python
def collate(self, batch, mlm_collator):
    # batch 是一个列表，长度 160，每个元素是 get_suite 返回的字典

    batch_size = len(batch)          # 160
    keys = set([key for b in batch for key in b.keys()])
    # keys = {"clip_img", "img_index", "cap_index", "raw_index",
    #         "clip_text", "replica", "path"}

    # 把相同 key 的值收集到一起
    dict_batch = {k: [dic[k] for dic in batch] for k in keys}

    # 处理图片：把 160 个 [3,224,224] tensor 堆成一个 [160,3,224,224] tensor
    img_keys = ["clip_img"]
    for img_key in img_keys:
        images = dict_batch[img_key]          # 160 个 tensor 的列表
        img = [img_i.cpu().numpy() for img_i in images]
        img = numpy.array(img)               # [160, 3, 224, 224]
        img = torch.tensor(img)
        dict_batch[img_key] = img            # Tensor [160, 3, 224, 224]

    # 处理文本：拆分原始字符串和 token tensor
    txt_keys = ["clip_text"]
    for text_key in txt_keys:
        txt = dict_batch[text_key]           # 160 个 (字符串, tensor[1,69]) 的列表
        text = [_txt for (_txt, _clip_txt) in txt]          # 字符串列表
        clip_txt = torch.cat([_clip_txt for (_, _clip_txt) in txt], dim=0)
        # dim=0 拼接 → [160, 69]

        dict_batch[f"{text_key}_txt"] = text            # 原始文本列表
        dict_batch[f"{text_key}_token"] = clip_txt      # Tensor [160, 69]

    return dict_batch
```

**最终一个 batch 的结构**：
```python
{
    "clip_img":          Tensor [160, 3, 224, 224],   # 160 张预处理后的图片
    "clip_text_txt":     List[str], 长度 160,           # 原始文本
    "clip_text_token":   Tensor [160, 69],              # token 化的文本
    "img_index":         List[int], 长度 160,           # 每张图片的编号
    # ... 其他元信息
}
```

### 3.6 数据到达模型前的最后一步

在 `ViLTransformerSS.forward()` 被调用前，Lightning 的 DataLoader 已经把 batch 字典转换为了元组：

```python
# 在 objectives.py compute_clip 函数中的用法：
img = batch["clip_img"]               # Tensor [160, 3, 224, 224]
txt = batch["clip_text_token"]        # Tensor [160, 69]
infer = pl_module.infer((img, txt,))  # 传入元组
```

---

## 4. 模型初始化：ViLTransformerSS 的每个模块

**文件**：[vilt_module.py](CUP/vilt/modules/vilt_module.py) `ViLTransformerSS.__init__()` 第 214-191 行

### 4.1 加载 CLIP

```python
# 第 244-246 行
self.name = config["clip_model"]                # "ViT-B/32"
self.model, self.preprocess = clip.load(self.name)
self.model = self.model.float()
```

**clip.load("ViT-B/32") 做了什么**（[clip.py](CUP/prompt_clip/clip.py) 第 94-144 行）：

```python
def load(name, device, ...):
    # 步骤1：确定模型路径
    if name in _MODELS:                                          # "ViT-B/32" 在字典里
        model_path = _download(_MODELS[name], ...)               # 从 OpenAI 服务器下载
        # _MODELS["ViT-B/32"] = "https://openaipublic.azureedge.net/.../ViT-B-32.pt"
        # 下载到 ~/.cache/prompt_clip/ViT-B-32.pt

    # 步骤2：加载权重
    state_dict = torch.load(model_path, map_location="cpu")      # 加载到 CPU

    # 步骤3：构建模型
    model = build_model(state_dict)                              # → prompt_clip/model.py
    # build_model 根据权重文件的结构自动推断模型参数
    # （视觉层数、文本层数、embedding维度等）
    # 然后创建 CLIP 类实例，加载权重

    # 步骤4：移到 GPU
    model = model.to(device)

    # 步骤5：返回
    return model, _transform(model.visual.input_resolution)       # (模型, 预处理函数)
```

**CLIP 模型内部结构**（[prompt_clip/model.py](CUP/prompt_clip/model.py) `CLIP.__init__()` 第 278-332 行）：

```python
class CLIP(nn.Module):
    def __init__(self, ...):
        # 图像编码器：ViT-B/32
        self.visual = VisionTransformer(
            input_resolution=224,     # 输入图片大小
            patch_size=32,            # 每个 patch 是 32×32 像素
            width=768,                # 特征维度
            layers=12,                # Transformer 层数
            heads=12,                 # 注意力头数
            output_dim=512            # 最终输出维度
        )

        # 文本编码器：12 层 Transformer + 因果注意力掩码
        self.transformer = Transformer(
            width=512,                # 特征维度
            layers=12,                # Transformer 层数
            heads=8,                  # 注意力头数
            attn_mask=causal_mask     # 因果掩码（每个词只能看到它前面的词）
        )

        # 文本 token 嵌入表
        self.token_embedding = nn.Embedding(49408, 512)
        # 49408 是词表大小，512 是嵌入维度

        # 文本位置编码
        self.positional_embedding = nn.Parameter([77, 512])
        # 77 个位置，每个位置 512 维

        # 最终的 LayerNorm
        self.ln_final = LayerNorm(512)

        # 文本投影矩阵（把文本特征映射到共享空间）
        self.text_projection = nn.Parameter([512, 512])

        # 温度参数（控制相似度的缩放）
        self.logit_scale = nn.Parameter(ones([]) * log(1/0.07))
        # log(1/0.07) ≈ 2.66，所以初始值 exp(2.66) ≈ 14.3
        # 这个参数把内积值缩放到合适范围，再送进 softmax
```

### 4.2 文本 Prompt 模块

```python
# 第 255-263 行
prompt_dim = config["txt_features_dim"]           # 512（对于 ViT-B/32）

# prompt_proj：投影层
self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)   # 512 → 512
nn.init.kaiming_normal_(self.prompt_proj.weight, ...)  # He 初始化

# prompt_embeddings：可学习的 prompt token 嵌入
val = math.sqrt(6. / float(1 + prompt_dim))           # Xavier 初始化的范围因子
self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_length, prompt_dim))
# 形状：[1, 8, 512]（prompt_length=8 时）
# 第 0 维是 1（batch 维度，通过 expand 扩展到实际 batch size）
# 第 1 维是 8（prompt token 数量）
# 第 2 维是 512（特征维度，与 CLIP 的文本嵌入维度一致）

nn.init.uniform_(self.prompt_embeddings.data, -val, val)   # 均匀分布初始化
```

**初始化值长什么样？** Xavier 均匀初始化使得方差在每层保持稳定。对于 512 维：
```
val = sqrt(6 / (1 + 512)) = sqrt(6/513) ≈ 0.108
```
所以初始值在 `[-0.108, 0.108]` 之间随机均匀分布。每个 prompt token 就是一个 512 维的小随机向量。

### 4.3 视觉 Prompt 模块

```python
# 第 273-282 行
visual_patch_size = config['image_size']          # [32, 32]
visual_prompt_dim = config['img_features_dim']    # 768（对于 ViT-B/32）

# 投影层
self.visual_prompt_proj = nn.Linear(visual_prompt_dim, visual_prompt_dim)
nn.init.kaiming_normal_(...)

# 可学习 prompt
visual_val = math.sqrt(6. / float(3 * reduce(mul, visual_patch_size, 1) + visual_prompt_dim))
# = sqrt(6 / (3 * 32 * 32 + 768)) = sqrt(6 / 3840) ≈ 0.040
self.visual_prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_length, visual_prompt_dim))
# 形状：[1, 8, 768]
nn.init.uniform_(..., -visual_val, visual_val)   # 初始值在 [-0.04, 0.04]

# 融合后的投影层（与前面结构相同，但独立参数）
self.visual_prompt_gather_dropout = Dropout(0.1)
self.visual_prompt_proj_gather = nn.Linear(visual_prompt_dim, visual_prompt_dim)
```

### 4.4 ContextCluster：视觉先验提取器

```python
# 第 287 行
self.cluster_model = cluster_model(num_classes=visual_prompt_dim)
```

**文件**：[context_cluster.py](CUP/model/context_cluster.py) `cluster_model()` 第 1082-1105 行

```python
def cluster_model(pretrained=False, num_classes=768, **kwargs):
    layers = [1, 1]           # 两个 stage，每个有 1 个 ClusterBlock
    embed_dims = [128, 256]   # Stage1: 128维, Stage2: 256维
    mlp_ratios = [8, 8]       # MLP 扩展比例
    heads = [4, 4]            # 注意力头数
    head_dim = [32, 32]       # 每个头的维度

    # Patch Embedding：用 4×4 卷积，步长 4，把 224×224 变成 56×56
    # 输入通道 5（RGB 3通道 + 坐标 2通道），输出 128 维

    model = ContextCluster(
        layers, embed_dims=embed_dims, num_classes=num_classes,
        # ... 更多参数
    )
    return model
```

**ContextCluster 做了什么**（`forward()` 第 1069-1080 行）：

```
输入图片 [B, 3, 224, 224]
  │
  ├→ forward_embeddings(): 拼接坐标信息，Patch Embedding
  │    [B, 3, 224, 224] + 坐标 [B, 2, 224, 224] = [B, 5, 224, 224]
  │    → Conv2d(5→128, 4×4, stride=4) → [B, 128, 56, 56]
  │
  ├→ forward_tokens(): 经过 ClusterBlock×1 → 降采样 → ClusterBlock×1
  │    Stage1: [B, 128, 56, 56] → ClusterBlock → [B, 128, 56, 56]
  │    Downsample: → PointReducer(stride=3) → [B, 256, 19, 19]
  │    Stage2: [B, 256, 19, 19] → ClusterBlock → [B, 256, 19, 19]
  │
  ├→ norm + Global Average Pooling([-2, -1])
  │    [B, 256, 19, 19] → mean → [B, 256]
  │
  └→ self.head: Linear(256 → 768)
       [B, 256] → [B, 768]
```

**Cluster 核心操作**（`Cluster.forward()` 第 724-778 行）：

1. **投影到相似度空间**：`self.f(x)` → `[B, heads×head_dim, H, W]`
2. **投影到值空间**：`self.v(x)` → 同上
3. **分割成局部区域**（fold）：把大特征图切成小块减少计算量
4. **生成聚类中心**：`AdaptiveAvgPool2d(proposal_w, proposal_h)` → 每个区域一个中心
5. **计算相似度**：每个像素与聚类中心的余弦相似度（经 sigmoid 归一化）
6. **硬分配**：每个像素只分配给最相似的中心
7. **聚合**：加权求和，得到每个聚类中心的输出特征
8. **分发**：把聚类中心的特征分发回各个像素

**直觉理解**：ContextCluster 是一种轻量级的"注意力"机制。它自动把图片分成几个区域，每个区域自动找到一个"代表性中心"，然后基于这个中心聚合信息。最终输出是 `[B, 768]`，可以理解为"这张遥感图的视觉摘要"。

### 4.5 先验映射与不确定性模块

```python
# 第 290-298 行

# 视觉先验分析：为后续的 KL 散度提供"方差"
self.prior_analysis_img = nn.Sequential(
    nn.Linear(prompt_dim, prompt_dim),     # 512 → 512
)
# 输入：[B, 512]（来自 prior_img 的输出）
# 输出：[B, 512]（经过 clamp(≥0)，每个维度是一个非负数，代表不确定性）

# 先验映射：把 ContextCluster 的输出映射到 prompt 空间
self.prior_img = nn.Linear(visual_prompt_dim, prompt_dim)
# 输入：[B, 768]（来自 cluster_model）
# 输出：[B, 512]（映射到与文本 prompt 相同的维度）

# 门控标量：控制先验和随机 prompt 的融合比例
self.cluster_mapping = nn.Linear(visual_prompt_dim, 1)
# 输入：[B, 8, 768]（visual_prior_prompt expand 后）
# 输出：[B, 8, 1]（每个 prompt token 一个 0~1 的比例）
```

### 4.6 Adapter 模块

```python
# 第 300-312 行

# 图像 adapter
self.adapter_img = nn.Sequential(
    nn.Linear(prompt_dim, prompt_dim),     # 512 → 512
    nn.GELU(),                             # 非线性激活
    nn.Linear(prompt_dim, prompt_dim),     # 512 → 512
)
self.adapter_img_mapping = nn.Sequential(
    nn.Linear(prompt_dim, 1),              # 512 → 1（输出一个标量）
)

# 文本 adapter（结构完全相同，独立参数）
self.adapter_txt = nn.Sequential(
    nn.Linear(prompt_dim, prompt_dim),
    nn.GELU(),
    nn.Linear(prompt_dim, prompt_dim),
)
self.adapter_txt_mapping = nn.Sequential(
    nn.Linear(prompt_dim, 1),
)
```

Adapter 的结构设计：
- 两层 MLP：`d → d → GELU → d → d`（瓶颈在中间，但实际上维度没变）
- `_mapping` 层：输出一个标量，经过 `clamp(0, 0.1)` 后作为"修正比例"

**为什么最大只到 0.1？** 这是 Adapter 设计的核心：不能让 adapter 主导输出，只允许它微调（最多 10% 的贡献）。这是为了防止 adapter 过拟合遥感数据，同时保留 CLIP 的泛化能力。

### 4.7 不确定性分析模块

```python
# 第 283-289 行

self.analysis_img = nn.Sequential(
    nn.Linear(prompt_dim, prompt_dim),     # 512 → 512
)
self.analysis_txt = nn.Sequential(
    nn.Linear(prompt_dim, prompt_dim),     # 512 → 512
)
```

这两个线性层的作用是：接收 adapter 修正后的特征，输出"每个维度的不确定性"。输出经过 `clamp(min=0)`，确保是非负数。

### 4.8 权重加载（仅测试模式）

```python
# 第 317-320 行
if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
    ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
    state_dict = ckpt["state_dict"]              # Lightning checkpoint
    self.load_state_dict(state_dict, strict=False)  # strict=False 允许部分加载
```

---

## 5. 一次训练 Step 的完整前向传播

### 5.1 入口：training_step

**文件**：[vilt_module.py](CUP/vilt/modules/vilt_module.py) 第 338-343 行

```python
def training_step(self, batch, batch_idx):
    vilt_utils.set_task(self)       # ① 设置当前任务
    output = self(batch)            # ② 调用 self.forward(batch)
    total_loss = sum([v for k, v in output.items() if "loss" in k])
    return total_loss               # Lightning 拿到 total_loss 后自动反向传播
```

#### 步骤 ①：set_task

**文件**：[vilt_utils.py](CUP/vilt/modules/vilt_utils.py) 第 296-299 行

```python
def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    # loss_names = {"clip": 1} → current_tasks = ["clip"]
```

#### 步骤 ②：self.forward(batch)

```python
# 第 303-336 行
def forward(self, batch):
    ret = dict()
    if len(self.current_tasks) == 0:
        # 推理模式（评估时用）
        ret.update(self.infer(batch))
        return ret

    if "clip" in self.current_tasks:            # ← 训练走这里
        ret.update(objectives.compute_clip(self, batch))

    # 其他可能的任务（本配置中不激活，因为 loss_names 只有 clip:1）
    if "mlm" in self.current_tasks: ...

    return ret
```

`objectives.compute_clip(self, batch)` 做了两件事：
1. 调用 `self.infer(batch)` ← 核心推理流程
2. 用推理结果计算三个损失

### 5.2 infer() 逐行详解

**这是整个代码中最重要的函数。** 输入是一个 batch，输出是所有中间结果。

**文件**：[vilt_module.py](CUP/vilt/modules/vilt_module.py) `infer()` 第 193-249 行

```python
def infer(self, batch):
    (img, txt) = batch
    # img: Tensor [160, 3, 224, 224]  ← 160 张预处理后的遥感图
    # txt: Tensor [160, 69]           ← 160 条 token 化的文本描述

    B = txt.shape[0]                   # 160（batch size）
```

#### 步骤 A（第 200-209 行）：生成视觉 Prompt

```python
    # A1：提取视觉先验
    visual_prior_prompt = self.cluster_model(img)
    # img [160, 3, 224, 224]
    # → ContextCluster.forward()
    #   → forward_embeddings: [160, 3+2, 224, 224] → [160, 128, 56, 56]
    #   → Stage1 ClusterBlock: [160, 128, 56, 56] → [160, 128, 56, 56]
    #   → Downsample: [160, 128, 56, 56] → [160, 256, 19, 19]
    #   → Stage2 ClusterBlock: [160, 256, 19, 19] → [160, 256, 19, 19]
    #   → norm + GAP + head: [160, 256] → [160, 768]
    # visual_prior_prompt: [160, 768]
```

```python
    # A2：映射到 prompt 空间
    visual_prior_prompt_out = self.prior_img(visual_prior_prompt)
    # self.prior_img = Linear(768 → 512)
    # [160, 768] → [160, 512]
    # 这个值将在 infer 的返回值中用于 KL 散度计算
```

```python
    # A3：计算先验的不确定性（"方差"）
    visual_prior_prompt_prior = torch.clamp(
        self.prior_analysis_img(visual_prior_prompt_out), min=0
    )
    # self.prior_analysis_img = Linear(512 → 512)
    # [160, 512] → [160, 512]（经过 clamp，所有值 ≥ 0）
    # 这个值越大，说明模型对对应维度的先验越不确定
```

```python
    # A4：扩展到 prompt 长度
    visual_prior_prompt = visual_prior_prompt.unsqueeze(1).repeat(1, self.prompt_length, 1)
    # [160, 768] → unsqueeze(dim=1) → [160, 1, 768]
    #             → repeat(1, 8, 1)  → [160, 8, 768]
    # 把同一个先验向量复制 8 份（每个 prompt token 一份）
```

```python
    # A5：计算融合比例
    visual_ratio = torch.clip(
        self.cluster_mapping(visual_prior_prompt), min=0, max=1
    )
    # self.cluster_mapping = Linear(768 → 1)
    # [160, 8, 768] → [160, 8, 1]
    # 每个 prompt token 位置都有一个 0~1 的比例值
    # ratio 越小 → 越信任先验
    # ratio 越大 → 越信任随机初始化的 prompt
```

```python
    # A6：生成随机视觉 prompt
    visual_random_prompt = self.visual_prompt_dropout(
        self.visual_prompt_proj(self.visual_prompt_embeddings).expand(B, -1, -1)
    )
    # self.visual_prompt_embeddings:       [1, 8, 768]
    # → expand(160, 8, 768):              [160, 8, 768]
    # → visual_prompt_proj (Linear 768→768): [160, 8, 768]
    # → visual_prompt_dropout (Dropout 0.1): [160, 8, 768]（10%的值随机变 0）
```

```python
    # A7：融合先验和随机 prompt
    visual_prompt = (1 - visual_ratio) * visual_prior_prompt + visual_ratio * visual_random_prompt
    # visual_ratio 趋近 0 → visual_prompt ≈ visual_prior_prompt（完全依靠先验）
    # visual_ratio 趋近 1 → visual_prompt ≈ visual_random_prompt（完全依靠随机prompt）
    # 形状：[160, 8, 768]
```

```python
    # A8：融合后的投影和 dropout
    visual_prompt = self.visual_prompt_gather_dropout(
        self.visual_prompt_proj_gather(visual_prompt)
    )
    # visual_prompt_proj_gather: Linear(768 → 768) → [160, 8, 768]
    # visual_prompt_gather_dropout: Dropout(0.1)
    # 最终视觉 prompt：[160, 8, 768]
```

**为什么视觉端的 prompt 比文本端复杂这么多？**

文本 prompt 是任务无关的——8 个 token 对所有图片都一样。但视觉 prompt 需要**感知图片内容**。ContextCluster 分析图片后说"这张图是机场，跑道区域很重要"，视觉 prompt 就会偏向编码"机场跑道"的信息。

`visual_ratio` 是一个自适应门控：如果 ContextCluster 很确定（比如图片确实是清晰的机场），ratio 就小，多依靠先验；如果不确定（图片模糊或场景复杂），ratio 就大，多用随机 prompt 的通用能力。

#### 步骤 B（第 211 行）：生成文本 Prompt

```python
    text_prompt = self.prompt_dropout(
        self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
    )
    # self.prompt_embeddings:          [1, 8, 512]
    # → expand(160, 8, 512):          [160, 8, 512]
    # → prompt_proj (Linear 512→512): [160, 8, 512]
    # → prompt_dropout (Dropout 0.1): [160, 8, 512]
```

文本端简单：不依赖图片内容，对所有 batch 样本都是一样的 prompt（经过相同的投影）。

#### 步骤 C（第 213-214 行）：CLIP 前向传播

```python
    logits_per_image, logits_per_text, image_features, text_features = \
        self.model(img, txt, visual_prompt, text_prompt)
```

这会调用 **`prompt_clip/model.py` 的 `CLIP.forward()`**（第 410-424 行）。下面逐个展开：

##### C1：图像编码 → `encode_image(image, visual_prompt)`（第 375-376 行）

```python
def encode_image(self, image, visual_prompt):
    return self.visual(image.type(self.dtype), visual_prompt)
    # → VisionTransformer.forward()
```

**`VisionTransformer.forward()`** 逐行解读（第 257-275 行）：

```python
def forward(self, x, visual_prompt):
    # x: [160, 3, 224, 224]
    # visual_prompt: [160, 8, 768]

    # 第1步：Patch 化
    x = self.conv1(x)
    # Conv2d(in=3, out=768, kernel=32, stride=32)
    # [160, 3, 224, 224] → [160, 768, 7, 7]
    # 224/32 = 7，所以每个维度有 7 个 patch，总共 49 个 patch

    # 第2步：展平并转置
    x = x.reshape(x.shape[0], x.shape[1], -1)   # [160, 768, 49]
    x = x.permute(0, 2, 1)                       # [160, 49, 768]
    # 现在每个 patch 是一个 768 维向量，序列长度 49

    # 第3步：添加 CLS token
    cls_token = self.class_embedding.to(x.dtype) + torch.zeros(B, 1, 768, ...)
    x = torch.cat([cls_token, x], dim=1)         # [160, 50, 768]
    # CLS token 是一个可学习的"汇总 token"，最终用它代表整张图片

    # 第4步：添加位置编码
    x = x + self.positional_embedding.to(x.dtype)
    # positional_embedding: [50, 768]
    # [160, 50, 768] + [50, 768] = [160, 50, 768]

    # 第5步：Pre-LayerNorm
    x = self.ln_pre(x)                           # [160, 50, 768]

    # 第6步：转置为 (L, N, D) 格式（Transformer 标准输入）
    x = x.permute(1, 0, 2)                       # [50, 160, 768]

    # 第7步：插入视觉 Prompt
    x = self.incorporate_prompt(x, visual_prompt)
    # incorporate_prompt（第 240-254 行）：
    #   visual_prompt [160, 8, 768] → permute → [8, 160, 768]
    #   x = cat([x[:1], visual_prompt, x[1:]], dim=0)
    #   x[:1] 是 CLS token: [1, 160, 768]
    #   x[1:] 是 49 个 patch: [49, 160, 768]
    #   拼接结果: [1+8+49, 160, 768] = [58, 160, 768]
    # 插入位置：CLS 之后，所有 patch 之前

    # 第8步：通过 12 层 Transformer
    x = self.transformer(x)                      # [58, 160, 768]
    # 每层内部：LayerNorm → Self-Attention → +残差 → LayerNorm → MLP → +残差

    # 第9步：转回 (N, L, D) 格式
    x = x.permute(1, 0, 2)                       # [160, 58, 768]

    # 第10步：取 CLS token 的输出
    x = self.ln_post(x[:, 0, :])                 # [160, 768]

    # 第11步：投影到共享空间
    x = x @ self.proj                            # [160, 768] @ [768, 512] = [160, 512]
    # self.proj 是一个可学习的投影矩阵（CLIP 预训练好的，冻结）

    return x  # [160, 512] ← 这就是图像的最终特征向量
```

##### C2：文本编码 → `encode_text(text, prompt)`（第 391-408 行）

```python
def encode_text(self, text, prompt=None):
    # text: [160, 69]，69 是预留了 prompt 位置后的长度
    # prompt: [160, 8, 512]

    # 第1步：Token Embedding
    x = self.token_embedding(text).type(self.dtype)
    # [160, 69] → [160, 69, 512]
    # 每个 token ID 被映射为一个 512 维的嵌入向量

    # 第2步：插入文本 Prompt
    prompt_length = 0
    if prompt != None:
        prompt_length = prompt.shape[1]          # 8
        x = self.incorporate_prompt(x, prompt)
        # incorporate_prompt（第 379-387 行）：
        #   x[:, :1, :] 是 SOT token: [160, 1, 512]
        #   x[:, 1:, :] 是剩余 token: [160, 68, 512]
        #   prompt: [160, 8, 512]
        #   拼接：cat([SOT, prompt, 剩余token], dim=1)
        #   结果：[160, 1+8+68, 512] = [160, 77, 512]
        # 正好占满 CLIP 的 77 个上下文位置！

    # 第3步：位置编码
    x = x + self.positional_embedding.type(self.dtype)
    # [160, 77, 512] + [77, 512] = [160, 77, 512]

    # 第4步：通过 Transformer
    x = x.permute(1, 0, 2)                      # [77, 160, 512]
    x = self.transformer(x)                      # [77, 160, 512]
    # 这里的 Transformer 有因果注意力掩码：每个词只能看到它前面的词
    x = x.permute(1, 0, 2)                      # [160, 77, 512]

    # 第5步：LayerNorm
    x = self.ln_final(x).type(self.dtype)       # [160, 77, 512]

    # 第6步：取 EOT 位置的输出并投影
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1) + prompt_length] @ self.text_projection
    # text.argmax(dim=-1): 找到每句文本中 token ID 最大的位置 = EOT token 的位置
    #   EOT 的 token ID 最大（49407），所以 argmax = EOT 在原始序列中的位置
    # + prompt_length: 因为前面插入了 8 个 prompt token，EOT 位置整体后移了 8
    # 取该位置的向量: [160, 512]
    # @ self.text_projection [512, 512]: [160, 512]
    # 最终文本特征：[160, 512]

    return x
```

##### C3：计算相似度矩阵（第 414-424 行）

```python
    # L2 归一化
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    # 每个图像向量除以自己的 L2 范数，长度变成 1
    # 例：[0.3, 0.4, 0.5] → norm=√(0.09+0.16+0.25)=√0.5≈0.707
    #                       → [0.424, 0.566, 0.707]
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # 计算余弦相似度（缩放后）
    logit_scale = self.logit_scale.exp()
    # 初始值约 14.3，训练过程中可能变化（因为 logit_scale 是 nn.Parameter）
    logits_per_image = logit_scale * image_features @ text_features.t()
    # [160, 512] @ [512, 160] = [160, 160]
    # 第 i 行第 j 列 = 第 i 张图片与第 j 句文本的缩放余弦相似度
    logits_per_text = logits_per_image.t()
    # [160, 160]，转置

    return logits_per_image, logits_per_text, image_features, text_features
```

#### 步骤 D（第 216-224 行）：Adapter 修正

```python
    # D1：图像 adapter
    img_adapter = self.adapter_img(image_features)
    # [160, 512] → MLP(512→512→GELU→512→512) → [160, 512]

    img_adapter_ratio = self.adapter_img_mapping(img_adapter)
    # Linear(512→1): [160, 512] → [160, 1]

    img_adapter_ratio = torch.clamp(img_adapter_ratio, min=0, max=0.1)
    # clamp 到 [0, 0.1]，限制 adapter 的最大影响

    image_features = img_adapter_ratio * img_adapter + (1 - img_adapter_ratio) * image_features
    # 如果 ratio=0.05，就只是 5% adapter + 95% 原始 = 几乎不变
    # 这就是"轻量级适配"的含义
```

```python
    # D2：文本 adapter（完全相同的结构）
    txt_adapter = self.adapter_txt(text_features)
    txt_adapter_scale = self.adapter_txt_mapping(txt_adapter)
    txt_adapter_scale = torch.clamp(txt_adapter_scale, min=0, max=0.1)
    text_features = txt_adapter_scale * txt_adapter + (1 - txt_adapter_scale) * text_features
```

#### 步骤 E（第 226-230 行）：不确定性分析

```python
    img_analy = self.analysis_img(image_features)
    # [160, 512] → Linear(512→512) → [160, 512]
    img_analy = torch.clamp(img_analy, min=0)
    # 所有值 ≥ 0，解释为"每个维度的标准差"

    txt_analy = self.analysis_txt(text_features)
    txt_analy = torch.clamp(txt_analy, min=0)
    # 同上
```

#### 步骤 F（第 234-248 行）：返回所有中间结果

```python
    ret = {
        "visual_prior_prompt": visual_prior_prompt_out,       # [160, 512] 先验特征
        "visual_prior_prompt_prior": visual_prior_prompt_prior, # [160, 512] 先验方差
        "loss_scale": loss_scale,                              # 标量
        "loss_scale_kl": self.loss_scale_kl,                  # 标量
        "loss_scale_un": self.loss_scale_un,                  # 标量
        "img_analysis": img_analy,                             # [160, 512] 图像不确定性
        "txt_analysis": txt_analy,                             # [160, 512] 文本不确定性
        "image_features": image_features,                      # [160, 512] adapter修正后
        "text_features": text_features,                        # [160, 512] adapter修正后
        "logits_per_image": logits_per_image,                  # [160, 160] 相似度矩阵
        "logits_per_text": logits_per_text,                    # [160, 160]
    }
    return ret
```

---

## 6. 损失计算：Monte Carlo 不确定性建模

**文件**：[objectives.py](CUP/vilt/modules/objectives.py) `compute_clip()` 第 404-526 行

这是 CUP 论文最核心的创新——用 Monte Carlo dropout 的思想估计不确定性，并基于不确定性加权损失。

### 6.1 函数入口

```python
def compute_clip(pl_module, batch):
    is_training_phase = pl_module.training
    NUM_SAMPLES = 10         # 采样 10 次（Monte Carlo 采样数）

    img = batch["clip_img"]            # [160, 3, 224, 224]
    txt = batch["clip_text_token"]     # [160, 69]

    # 调用 infer，得到所有中间结果
    infer = pl_module.infer((img, txt,))

    # 从 infer 返回值中取出需要的变量
    loss_scale_kl = infer["loss_scale_kl"]                    # 标量，KL损失权重
    loss_scale_un = infer["loss_scale_un"]                    # 标量，不确定性损失权重
    image_features = infer["image_features"]                  # [160, 512]
    text_features = infer["text_features"]                    # [160, 512]
    img_analy = infer["img_analysis"]                         # [160, 512] 图像不确定
    txt_analy = infer["txt_analysis"]                         # [160, 512] 文本不确定
    visual_prior_prompt = infer["visual_prior_prompt"]       # [160, 512] 视觉先验
    visual_prior_prompt_prior = infer["visual_prior_prompt_prior"]  # [160, 512] 先验方差
    logits_per_image = infer["logits_per_image"]              # [160, 160]
    logits_per_text = infer["logits_per_text"]                # [160, 160]
```

### 6.2 初始化 10 次采样的容器

```python
    # 为什么是 (NUM_SAMPLES, B, B)？
    # 每次采样得到一个 [160, 160] 的 softmax 概率矩阵
    # 10 次采样就需要 [10, 160, 160]
    NUM_SAMPLES = 10
    prob_total_img = torch.zeros((NUM_SAMPLES, B, B))        # [10, 160, 160]
    prob_total_txt = torch.zeros((NUM_SAMPLES, B, B))
    prob_total_img_prompt = torch.zeros((NUM_SAMPLES, B, D)) # [10, 160, 512]
    prob_total_img_prior = torch.zeros((NUM_SAMPLES, B, D))
    prob_toal_img_intra = torch.zeros((NUM_SAMPLES, B, D))
    prob_toal_txt_intra = torch.zeros((NUM_SAMPLES, B, D))
```

### 6.3 Monte Carlo 采样循环

```python
    for t in range(NUM_SAMPLES):          # 循环 10 次

        # ----- 采样1：图像特征 + 不确定性噪声（用于 clip_loss）-----
        img_epsilon = torch.randn(image_features.size()).cuda()
        # randn 生成标准正态分布 N(0,1) 的随机噪声
        # [160, 512]，均值为0，标准差为1
        img_logit = image_features + torch.mul(img_analy, img_epsilon)
        # img_analy 控制每个维度的噪声大小
        # 如果 img_analy[0, 3] = 0.01（小）→ 噪声很小 → 特征几乎不变
        # 如果 img_analy[0, 3] = 0.50（大）→ 噪声大 → 特征会波动
        # 这就是"不确定性"的体现：不确定的维度波动大

        # ----- 采样2：图像特征 + 固定方差的噪声（用于 uncertainty_loss）-----
        img_epsilon_intra = torch.randn(image_features.size()).cuda()
        img_logit_intra = image_features + torch.mul(img_analy.detach(), img_epsilon_intra)
        # .detach() 表示把 img_analy 从计算图中分离
        # 这意味着 uncertainty_loss 不会通过 img_analy 反向传播
        # 即 img_analy 只通过 clip_loss 被优化

        # ----- 采样3：图像特征 + 无梯度的随机噪声（用于 KL 散度）-----
        img_epsilon_prompt = torch.randn(image_features.size()).cuda()
        img_logit_prompt = image_features + torch.mul(img_analy.detach(), img_epsilon_prompt)

        # ----- 采样4：视觉先验特征 + 先验噪声（用于 KL 散度另一端）-----
        img_epsilon_prompt_prior = torch.randn(visual_prior_prompt.size()).cuda()
        img_logit_prior = visual_prior_prompt + torch.mul(visual_prior_prompt_prior, img_epsilon_prompt_prior)
        # visual_prior_prompt_prior 是 prior_analysis_img 的输出
        # 代表"先验的不确定性"

        # ----- 采样5：文本特征 + 不确定性噪声（用于 clip_loss）-----
        txt_epsilon = torch.randn(text_features.size()).cuda()
        txt_logit = text_features + torch.mul(txt_analy, txt_epsilon)

        # ----- 采样6：文本特征 + 固定方差的噪声（用于 uncertainty_loss）-----
        txt_epsilon_intra = torch.randn(text_features.size()).cuda()
        txt_logit_intra = text_features + torch.mul(txt_analy.detach(), txt_epsilon_intra)

        # ----- 计算本次采样的相似度矩阵 -----
        logit_scale = pl_module.model.logit_scale.exp()
        logit_img = logit_scale * img_logit @ txt_logit.t()          # [160, 160]
        logit_txt = logit_img.t()

        # ----- Softmax 得到概率分布 -----
        prob_total_img[t] = F.softmax(logit_img, dim=1)              # 第 t 次采样的概率
        prob_total_txt[t] = F.softmax(logit_txt, dim=1)

        # 图像内部（intra）概率
        img_intra = img_logit_intra                                   # [160, 512]
        txt_intra = txt_logit_intra                                   # [160, 512]
        prob_toal_img_intra[t] = F.softmax(img_intra, dim=1)
        prob_toal_txt_intra[t] = F.softmax(txt_intra, dim=1)

        # 先验相关概率
        prob_total_img_prompt[t] = F.softmax(img_logit_prompt, dim=1)
        prob_total_img_prior[t] = F.softmax(img_logit_prior, dim=1)
```

**关键理解：为什么需要 10 次采样？**

单次推理给出的结果是一个"点估计"——模型说相似度为 0.9。但模型对自己的判断有多确定？10 次采样加入了噪声，观察结果的波动：
- 如果 10 次 softmax 概率分布都很接近 → 模型很确定 → 这个样本更可靠
- 如果 10 次结果差异很大 → 模型不确定 → 这个样本在损失中的"话语权"应该降低

取平均后，不确定的维度受到的自然"惩罚"：
```python
prob_total_img_ave = torch.mean(prob_total_img, 0)   # [160, 160] 平均概率
```

### 6.4 三个损失的计算

#### 6.4.1 clip_loss：带不确定性的对比损失

```python
    # 第 493-499 行
    label1 = torch.arange(_b).cuda()          # [0, 1, 2, ..., 159]
    # ground truth：第 i 张图片应该匹配第 i 句文本

    criterion3 = nn.NLLLoss().cuda()           # 负对数似然损失

    loss_img = criterion3(torch.log(prob_total_img_ave), label1)
    # NLLLoss 要求输入是 log-概率，目标是对应类别的索引
    # prob_total_img_ave[i] 是一个 160 维概率分布（对每句文本的概率）
    # label1[i] = i，希望第 i 维的概率最大
    # 损失 = -log(prob_total_img_ave[i][i])

    loss_text = criterion3(torch.log(prob_total_txt_ave), label1)

    clip_loss = (loss_img + loss_text) / 2

    # 例：第 0 张图片对第 0 句文本的 softmax 概率 = 0.85
    # loss_img = -log(0.85) ≈ 0.163
    # 如果概率只有 0.01：
    # loss_img = -log(0.01) = 4.605（大了很多！）
```

**为什么是 NLLLoss 而不是 CrossEntropyLoss？**

CrossEntropyLoss = LogSoftmax + NLLLoss。这里已经手动做了 softmax，所以直接用 NLLLoss。

**不确定性的作用**：如果 img_analy 在第 0 张图片上值很大（不确定），10 次采样后 prob_total_img_ave[0] 会是一个"分散"的分布（不会集中在第 0 句文本上），导致 loss 自然变大。但模型可以通过梯度下降调小 img_analy 的值来降低 loss——这就是"让模型学会更确定"的机制。

#### 6.4.2 prior_loss：先验 KL 散度

```python
    # 第 505-508 行
    prob_total_img_prompt = prob_total_img_prompt.detach()   # 分离梯度
    # prob_total_img_prompt：基于随机 prompt 的 softmax 概率
    # prob_total_img_prior：基于 ContextCluster 先验的 softmax 概率

    prior_kl = F.kl_div(
        torch.log(prob_total_img_prior),    # log P(prior)
        prob_total_img_prompt               # P(prompt)
    )
    # KL(P(prior) || P(prompt))：衡量两个分布的距离
    # 如果两个分布完全相同，KL = 0
    # 差异越大，KL 越大
```

**这个损失的设计意图**：
1. ContextCluster 给出的先验不一定完美（它也是随机初始化的），但它有结构（CNN 架构），倾向于捕获位置相关的视觉信息
2. 随机视觉 prompt 是完全自由的形式，很容易学到"捷径"
3. KL 散度约束随机 prompt 的输出分布不能与 ContextCluster 先验的分布差太远
4. 效果：先验给随机 prompt 一个"方向感"，避免它乱学

#### 6.4.3 uncertainty_loss：跨模态不确定性对齐

```python
    # 第 500-504 行
    prob_toal_img_intra_ave = prob_toal_img_intra_ave @ prob_toal_img_intra_ave.t()
    # [160, 512] @ [512, 160] = [160, 160]
    # 这是图像特征内部的"自相似度"矩阵：不同样本图像特征之间的关系

    prob_toal_txt_intra_ave = prob_toal_txt_intra_ave @ prob_toal_txt_intra_ave.t()
    # [160, 160] 文本特征内部的"自相似度"矩阵

    uncertainty_loss = torch.abs(
        prob_toal_img_intra_ave - prob_toal_txt_intra_ave
    ).mean()
    # L1 距离：图像内部关系和文本内部关系应该一致
```

**这个损失的设计意图**：

图像和文本是两个不同的"世界"，但它们描述的应该是同一批样本。如果两幅图很相似，它们的文本描述也应该很接近。

举例：第 3 张图和第 7 张图都是机场，第 3 句和第 7 句文本也都是描述机场的。那么：
- `prob_toal_img_intra_ave[3, 7]` 应该大（两张机场图相似）
- `prob_toal_txt_intra_ave[3, 7]` 也应该大（两句机场描述相似）
- 如果不一致（比如图片很相似但文本不相似），说明模型对某个模态编码得不好

这个损失约束了两个模态在 "样本间关系" 上的表达一致性。

### 6.5 损失返回值

```python
    ret = {
        "clip_loss": clip_loss,                     # 主要对比损失
        "prior_loss": loss_scale_kl * prior_kl,     # 加权 KL 散度
        "uncertainty_loss": loss_scale_un * uncertainty_loss,  # 加权不确定性损失
    }
    return ret
```

回到 `training_step`：
```python
total_loss = clip_loss + loss_scale_kl * prior_kl + loss_scale_un * uncertainty_loss
```

---

## 7. 反向传播和参数更新

### 7.1 哪些参数会被更新？

**核心原则**：只有 `requires_grad=True` 的参数才会被更新。

回顾 `MyBackboneFinetuning.freeze_before_training()`（run_frozen.py 第 32 行）：
```python
def freeze_before_training(self, pl_module):
    self.freeze_clip(pl_module.model)    # 冻结 self.model（CLIP 的视觉编码器和文本编码器）
```

但 `self.model` 只是 `ViLTransformerSS` 的一个属性。其他属性（`prompt_embeddings`, `cluster_model`, `adapter_img` 等）在初始化时 `requires_grad` 默认就是 `True`。

**被更新的参数清单**（以 ViT-B/32 为例，大概估算）：

| 模块 | 参数量 | 说明 |
|------|--------|------|
| `prompt_embeddings` | 8×512 = 4,096 | 文本 prompt |
| `prompt_proj` | 512×512 + 512 = 262,656 | 文本 prompt 投影 |
| `visual_prompt_embeddings` | 8×768 = 6,144 | 视觉 prompt |
| `visual_prompt_proj` | 768×768 + 768 = 590,592 | 视觉 prompt 投影 |
| `visual_prompt_proj_gather` | 768×768 + 768 = 590,592 | 视觉 prompt 融合投影 |
| `cluster_model` | ~500,000 | ContextCluster 网络 |
| `prior_img` | 768×512 + 512 = 393,728 | 先验映射 |
| `prior_analysis_img` | 512×512 + 512 = 262,656 | 先验不确定性 |
| `cluster_mapping` | 768×1 + 1 = 769 | 门控标量 |
| `adapter_img` | 512×512×2 + 512×2 = 525,312 | 图像 adapter MLP |
| `adapter_img_mapping` | 512×1 + 1 = 513 | 图像门控 |
| `adapter_txt` | 525,312 | 文本 adapter MLP |
| `adapter_txt_mapping` | 513 | 文本门控 |
| `analysis_img` | 512×512 + 512 = 262,656 | 图像不确定性 |
| `analysis_txt` | 262,656 | 文本不确定性 |
| **总计** | **≈ 4.2M** | |

对比 CLIP ViT-B/32 约 150M 的参数量，CUP 只需要训练约 3% 的参数。

### 7.2 优化器配置

**文件**：[vilt_utils.py](CUP/vilt/modules/vilt_utils.py) `set_schedule()` 第 303-422 行

```python
def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]     # 5e-4
    wd = pl_module.hparams.config["weight_decay"]      # 0.01

    # no_decay 列表：这些参数不用 L2 正则化
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", ...]

    # 参数分四组：
    # 1. 需要 weight_decay 的非 head 参数 → lr=5e-4, wd=0.01
    # 2. 不需要 weight_decay 的非 head 参数 → lr=5e-4, wd=0
    # 3. 需要 weight_decay 的 head 参数 → lr=5e-4*lr_mult, wd=0.01
    # 4. 不需要 weight_decay 的 head 参数 → lr=5e-4*lr_mult, wd=0

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))

    # 学习率调度：warmup + 多项式衰减
    warmup_steps = int(max_steps * 0.1)   # 前 10% 步数做 warmup
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, warmup_steps, max_steps, lr_end=0, power=1
    )
    # 学习率从 0 线性增长到 lr（warmup 阶段）
    # 然后从 lr 线性衰减到 0

    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
```

### 7.3 梯度累积

```python
# run_frozen.py 第 75-78 行
grad_steps = _config["batch_size"] // (
    _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
)
# = 256 // (160 × 1 × 1) = 1（对于这些配置）
# 如果 batch_size=256, per_gpu_batchsize=64, num_gpus=1:
# = 256 // 64 = 4（每 4 步更新一次参数）
```

梯度累积意味着：前向传播 N 次，把梯度累加起来，然后一次性更新参数。这等价于用更大的 batch size 训练。

---

## 8. 验证和评估

### 8.1 验证触发

在 `run_frozen.py` 中设置了：
```python
val_check_interval = 1.0   # 每个 epoch 验证一次
```

每个 epoch 训练结束后，PyTorch Lightning 自动调用：
```python
def validation_step(self, batch, batch_idx):
    self.hparams.config["current_epoch"] = self.current_epoch
    vilt_utils.set_task(self)
    output = self(batch)   # 和训练一样的前向传播
```

然后调用 `validation_epoch_end` → `epoch_wrapup`。

### 8.2 epoch_wrapup 中的评估

**文件**：[vilt_utils.py](CUP/vilt/modules/vilt_utils.py) `epoch_wrapup()` 第 50-285 行

```python
def epoch_wrapup(pl_module):
    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), results = compute_clip_recall(pl_module)
```

### 8.3 compute_clip_recall：评估核心

**文件**：[objectives.py](CUP/vilt/modules/objectives.py) `compute_clip_recall()` 第 537-834 行

这个函数是整个项目中最长的函数。分三步走：

#### 阶段 A：提取所有文本特征

```python
    # 第 562-630 行
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset()
    # 创建一个"无假样本"的 Dataset，image_only=False
    # 数据集大小 = 所有 (图片, 描述) 对的数量

    text_loader = DataLoader(text_dset, batch_size=64, ...)

    text_preload = list()
    for _b in text_loader:                      # 遍历整个数据集
        text_preload.append({
            "text": _b["clip_text"],            # 原始文本字符串
            "clip_text": _b["clip_text_token"], # Tensor [64, 69]
            "img_index": _b["img_index"],       # 每张图片的编号
            "clip_img": _b["clip_img"],         # Tensor [64, 3, 224, 224]
            # ...
        })

    # 提取文本特征
    txt_feats = torch.zeros(size=[N, D])        # N=数据集样本数, D=512
    for i, _batch_txt in enumerate(text_preload):
        _t = _batch_txt["clip_text"]            # [64, 69]
        out = pl_module.txt_embeds(_t)          # 调用 txt_embeds（只用文本 prompt）
        txt = out["text_feats"]                 # [64, 512]
        txt_feats[i*64 : i*64+_l] = txt         # 填充到 txt_feats 中
```

**`txt_embeds()` 方法**（vilt_module.py 第 251-271 行）：

```python
def txt_embeds(self, batch):
    txt = batch                                # [64, 69]
    B = txt.shape[0]
    text_prompt = self.prompt_dropout(
        self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
    )
    # 和训练时一样生成文本 prompt

    text_features = self.model.encode_text(txt, text_prompt)
    # [64, 512] 纯文本编码（不需要图片）

    # adapter 修正（和训练时一样的流程）
    txt_adapter = self.adapter_txt(text_features)
    txt_adapter_scale = self.adapter_txt_mapping(txt_adapter)
    txt_adapter_scale = torch.clamp(txt_adapter_scale, min=0, max=0.1)
    text_features = txt_adapter_scale * txt_adapter + (1 - txt_adapter_scale) * text_features

    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    return {"text_feats": text_features}       # [64, 512]
```

#### 阶段 B：提取所有图像特征

```python
    # 第 658-726 行
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset(image_only=True)
    # image_only=True：只返回图片，不返回文本
    # 数据集大小 = 不同图片的数量（不是所有图文对的数量！）

    image_loader = DataLoader(image_dset, batch_size=64, ...)

    img_feats = torch.zeros(size=[M, D])        # M=不同图片数, D=512
    for i, _batch_img in enumerate(image_loader):
        _i = _batch_img["clip_img"]             # [64, 3, 224, 224]
        out = pl_module.img_embeds(_i)          # 调用 img_embeds
        feature = out["image_feats"]            # [64, 512]
        img_feats[i*64 : i*64+_l] = feature
```

**`img_embeds()` 方法**（vilt_module.py 第 273-301 行）：

```python
def img_embeds(self, batch):
    img = batch                                # [64, 3, 224, 224]
    B = img.shape[0]

    # 完整的视觉 pipeline（和训练时一样）
    visual_prior_prompt = self.cluster_model(img)
    visual_prior_prompt = visual_prior_prompt.unsqueeze(1).repeat(1, self.prompt_length, 1)
    visual_ratio = torch.clamp(self.cluster_mapping(visual_prior_prompt), 0, 1)
    visual_random_prompt = self.visual_prompt_dropout(
        self.visual_prompt_proj(self.visual_prompt_embeddings).expand(B, -1, -1))
    visual_prompt = (1 - visual_ratio) * visual_prior_prompt + visual_ratio * visual_random_prompt
    visual_prompt = self.visual_prompt_gather_dropout(
        self.visual_prompt_proj_gather(visual_prompt))

    image_features = self.model.encode_image(img, visual_prompt)

    # adapter 修正
    img_adapter = self.adapter_img(image_features)
    img_adapter_ratio = self.adapter_img_mapping(img_adapter)
    img_adapter_ratio = torch.clamp(img_adapter_ratio, min=0, max=0.1)
    image_features = img_adapter_ratio * img_adapter + (1 - img_adapter_ratio) * image_features

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    return {"image_feats": image_features}     # [64, 512]
```

#### 阶段 C：计算检索指标

```python
    # 第 754 行
    scores = logit_scale * img_feats @ txt_feats.t()
    # [M, N] 相似度矩阵
    # M = 不同图片数（如 1093）
    # N = 所有图文对（如 5465 = 1093 × 5）

    # === 图像检索（Image Retrieval）===
    # 给定图片，找最匹配的文本
    topk1 = scores.topk(1, dim=1)            # 每行取最大值的索引
    topk5 = scores.topk(5, dim=1)
    topk10 = scores.topk(10, dim=1)

    # 判断命中：topk 的 img_index 是否等于当前图片的 img_index
    ir_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()
    # iids: [M]，每张图片的真实编号
    # topk1_iids: [M, 1]，top-1 文本所对应的图片编号
    # 如果匹配：unsqueeze(1) 使 iids 变成 [M, 1]，与 topk1_iids [M, 1] 逐元素比较
    # .max(dim=1)[0]: 取每行的最大值（1=命中, 0=未命中）
    # .mean(): 所有图片的命中率

    # === 文本检索（Text Retrieval）===
    # 给定文本，找最匹配的图片（操作相同，在 dim=0 上）
    topk1 = scores.topk(1, dim=0)
    tr_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    # === 改进的评估指标 ===
    # 因为每张图片有 5 句描述，原始指标会把人造的多说成多次命中
    # 改进版：同图多句描述只算一次命中
    ttr_top1 = tr_top_k(iids.unsqueeze(0), text_correspondence, rep_topk1_iids, 1)
    iir_top1 = ir_top_k(rep_texi_id.unsqueeze(1), text_correspondence, topk1_iids, 1)
```

**tr_top_k 怎么改进的**（第 969-985 行）：

```python
def tr_top_k(img_idx, relate, idx, k):
    # img_idx: [1, M]，每张图片的编号
    # relate: {text_id: [所有描述这张图片的img_id列表]}
    # idx: [M, k]，top-k 的文本编号
    batch = idx.shape[0]
    top_sum = 0
    for i in range(batch):
        for j in range(k):
            rep_text = idx[i][j]                            # 第 i 个检索结果的第 j 个候选
            all_relation = relate[str(rep_text.item())]      # 这个文本描述的所有图片
            if img_idx[0, i] in all_relation:               # 当前图片在列表中 → 命中
                top_sum += 1
                break                                        # 命中一次就够了，不重复计算
    return top_sum / batch
```

**最终输出的评估结果**：

```
################  Improved Evaluation Metrics  ################
im_ir_top1: 23.45%, im_ir_top5: 52.31%, im_ir_top10: 68.72%,
im_tr_top1: 25.18%, im_tr_top5: 55.67%, im_tr_top10: 72.14%,
im_mean: 49.58%

################  Original Evaluation Metrics  ################
ir_top1: 25.12%, ir_top5: 54.89%, ir_top10: 71.23%,
tr_top1: 26.87%, tr_top5: 58.34%, tr_top10: 74.56%,
mean: 51.84%
```

改进指标通常比原始指标低几个百分点，因为它避免了"同图多句"带来的虚高。

---

## 9. 完整调用链速查表

下面用一个具体的训练 step 为例，标注每个函数调用、文件位置和关键行号。

```
训练一个 step 的完整调用链

① shell: run_all.sh (第 40-41 行)
    python run_frozen.py with task_finetune_irtr_rsitmd_randaug_ViTB32 ...

② run_frozen.py: main() (第 41 行)
    @ex.automain → Sacred 加载 config.py 的命名配置合并

③ config.py: task_finetune_irtr_rsitmd_randaug_ViTB32 (第 429 行)
    设置 datasets=["rsitmd"], loss_names={"clip":1}, clip_model="ViT-B/32", ...

④ run_frozen.py: main() 第 46 行
    dm = MTDataModule(_config, dist=False)
    → BaseDataModule.__init__()
    → 根据 datasets 自动选择 RSITMDCaptionKarpathyDataModule

⑤ run_frozen.py: main() 第 48 行
    model = ViLTransformerSS(_config)
    → vilt_module.py: ViLTransformerSS.__init__() (第 214 行)
        ├→ clip.load("ViT-B/32") (第 245 行)
        │    → prompt_clip/clip.py: load() (第 94 行)
        │    → prompt_clip/model.py: build_model() (第 451 行)
        │    → 返回 CLIP 实例
        ├→ 初始化 prompt_embeddings, prompt_proj (第 255-263 行)
        ├→ 初始化 visual_prompt_embeddings, visual_prompt_proj (第 273-292 行)
        ├→ 初始化 cluster_model (第 287 行)
        │    → context_cluster.py: cluster_model() (第 1082 行)
        ├→ 初始化 prior_img, prior_analysis_img, cluster_mapping (第 290-298 行)
        ├→ 初始化 adapter_img, adapter_txt (第 300-312 行)
        └→ 初始化 analysis_img, analysis_txt (第 283-289 行)

⑥ run_frozen.py: main() 第 81-109 行
    trainer = pl.Trainer(...)
    callbacks = [MyBackboneFinetuning(), checkpoint_callback, lr_callback]

⑦ run_frozen.py: main() 第 113 行
    trainer.fit(model, datamodule=dm)
    → PyTorch Lightning 接管训练循环

⑧ Lightning 内部：
    ├→ MyBackboneFinetuning.freeze_before_training() (第 32 行)
    │    → freeze_clip(pl_module.model)  冻结 CLIP 所有参数
    │
    ├→ set_schedule() → vilt_utils.py 第 303 行
    │    配置 AdamW 优化器 + Warmup + 多项式衰减
    │
    └→ 每个 epoch 的每个 batch：

        ⑨ DataLoader 获取 batch
            → base_dataset.py: get_suite() (第 174 行)
                ├→ get_image() (第 120 行): arrow → PIL → CLIP预处理 → [3,224,224]
                └→ get_text() (第 142 行): 文本 → tokenize → [1,69]
            → base_dataset.py: collate() (第 192 行)
                堆成 batch [160,3,224,224] 和 [160,69]

        ⑩ vilt_module.py: training_step() (第 338 行)
            └→ set_task() → vilt_utils.py 第 296 行
                current_tasks = ["clip"]

        ⑪ vilt_module.py: forward() (第 303 行)
            └→ compute_clip() → objectives.py 第 404 行
                └→ infer() → vilt_module.py 第 193 行
                    ├→ cluster_model(img)              [160,3,224,224]→[160,768]
                    ├→ prior_img                       [160,768]→[160,512]
                    ├→ prior_analysis_img              [160,512]→[160,512]
                    ├→ 先验与随机prompt融合              [160,8,768]
                    ├→ text_prompt生成                  [160,8,512]
                    ├→ CLIP.encode_image                [160,3,224,224]→[160,512]
                    │    → prompt_clip/model.py 第 375 行
                    │    → VisionTransformer.forward() 第 257 行
                    │        ├→ Conv2d patch化          [160,768,7,7]
                    │        ├→ flatten + CLS           [160,50,768]
                    │        ├→ incorporate_prompt      [160,58,768]
                    │        └→ Transformer×12          [58,160,768]→取CLS→[160,512]
                    ├→ CLIP.encode_text                 [160,69]→[160,512]
                    │    → prompt_clip/model.py 第 391 行
                    │        ├→ token_embedding         [160,69,512]
                    │        ├→ incorporate_prompt      [160,77,512]
                    │        └→ Transformer×12          [77,160,512]→取EOT→[160,512]
                    ├→ adapter修正                      图像/文本各自+adapter
                    └→ analysis                         计算不确定性

        ⑫ objectives.py: compute_clip() 第 440-479 行
            Monte Carlo × 10:
            ├→ 噪声扰动特征 10 次
            ├→ 每次计算 softmax 概率
            └→ 取平均

        ⑬ objectives.py: compute_clip() 第 493-508 行
            ├→ clip_loss = NLLLoss(avg_log_prob, labels)
            ├→ prior_loss = KL_div(log(P_prior), P_prompt)
            └→ uncertainty_loss = |img_intra - txt_intra|.mean()

        ⑭ vilt_module.py: training_step() 第 343 行
            total_loss = clip_loss + w_kl×prior_kl + w_un×uncertainty_loss
            return total_loss → Lightning 自动反向传播

        ⑮ Lightning: optimizer.step()
            只更新 requires_grad=True 的参数（~4.2M，占全部的 ~3%）

    └→ 每个 epoch 结束：

        ⑯ vilt_module.py: validation_epoch_end() 第 355 行
            └→ vilt_utils.py: epoch_wrapup() 第 50 行
                └→ objectives.py: compute_clip_recall() 第 537 行
                    ├→ txt_embeds() 提取所有文本特征 [N, 512]
                    ├→ img_embeds() 提取所有图像特征 [M, 512]
                    ├→ scores = img @ txt.t()  [M, N]
                    ├→ 计算 ir_r@K, tr_r@K
                    └→ 计算改进版 iir_r@K, ttr_r@K
```

---

## 总结：CUP 到底做了什么（一句话概括）

**CLIP 权重不动，给图像和文本各加 8 个可学习的"暗号 token"（Prompt），用 ContextCluster 从图片中自动提取视觉线索来初始化视觉暗号，再用 Adapter 微调输出，用 Monte Carlo 采样估计模型不确定性来引导训练。最后只训练约 3% 的参数（~4M），就能让通用 CLIP 在遥感检索任务上大幅提升。**
