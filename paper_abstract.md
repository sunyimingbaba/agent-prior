# 论文摘要(定稿 2026-09-28)

## English (215 words)

Cross-modal remote sensing image-text retrieval (CMRSITR) aims to achieve bidirectional matching between remote sensing images and textual descriptions. Benefiting from large pretrained vision-language models, CMRSITR has advanced substantially in recent years. However, existing prompt-based methods typically rely on static visual priors derived from image-level feature statistics, which cannot fully bridge the information asymmetry between semantically sparse descriptions and visually complex remote sensing scenes, while the potential of agent-based reasoning, domain-specific tool augmentation, and retrieval-oriented reinforcement learning for constructing richer semantic priors remains largely unexplored. To this end, we propose a tool-augmented agent reinforcement learning framework. First, considering that static visual priors are query-agnostic, we distill query-aware priors from a multimodal agent's reasoning over the image and query. The query-aware priors are rich in query-relevant semantics, which help bridge the information asymmetry between sparse texts and complex images. Second, to compensate for generic multimodal agents' lack of remote-sensing domain expertise, we design a scene classification tool with an iterative invocation process, enabling the agent to evolve its analysis through tool-callback loops. Third, we find that, without any supervised fine-tuning, cold-starting the agent directly with reinforcement learning outperforms SFT-based initialization at lower cost, aligning agent behavior with retrieval objectives. Experiments on three public benchmarks demonstrate that our framework achieves competitive performance compared with many existing CMRSITR methods.

## 中文

跨模态遥感图像-文本检索(CMRSITR)旨在实现遥感图像与文本描述之间的双向匹配。得益于大规模预训练视觉-语言模型,CMRSITR 近年来取得了显著进展。然而,现有基于提示的方法通常依赖由图像级特征统计导出的静态视觉先验,难以弥合语义稀疏的文本描述与视觉复杂的遥感场景之间的信息不对称,同时基于智能体的推理、领域专用工具增强和面向检索的强化学习在构建更丰富语义先验方面的潜力尚未得到充分探索。为此,本文提出一个工具增强的智能体强化学习框架。首先,考虑到静态视觉先验与查询无关,本文将多模态智能体对图像与查询联合推理的最终隐藏状态蒸馏为查询感知先验。该先验富含查询相关语义,有助于弥合稀疏文本与复杂图像之间的信息不对称。第二,为弥补通用多模态智能体在遥感领域知识上的不足,本文设计了场景分类工具与迭代调用流程,使智能体通过工具回调循环逐步演化分析。第三,我们发现,即便不做任何监督微调(SFT),直接以强化学习对智能体进行冷启动训练,反而比SFT初始化表现更好且成本更低,同时使智能体行为对齐检索目标。在三个公开基准数据集上的实验表明,本文框架相比许多现有 CMRSITR 方法取得了具有竞争力的性能。

## 定稿说明

- 中文由用户最终审定(2026-09-28);英文同步 215 词(≤250)。
- 结构:①任务定义 → ②问题与 gap → ③框架(工具增强的智能体强化学习框架)→ ④⑤⑥组件(原因式/目的式/发现式)→ ⑦结果。
- ②句三方向由④⑤⑥ First/Second/Third 承接(CUP 同款结构)。
- 表述安全线:结果句用 "competitive" 覆盖现有方法(不踩纯推理先验高于主方法的坑);冷启动优于 SFT 有实验支撑(54.90/39.00/59.78 vs 52.16/30.24/41.62)。
- 已知风险(用户知情后仍保留):⑦句与 CUP 结果句同构度较高;①句与 CUP 首句骨架相似。
