# Experiment Audit Report

- 日期：2026-11-20
- 审计：独立审阅代理 `/root/coverage_integrity`；执行端确认调度为 GPT-6-Astra ultra。数值复核另委派 `/root/coverage_integrity/deterministic_metrics`。
- 审计级别：**same-family / provisional**。
- 总体结论：**WARN**；确定性证据检查：**PASS**。
- 终稿：`实验交付.md`，SHA-256 `cd87a513b4cfc7de9802a62f3d938fae2a3ea1d272241925c0a220d14b995f97`。
- 追踪：`.aris/traces/experiment-audit/2026-11-20_coverage/`。

全部检查使用归档原始输出与现存代码。先完成像素盲审，再读取标签和分数；受测策略、结果文件和用户原文保持原样。WARN 对应已披露的助手标注与样本范围，现存结果的数值与主张吻合。

## A. Ground Truth Provenance — WARN

16 道候选题的初始方向由像素参照点坐标生成，审阅者第一阶段只看 `review/panel_00.jpg` 至 `panel_03.jpg` 和 `blind_questions.json`。488.tif 的屋顶、1087.tif 的船组因指代歧义剔除；其余14项方向与坐标结果一致。`freeze_fresh.py` 按像素票据、方向一致与重叠筛查规则保留全部合格项，最终选择与独立重算相同。

14图来自此前已参与检索评估的210图验证库；它们与旧方向/E1清单及16张训练图的图名、RGB哈希分离。旧资料的11项来源哈希均吻合。归档同场景配准共1,417次、检出0项。该检查的阈值及场景范围保存在代码，地理独立性仍为待确认。方向标签属于助手像素标注，交付稿已说明来源。

证据：`code/prepare_labels.py:11`、`code/freeze_fresh.py:5`、`code/source_inventory.py:9`、`code/overlap.py:31`、`review/pixel_votes.json:1`、`实验交付.md:18`。

## B. Score Normalization — PASS

普通读出保留两个朝向的原始余弦分差，严格双视图正确要求原图正差且半转图负差，统计直接按固定题数求值。代码中的L2归一化属于嵌入相似度定义。性能计算未使用受测输出最大值、最小值或均值作归一化分母。

两视图诊断另计算 `(u-v)/2` 及 `I(u>v)+0.5 I(u=v)`。该变换构造相反预测，50%对应源图整体符号翻转下的条件参考。两种措辞绑定在同一源图，独立整数分布复算全部精确p值均吻合。原始零差按未判对计、centered零差半分，已在复现说明明确；现有8条原始endpoint包含零差。

证据：`code/analyze.py:18`、`code/analyze.py:28`、`code/metrics.py:8`、`CUP/vilt/modules/vilt_module.py:558`、`实验交付.md:42`、`复现.md:14`。

## C. Result File Existence — PASS

独立复算7,904个原始方向分差、480条endpoint、480条覆盖、180条先验干预；最大算术差为1.0998146837692957e-15。部署/覆盖/donor汇总分别96/96/36条，108组按源图bootstrap配对区间全部吻合。进一步核对同状态240份状态及120条endpoint、骨干对照120条endpoint、选项顺序480次呈现、图库12项Recall，结果全部吻合。全部p值另用整数分布算法复算，matched两组区间按20,000次源图bootstrap复算。

输入锁、30图输出完成清单、张量哈希、补充任务回执均完整；20个本机与服务器复算输出的校验表与当前文件吻合。核心决策为旧难例对象提示4/10、场景3/10，新图八次平均对象0/28、场景1/28。预设两项门槛均未达到，保留历史部署的结论获得支持。

同状态新图为Agent 17/28、CUP 1/28；共同正确0、仅Agent正确17、仅CUP正确1、共同失败10。差值57.14个百分点，95%源图bootstrap区间[32.14,78.57]。signed report中的数值、实例、图表均与其指定读出口径一致。

证据：`code/analyze.py:38`、`code/analyze.py:171`、`results/verification.json:1`、`results/local_recompute.json:1`、`results/matched_choice.json:1`、`实验交付.md:28`、`实验交付.md:40`。

## D. Dead Code / Actual Binding — PASS

实际策略是历史LoRA，active hash `2f7e0f3af73b835f2111152bfd76a47f3b811737f24fa51d24c6b6234e005ab6`；实际读出严格加载冻结CUP checkpoint `09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc`。`SCORER`只是`shared.py:7`的未使用常量。六份本地CUP/Agent源文件哈希与环境记录相同。训练32个贪心状态、向量与token轨迹均与前轮逐值吻合。

matched脚本读取回答字母前的最后层状态，检查当次logits与语言头重算一致，将同一状态送进CUP。240份状态的哈希、已保存A/B logits及top token与原选择记录逐项一致。语言头随选项位置翻转符号；CUP始终比较相同语义句子，再对两次顺序取平均，避免对同一文本误做符号翻转。`fixed`、`recalls`、`ci`、`endpoint`、`coverage`、`sign_test`均有调用与结果记录。

证据：`code/run.py:33`、`code/readouts.py:14`、`code/matched_choice.py:33`、`code/matched_post.py:23`、`code/analyze.py:111`、`CUP/vilt/modules/vilt_module.py:584`。

## E. Scope / Matched Budget / Oracle — WARN

本轮覆盖一个历史策略、16旧训练图及14新方向图/8场景块。每提示共540轨迹，种子槽位匹配、每次上限3轮×64 tokens、过滤数全0；实际生成tokens为5,757/7,655，工具调用266/375。预算匹配指候选数量及生成上限，正文已给出实际词元差异。

候选覆盖允许两朝向分别挑最有利状态，实际同槽成功另表记录。外图先验训练每目标28份，排除整个反向训练图对；新图32份。两侧固定共享一份状态与两侧分别挑选严格区分：新图合并状态库为12/28与18/28，旧10难例合并分别挑选5/10。该结果描述已枚举状态集合的可达范围。

直接选择17/28是两种选项顺序平均，固定顺序实际top-1为12/28、10/28，四次全部正确9/28；112次新图呈现全词表首选均为A/B。关闭历史LoRA为16/28，四次全对7/28。主文与图均披露顺序平均。matched在主结果后追加，检验choice-prompt状态经整个CUP检索链路后的排序；因果定位到单独投影层尚需独立干预，正文保留了这一研究任务。

普通检索保持210图、1,050句和原顺序。两套共420份图像向量与前轮归档逐位一致，文本银行哈希一致；mR复算为0.6005540777564587和0.4503835978835979。首句条件使用同图第一条描述，该描述也在图库内；交付稿明确说明。新方向提问和同状态试验的结果单独报告。

证据：`code/run.py:67`、`code/run.py:81`、`code/analyze.py:131`、`code/order_checks.py:23`、`code/gallery.py:15`、`inputs/matched_choice_protocol.json:2`、`实验交付.md:7`、`实验交付.md:38`、`实验交付.md:46`。

## F. Evaluation Type — PASS（已分类）

- 方向诊断：`synthetic_proxy`，细分为与受测策略分离的助手像素标注；同模型家族暂定审核。
- 原图库检索：`real_gt`，采用原始UCM图文对应关系；历史验证图库复放。

原始标签、独立票据、坐标规则和模型输出各自保存，来源可追溯。后续泛化验证可加入人工标注复核与新的独立图集。

## 审计限制与主张影响

本机审计对归档张量、原始logits、源码、来源与汇总完成独立核对。全词表逐值复原由运行时assert及30份`all_vocab_logits_exact`回执支持；本机未再次载入完整生成模型执行该前向。原始Arrow解码及配准由归档运行校验支持，审阅者另完成像素盲审、来源哈希和历史向量交叉核验。

已审签正文关于保留权重、候选覆盖有限、同状态两个读出存在差异以及优先验证方向辅助评分的判断，均有对应证据。后续训练优越性、未见图集上的泛化、单个投影模块的因果责任保留为研究问题。本次确定性结果可接受；语义审计维持**same-family / provisional / WARN**。
