# 审阅通信记录
主审阅代理为 `/root/format_reward_integrity`。下列为结果封存前的通信。

> 初步核对：训练/验证/16图细节输入与前包逐字节一致，锁定哈希和 E1 五个文件哈希均吻合；训练24图/7类，验证210图/21类，16图细节覆盖10类，训练与验证图号/RGB均无直接重复。需在最终报告持续明确 native 使用每图首条真描述构建先验，generic 使用全图固定句，detail 使用场景提示；E1 标签包含既有用户确认数量及同家族助手像素复核，属于诊断集。小型代码维护问题：shared.locked() 引用过时 ROOT/name 及 selected.json，readouts.scorer_for()/load_model(True) 无本轮调用，均未进入实际指标路径。待全输出及交付文本后给结论。

> 补充直接核验：本轮 E1 的 features/rows/settings 与原 E1 outputs/AGSE_archived 对应文件字节相同；CUP/agent/rollout.py、grpo/reward.py、tools.py 与 environment.json 运行源码哈希相同。已查看前包全部细节拼图，16题修订后指称与旋转方向成立。统计单位检查：数量 A/B 各22图；属性、关系各36图、18互逆对，维度内图像无重复。需保留原始首句条件下图像先验的身份说明，格式函数实际只要求完整 answer 标签，空内容同样得0.2。当前尚无额外阻断项。

> 我将遵守只写 review/EXPERIMENT_AUDIT.md/.json 的范围，JSON 引用指定 .aris/traces/experiment-audit/2026-11-20_format/。完整审阅对话和证据命令请由主代理落入该 trace 目录；报告内会列已读取输入哈希及独立复算摘要。审阅代理使用继承模型，元数据将如实标注 same-family/provisional，避免写入未核实的模型/推理档位。

执行者回复

> 收到。trace 由我在实验目录内封存。spawn 时显式请求的模型是 gpt-6-astra、reasoning_effort=ultra（fork_turns=none）；可以把它们记为请求配置，实际后端若不能自行确认可分别注明。请继续独立复核，结果同步完会发路径。

> 评价类型会分开记录：全库检索的图文归属是 real_gt；16图关系与 E1 新增属性/关系属于视觉助手构造并复核的自定义标签（按技能枚举记 synthetic_proxy，并注明来源于真实像素、与被测策略输出独立）；历史数量标签记用户确认的外部标签。格式/EOS/工具计数是 self_supervised_proxy 行为测量。这样避免将所有辅助题统称数据集原生标注。当前底层源码和已有声明已充分披露助手来源。

> 源代码逐行审阅、输入/外部源码哈希、全部16图视觉复查和已到输出的CPU独立复算已完成。默认内核数值对照也独立确认第2组已分化、第5组才首次出现差异格式信号，主实验与之分开保存符合证据。下一步等待你通知最终同步和交付文本路径，再统一复算所有17个状态、区间和报告数值。

> 找到本地归档 tokenizer.json，与 environment 中运行时 tokenizer 的 SHA256 完全一致。已用独立 tokenizers 解码 A 两组全部训练轨迹，逐条复算 r_format，并用自行实现的平均并列名次重算两项语义奖励：全部吻合。A off 3/192、A on 4/192 条得到格式分，均有文字内容；最先触发在第5组。完成后同法覆盖 B，审计证据无需依赖训练审计脚本的自述。

> 额外静态统计复查代理 /root/format_reward_integrity/static_stat_check 已覆盖聚类、并列召回和 EOS 实现，目前未发现实质错误。请将此只读子审阅一并写入 trace。另直接核验 design_receipt 的7个哈希均与 Git提交 0d6f0b7（2026-11-20 16:02:25+08:00）内容一致，且当前文件相同。

> 只读统计子复查已完成：330组并列概率枚举与公式最大差1.11e-16；E1的156条输入与归档逐条对应；首token EOS范围、图级/图对级配对bootstrap均通过。该子代理最终答复可直接封存进审计trace。我继续等待最终同步信号。

> 新增直接权重复核：已CPU读取 A off/on、B off 的 resume.pt，按保存参数顺序重新SHA256，与末步日志和 update.json 一致；各6/12/24检查点文件哈希也吻合。需注意报告的“24步”口径：A off 优化器每参数实际 step=22，A on=21，B off=24；其余为零优势/无梯度训练组（train.py跳过adv=0）。建议用“每组24个候选组；有效参数更新/非零梯度组另列”并保留 training_checks 对应计数。
