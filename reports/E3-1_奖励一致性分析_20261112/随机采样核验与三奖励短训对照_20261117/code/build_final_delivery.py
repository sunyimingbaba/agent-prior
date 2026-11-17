#!/usr/bin/env python3
"""Build a new, traceable Chinese delivery from completed RFPO evidence.

Usage (paths may contain spaces):
  python3 build_final_delivery.py --raw-root /absolute/raw --output-root /absolute/new

Inputs are read only. The output directory must be new and outside the raw tree.
No model loading, metric recomputation, remote access, or synthetic data is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


POLICIES = ("baseline_continuation0", "proxy_rank", "quantity_pair", "quantity_hard")
ARMS = POLICIES[1:]
PROTOCOLS = ("repository_duplicate_token_positive", "strict_index", "strict_tie_averaged")
LABELS = {
    "baseline_continuation0": "balanced32（本阶段 0 步）",
    "proxy_rank": "代理排名奖励",
    "quantity_pair": "数量文本差奖励",
    "quantity_hard": "困难数词奖励",
    "repository_duplicate_token_positive": "重复 token 正例",
    "strict_index": "严格索引",
    "strict_tie_averaged": "严格平局平均",
}
COLORS = ("#687786", "#0072B2", "#E69F00", "#009E73")
VERIFY_FILES = ("independent_raw_verification.json", "independent_training_verification.json",
                "independent_metric_verification.json")
REQUIRED = (
    "plan.json", "analysis_protocol_lock.json", "cached_replay_addendum.json", "analysis.json",
    *VERIFY_FILES, "post_job_result.json", "prompt_selection_lock.json",
    "continuation_job_result.json", "continuation_job_result_v2.json",
    "training/proxy_rank/failure.json", "probability_gate_v2/result.json",
    "probability_gate_v2/rounds.json", "qualitative_case_lock.json",
    "source_executed/rl/plan.json", "inputs/balanced_checkpoint_reference.json",
)
GATE_LABELS = {
    "primary_lower_gt0": "主数量改善：困难数词 − 代理排名的 95% CI 下限 > 0",
    "retrieval_lower_gt_minus0.02": "主检索警戒：困难数词 − 阶段起点的 mR 95% CI 下限 > −2 百分点",
    "sampled_protocol_all_policies": "四策略随机严格完成率 ≥90%，自行生成结果标签比例 ≤2%",
    "same32actual_updates": "三奖励组均完成 32 次实际优化器更新",
}


class DeliveryError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DeliveryError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invalid_constant(value):
    raise DeliveryError(f"JSON 包含非有限值：{value}")


def close(left, right):
    return math.isclose(float(left), float(right), abs_tol=1e-12, rel_tol=1e-12)


def pct(value):
    return f"{100 * value:.2f}%"


def number(value, digits=4):
    return "未定义" if value is None else f"{value:.{digits}f}"


def pp_interval(item, field):
    lo, hi = item["interval95"]
    return f"{100 * item[field]:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"


def md_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(md_cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(md_cell, row)) + " |" for row in rows)])


def link(label, path):
    return f"[{label}](<{path.resolve()}>)"


def unique(items, **keys):
    selected = [item for item in items if all(item.get(key) == value for key, value in keys.items())]
    require(len(selected) == 1, f"对照条目数量应为 1：{keys}，实际 {len(selected)}")
    return selected[0]


def quantity_contrast(analysis, policy, reference, mode="greedy", metric="B_hard_correct"):
    return unique(analysis["quantity_contrasts"], policy=policy, reference=reference,
                  mode=mode, metric=metric)


def retrieval_contrast(analysis, policy, reference, protocol):
    return unique(analysis["retrieval_contrasts"], policy=policy, reference=reference, protocol=protocol)


def load_and_validate(raw):
    missing = [name for name in REQUIRED if not (raw / name).is_file()]
    require(not missing, "最终证据尚未齐全：" + ", ".join(missing))
    docs, parsed_hashes = {}, {}
    def tracked_read(name):
        data = (raw / name).read_bytes()
        value = json.loads(data.decode("utf-8"), parse_constant=invalid_constant)
        docs[name] = value
        parsed_hashes[name] = hashlib.sha256(data).hexdigest()
        return value
    for name in REQUIRED:
        tracked_read(name)
    p, a, post = docs["plan.json"], docs["analysis.json"], docs["post_job_result.json"]
    require(post["status"] == "complete", "post_job_result 尚未 complete")
    require(post["analysis_sha256"] == sha256(raw / "analysis.json"), "analysis SHA256 与 post_job_result 不一致")
    require(a["ok"] is True and a["training_completed"] is True, "analysis 尚未完成三奖励评估")
    require(a["job_status"] == "generation_training_evaluation_complete", "analysis job_status 尚未完成")
    require(docs["continuation_job_result_v2.json"]["status"] == a["job_status"], "正式运行状态与 analysis 不一致")
    for name in VERIFY_FILES:
        require(docs[name]["ok"] is True, f"独立核验未通过：{name}")
    for name, key in zip(VERIFY_FILES, ("independent_raw_sha256", "independent_training_sha256", "independent_metrics_sha256")):
        require(post[key] == parsed_hashes[name], f"独立核验 SHA256 与 post_job_result 不一致：{name}")
    require(tuple(p["evaluation"]["policies"]) == POLICIES and tuple(p["reward_arms"]) == ARMS,
            "策略或奖励顺序与本阶段封存计划不一致")
    lock = docs["analysis_protocol_lock.json"]
    require(lock["sealed_before_continuation_training"] is True and
            lock["primary_quantity_generation_mode"] == "greedy" and
            lock["primary_retrieval_guard"] == "repository_duplicate_token_positive mR on fullval210/1050",
            "主分析锁与已知协议不一致")
    require(p["training"]["attempts_per_arm"] == 32 and p["training"]["group_size"] == 8,
            "训练预算与本阶段封存计划不一致")
    require(p["analysis"]["bootstrap"] == 10000 and len(p["training_images"]) == 8,
            "bootstrap 次数或训练图像分母发生变化")
    prior_plan = docs["source_executed/rl/plan.json"]
    prior_checkpoint = docs["inputs/balanced_checkpoint_reference.json"]
    require(prior_plan["training"]["optimizer_updates"] == 32 and prior_plan["training"]["group_size"] == 4 and
            prior_plan["training"]["reward"] == "0.2 when inherited rollout quality passes, plus frozen downstream-AGSE A margin" and
            set(prior_plan["training_images"]) == set(p["training_images"]), "共同起点的既往训练方案发生变化")
    require(prior_checkpoint["ok"] is True and prior_checkpoint["optimizer_updates"] == 32 and
            prior_checkpoint["parameter_hash"] == p["policy"]["lora_hash"], "共同起点的既往实际更新或参数哈希未绑定")
    require(len(prior_plan["confirmation_train_holdout_images"]) == 55, "共同起点的既往评估图像分母发生变化")
    require(a["primary_holdout_strata"] == {"1": 3, "2": 34, "3": 12, "4": 3}, "主数量分层分母发生变化")
    require(a["new_test_scores"] == a["readout_optimizer_updates"] == 0, "新增 test 得分或读出更新计数发生变化")
    require(docs["prompt_selection_lock.json"]["selected_prompt"] == a["selected_prompt"], "提示锁与 analysis 不一致")
    require(docs["prompt_selection_lock.json"]["outcomes"] == a["pilot_prompt_outcomes"], "预试验结果与提示锁不一致")
    require(sum(x["records"] for x in a["pilot_prompt_outcomes"].values()) == a["pilot_records"],
            "预试验候选总数与 analysis 不一致")
    training_verify = docs["independent_training_verification.json"]
    for arm in ARMS:
        t = a["training"][arm]
        require(t["ok"] is True and t["attempts"] == 32 and t["samples"] == 256 and t["training_images"] == 8,
                f"训练覆盖发生变化：{arm}")
        require(t["initial_policy_hash"] == p["policy"]["lora_hash"], f"训练共同起点未绑定：{arm}")
        require(t["optimizer_updates"] + t["zero_variance_skips"] == 32, f"更新与跳过分母不一致：{arm}")
        for key in ("attempts", "samples", "optimizer_updates", "zero_variance_skips"):
            require(training_verify["arms"][arm][key] == t[key], f"独立训练核验与 analysis 不一致：{arm}/{key}")
        for signal in a["training_signals"][arm].values():
            require(signal["image_groups"] == 32 and signal["spearman_image_bootstrap95"] is None,
                    f"训练相关性协议发生变化：{arm}")
    require(a["Agent_optimizer_updates"] == sum(a["training"][arm]["optimizer_updates"] for arm in ARMS),
            "训练实际更新总数不一致")
    for policy in POLICIES:
        evaluation = a["evaluation"][policy]
        for mode, expected in (("greedy", 273), ("sampled", 110)):
            entry = evaluation[mode]
            require(entry["records"] == expected, f"评估记录数发生变化：{policy}/{mode}")
            require(close(entry["strict_rate"], entry["strict_completed"] / expected) and
                    close(entry["generated_result_rate"], entry["generated_results"] / expected),
                    f"评估比例与计数不一致：{policy}/{mode}")
            require(entry["heldout_train55"]["images"] == 55 and
                    entry["heldout_train55"]["records"] == (55 if mode == "greedy" else 110),
                    f"数量评估分母发生变化：{policy}/{mode}")
            require("B_hard_correct_macro1to4" in entry["heldout_train55"], f"主数量指标缺失：{policy}/{mode}")
        require(evaluation["greedy"]["all85"]["images"] == 85 and
                evaluation["greedy"]["all85"]["records"] == 85, f"all85 分母发生变化：{policy}")
        require(set(a["retrieval"][policy]) == set(PROTOCOLS), f"检索口径缺失：{policy}")
        for arm in ARMS:
            ranks = a["sampled_rank_patterns"][policy][arm]
            require(ranks["images"] == 55 and len(ranks["per_image"]) == 55,
                    f"随机两候选分母发生变化：{policy}/{arm}")
            require(sum(ranks[key] for key in ("agree", "reverse", "reward_tie_only", "utility_tie_only", "both_tie")) == 55,
                    f"随机两候选分类计数不一致：{policy}/{arm}")
            require(ranks["comparable"] == ranks["agree"] + ranks["reverse"], f"可比图分母不一致：{policy}/{arm}")
    primary = quantity_contrast(a, "quantity_hard", "proxy_rank")
    guard = retrieval_contrast(a, "quantity_hard", "baseline_continuation0", PROTOCOLS[0])
    require(a["adoption"]["primary"] == primary and a["adoption"]["retrieval_guard"] == guard,
            "采纳记录与主对照条目不一致")
    checks = {
        "primary_lower_gt0": primary["interval95"][0] > 0,
        "retrieval_lower_gt_minus0.02": guard["interval95"][0] > -.02,
        "sampled_protocol_all_policies": all(a["evaluation"][x]["sampled"]["strict_rate"] >= .90 and
                                             a["evaluation"][x]["sampled"]["generated_result_rate"] <= .02 for x in POLICIES),
        "same32actual_updates": all(a["training"][x]["optimizer_updates"] == 32 for x in ARMS),
    }
    require(a["adoption"]["checks"] == checks and a["adoption"]["passed"] == all(checks.values()), "采纳门槛重读不一致")
    require(docs["independent_metric_verification.json"]["all_adoption_gates_rebuilt"] == checks,
            "独立指标核验与采纳门槛不一致")
    require(len(a["quantity_contrasts"]) == 72 and len(a["retrieval_contrasts"]) == 18, "配对对照数量发生变化")
    for item in a["quantity_contrasts"]:
        require(item["interval95"][0] <= item["interval95"][1], "数量 CI 顺序错误")
        key = item["metric"] + "_macro1to4"
        expected = (a["evaluation"][item["policy"]][item["mode"]]["heldout_train55"][key] -
                    a["evaluation"][item["reference"]][item["mode"]]["heldout_train55"][key])
        require(close(item["delta_macro1to4"], expected), "数量点估计与策略汇总不一致")
    for item in a["retrieval_contrasts"]:
        require(item["interval95"][0] <= item["interval95"][1], "检索 CI 顺序错误")
        expected = (a["retrieval"][item["policy"]][item["protocol"]]["mR"] -
                    a["retrieval"][item["reference"]][item["protocol"]]["mR"])
        require(close(item["delta_mR"], expected), "检索点估计与策略汇总不一致")
    failure = docs["training/proxy_rank/failure.json"]
    require(failure["optimizer_updates"] == 0 and failure["attempts_completed"] == 0,
            "原始失败阶段零更新证据不一致")
    require(docs["cached_replay_addendum.json"]["observed_failure"]["Agent_optimizer_updates"] == 0,
            "缓存重放补充与原始零更新证据不一致")
    gate = docs["probability_gate_v2/result.json"]
    require(gate["ok"] is True and gate["permanent_Agent_updates"] == 0 and gate["restored_exact"] is True and
            gate["initial_policy_hash"] == gate["restored_policy_hash"] == p["policy"]["lora_hash"],
            "缓存概率重放恢复证据未通过")
    rounds = docs["probability_gate_v2/rounds.json"]
    require(gate["checked_samples"] == 10 and gate["checked_rounds"] == len(rounds) == 20,
            "缓存概率重放覆盖发生变化")
    require(all(x["max_token_logprob_gap"] <= p["probability_gate"]["max_token_gap"] and
                x["mean_absolute_token_logprob_gap"] <= p["probability_gate"]["mean_absolute_gap"] and
                x["last_hidden_cosine"] >= .999 for x in rounds), "缓存概率重放门槛未通过")
    qualitative_lock = docs["qualitative_case_lock.json"]
    require(qualitative_lock["mode"] == "greedy" and qualitative_lock["policies"] == list(POLICIES) and
            qualitative_lock["additional_generations"] == 0, "定性样例锁发生变化")
    require([case["actual_count"] for case in qualitative_lock["images"]] == [1, 2, 3, 4] and
            all(case["split"] == "train" and case["image"] not in p["training_images"] for case in qualitative_lock["images"]),
            "定性样例留出与分层设置发生变化")
    examples = []
    for policy in POLICIES:
        records = tracked_read(f"evaluation/{policy}/records.json")
        require(len(records) == 383, f"定性样例来源评估记录数发生变化：{policy}")
        require({row["image"] for row in records if row["mode"] == "sampled"} == set(prior_plan["confirmation_train_holdout_images"]),
                f"本轮与既往 55 图评估集合未对齐：{policy}")
        for case in qualitative_lock["images"]:
            row = unique(records, image=case["image"], mode="greedy")
            trace_name = str(Path(row["trace_file"]))
            trace_path = (raw / trace_name).resolve()
            require(raw in trace_path.parents, f"定性轨迹路径超出原始目录：{trace_name}")
            trace = tracked_read(trace_name)
            require(parsed_hashes[trace_name] == row["trace_sha256"], f"定性轨迹哈希不一致：{trace_name}")
            require(bool(trace["rounds"]), f"定性轨迹轮次为空：{trace_name}")
            examples.append({"policy": policy, **case,
                             "final_generated_text": trace["rounds"][-1]["generated_text"],
                             "trace_file": trace_name, "trace_sha256": row["trace_sha256"],
                             **{key: row[key] for key in ("strict_completed_consumed_tool", "feedback_consumed", "tools_used",
                                                         "generated_result_tag", "stop_reason", "B_pair_correct", "B_hard_correct", "B_hard_margin")}})
    docs["_qualitative_examples"] = examples
    for name, expected in parsed_hashes.items():
        require(sha256(raw / name) == expected, f"读取核验期间输入发生变化：{name}")
    return docs, parsed_hashes


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    require(bool(rows), f"CSV 记录为空：{path.name}")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_tables(output, analysis):
    folder = output / "tables"
    folder.mkdir()
    write_csv(folder / "training.csv", [{"policy": arm, **{key: analysis["training"][arm][key]
              for key in ("attempts", "samples", "optimizer_updates", "zero_variance_skips", "initial_policy_hash", "final_policy_hash", "checkpoint")}} for arm in ARMS])
    primary, quantities, protocols, retrieval, ranks, rank_images, signals = [], [], [], [], [], [], []
    for policy in POLICIES:
        for mode in ("greedy", "sampled"):
            entry = analysis["evaluation"][policy][mode]
            primary.append({"policy": policy, "mode": mode, "images_count1to4": 52,
                            "B_hard_correct_macro1to4": entry["heldout_train55"]["B_hard_correct_macro1to4"]})
            protocols.append({"policy": policy, "mode": mode, **{key: entry[key] for key in
                              ("records", "strict_completed", "strict_rate", "generated_results", "generated_result_rate", "consumed_feedback")}})
            for cohort, values in entry.items():
                if isinstance(values, dict):
                    quantities.append({"policy": policy, "mode": mode, "cohort": cohort,
                                       "metrics_json": json.dumps(values, ensure_ascii=False, sort_keys=True)})
        for protocol in PROTOCOLS:
            retrieval.append({"policy": policy, "protocol": protocol, **analysis["retrieval"][policy][protocol]})
        for arm in ARMS:
            values = analysis["sampled_rank_patterns"][policy][arm]
            ranks.append({"policy": policy, "reward": arm, **{key: value for key, value in values.items() if key != "per_image"}})
            rank_images.extend({"policy": policy, "reward": arm, **row} for row in values["per_image"])
    for trained_arm in ARMS:
        for reward, values in analysis["training_signals"][trained_arm].items():
            signals.append({"trained_arm": trained_arm, "reward": reward,
                            **{key: value for key, value in values.items() if key != "groups"}})
    for filename, rows in (("quantity_primary.csv", primary), ("quantity_cohorts.csv", quantities),
                           ("protocol.csv", protocols), ("retrieval.csv", retrieval),
                           ("sampled_two_candidate_patterns.csv", ranks), ("training_descriptive_correlations.csv", signals)):
        write_csv(folder / filename, rows)
    write_csv(folder / "sampled_two_candidate_per_image.csv", rank_images)
    for name in ("quantity_contrasts.csv", "retrieval_contrasts.csv"):
        source = analysis[name.removesuffix(".csv")]
        rows = [{**{k: v for k, v in row.items() if k != "interval95"},
                 "ci95_lower": row["interval95"][0], "ci95_upper": row["interval95"][1]} for row in source]
        write_csv(folder / name, rows)


def make_figures(output, analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    needed = set(ord(c) for c in "数量困难数词代理排名奖励初始策略本阶段起点步宏平均准确率百分比百分点完整验证检索口径重复正例严格索引平局训练采样机会实际更新零方差跳过随机完成率自行生成结果标签")
    candidates = []
    for name in ("Arial Unicode MS", "PingFang SC", "PingFang HK", "STHeiti", "Songti SC", "Heiti TC", "Noto Sans CJK SC", "SimHei"):
        candidates.extend(f for f in font_manager.fontManager.ttflist if f.name == name)
    chosen = next((f for f in candidates if needed <= set(FT2Font(f.fname).get_charmap())), None)
    require(chosen is not None, "本机缺少覆盖图表中文的字体，请安装中文字体后重试")
    matplotlib.rcParams.update({"font.family": chosen.name, "axes.unicode_minus": False,
                                "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                                "pdf.fonttype": 42, "svg.fonttype": "path", "savefig.facecolor": "white"})
    folder = output / "figures"
    folder.mkdir()
    def save(fig, stem):
        for extension in ("png", "pdf", "svg"):
            fig.savefig(folder / f"{stem}.{extension}", dpi=220)
        plt.close(fig)

    def interval_plot(ax, rows, value_key, labels, threshold=None):
        for index, row in enumerate(rows):
            point, lo, hi = (100 * row[value_key], *(100 * x for x in row["interval95"]))
            # Draw endpoints directly; percentile intervals may exclude the point estimate.
            ax.hlines(index, lo, hi, color=COLORS[-1], lw=2)
            ax.plot([lo, hi], [index, index], "|", color=COLORS[-1], ms=8)
            ax.plot(point, index, "o", color=COLORS[-1], ms=5)
        ax.set_yticks(range(len(rows)), labels)
        ax.axvline(0, color="#7F7F7F", lw=1)
        if threshold is not None:
            ax.axvline(threshold, color="#B94242", lw=1, linestyle="--", label="警戒下限 −2 百分点")
            ax.legend(loc="best", frameon=False, fontsize=8)
        ax.set_xlabel("差值（百分点），图像配对 bootstrap 95% CI")
        ax.grid(axis="x", alpha=.18)
        ax.invert_yaxis()

    values = [100 * analysis["evaluation"][x]["greedy"]["heldout_train55"]["B_hard_correct_macro1to4"] for x in POLICIES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained", gridspec_kw={"width_ratios": (1, 1.25)})
    axes[0].bar(range(4), values, color=COLORS, width=.6)
    axes[0].set_xticks(range(4), ["balanced32\n本阶段 0 步", "代理排名\n奖励", "数量文本差\n奖励", "困难数词\n奖励"])
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("B 困难排序宏平均准确率（%）")
    axes[0].set_title("主数量指标：贪心生成，52 图")
    for index, value in enumerate(values):
        axes[0].text(index, value + 1.5, f"{value:.2f}", ha="center", fontsize=9)
    contrasts = [quantity_contrast(analysis, "quantity_hard", x) for x in ("proxy_rank", "quantity_pair", "baseline_continuation0")]
    interval_plot(axes[1], contrasts, "delta_macro1to4", ["困难数词 − 代理排名", "困难数词 − 数量文本差", "困难数词 − 阶段起点"])
    axes[1].set_title("主对照与补充对照")
    save(fig, "01_quantity_primary")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained", gridspec_kw={"width_ratios": (1, 1.25)})
    for index, policy in enumerate(POLICIES):
        axes[0].bar([x + (index - 1.5) * .18 for x in range(3)],
                    [100 * analysis["retrieval"][policy][proto]["mR"] for proto in PROTOCOLS],
                    width=.18, color=COLORS[index], label=LABELS[policy])
    axes[0].set_xticks(range(3), ["重复 token\n正例", "严格索引", "严格平局\n平均"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("mR（%）")
    axes[0].set_title("完整验证集：210 图 / 1050 句")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    contrasts = [retrieval_contrast(analysis, "quantity_hard", "baseline_continuation0", x) for x in PROTOCOLS]
    interval_plot(axes[1], contrasts, "delta_mR", [LABELS[x] for x in PROTOCOLS], threshold=-2)
    axes[1].set_title("困难数词 − 阶段起点")
    save(fig, "02_retrieval_protocols")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
    actual = [analysis["training"][x]["optimizer_updates"] for x in ARMS]
    skip = [analysis["training"][x]["zero_variance_skips"] for x in ARMS]
    axes[0].bar(range(3), actual, color=COLORS[1:], label="实际更新")
    axes[0].bar(range(3), skip, bottom=actual, color="#D9DFE4", label="零方差跳过")
    axes[0].set_xticks(range(3), ["代理排名", "数量文本差", "困难数词"])
    axes[0].set_ylim(0, 36)
    axes[0].set_ylabel("次数")
    axes[0].set_title("每组 32 次采样机会")
    axes[0].legend(frameon=False, fontsize=8)
    for index, value in enumerate(actual):
        axes[0].text(index, 33, str(value), ha="center")
    for ax, field, limit, title in ((axes[1], "strict_rate", 90, "随机严格完成率"),
                                   (axes[2], "generated_result_rate", 2, "随机自行生成结果标签比例")):
        rates = [100 * analysis["evaluation"][x]["sampled"][field] for x in POLICIES]
        ax.bar(range(4), rates, color=COLORS)
        ax.set_xticks(range(4), ["0 步", "代理排名", "文本差", "困难数词"])
        ax.axhline(limit, color="#B94242", ls="--", lw=1)
        ax.set_title(title + "\n各 55 图 × 2 候选")
        ax.set_ylabel("%")
        ax.set_ylim(0, 108 if field == "strict_rate" else max(5, max(rates) * 1.25))
        for index, value in enumerate(rates):
            ax.text(index, value + (.8 if field == "strict_rate" else .1), f"{value:.2f}", ha="center", fontsize=8)
    save(fig, "03_updates_and_protocol")
    return {"matplotlib": matplotlib.__version__, "font_name": chosen.name, "font_path": str(Path(chosen.fname).resolve())}


def build_report(raw, output, docs, graphics):
    p, a = docs["plan.json"], docs["analysis.json"]
    primary = a["adoption"]["primary"]
    hard_pair = quantity_contrast(a, "quantity_hard", "quantity_pair")
    guard = a["adoption"]["retrieval_guard"]
    adopted = f"{sum(a['adoption']['checks'].values())}/{len(a['adoption']['checks'])} 项达到"
    lines = ["# RFPO 随机采样核验与三奖励短训对照：最终交付", "",
             f"采纳门槛判定：**{adopted}**。主数量对照（困难数词 − 代理排名）为 **{pp_interval(primary, 'delta_macro1to4')} 百分点**；"
             f"困难数词 − 数量文本差为 **{pp_interval(hard_pair, 'delta_macro1to4')} 百分点**。"
             f"主检索警戒（困难数词 − 阶段起点）为 **{pp_interval(guard, 'delta_mR')} 百分点**。方括号为配对图像 bootstrap 95% CI。", "",
             f"本交付直接读取 {link('analysis.json', raw / 'analysis.json')} 及三份独立核验结果。原始记录目录：`{raw}`。",
             "", "## 1. 运行与分析协议", "",
             "共同起点 balanced32 已在相同 8 幅图像上完成过 32 次数量文本差奖励更新，历史每组为 4 个候选。"
             "源计划的训练奖励为 `" + docs["source_executed/rl/plan.json"]["training"]["reward"] + "`；"
             "检查点引用记录的实际更新次数为 32，参数哈希与本轮起点一致。来源：" +
             link("历史源计划", raw / "source_executed/rl/plan.json") + "、" +
             link("balanced32 检查点引用", raw / "inputs/balanced_checkpoint_reference.json") + "。", "",
             "本轮在已进行数量优化的共同起点上继续比较三种奖励，每组候选数调整为 8；"
             "`baseline_continuation0` 表示本阶段新增 0 次更新。以下奖励组的更新计数均指本阶段新增更新。", "",
             f"三奖励组均从同一 balanced32 参数 `{p['policy']['lora_hash']}` 开始；每组 32 次采样机会、每次同图 8 候选，"
             "共 256 条正式训练候选，三组共 768 条。8 幅训练图按封存顺序循环四次，共用采样种子安排；每组使用新建 AdamW。"
             "奖励标准差 ≤1e−8 的组保留记录并跳过优化器。固定采用第 32 次机会结束后的检查点。", "",
             f"提示预试验共 {a['pilot_records']} 条，按封存优先级选定 `{a['selected_prompt']}`；选择依据为训练 8 图的协议表现。"
             "三种奖励与原有采纳阈值均按预先封存方案执行。", "",
             table(["提示", "严格完成 / 候选", "读入工具反馈 / 候选", "自行生成结果标签 / 候选", "预试验门槛"],
                   [[name, f"{row['strict_completed']}/{row['records']}", f"{row['consumed_tool']}/{row['records']}",
                     f"{row['generated_result']}/{row['records']}", "通过" if row["passed"] else "未通过"]
                    for name, row in a["pilot_prompt_outcomes"].items()]), "",
             table(["奖励组", "封存奖励公式"], [[LABELS[arm], formula] for arm, formula in p["reward_arms"].items()]), "",
             "三种训练奖励均包含严格完成并读入工具反馈时的 0.2 格式分。数量奖励按 A 措辞计算，主数量评估使用 B 措辞。"
             "代理排名在训练图库内比较目标图像的得分名次。", "",
             "工具反馈采用模型发出 scene_classify 请求后，由外部回放历史真实分类结果的方式。生成输入为对应图像与通用查询；"
             "每轮实际生成状态与冻结评分器接收状态保持绑定。", "",
             f"冻结读出为 `{p['readout']['arm']}`：`{p['readout']['checkpoint']['path']}`，SHA256 `{p['readout']['checkpoint']['sha256']}`，"
             f"记录 global_step={p['readout']['checkpoint']['global_step']}。"
             "代理图库覆盖训练 Arrow 的 1680 幅图像；三种奖励共享该读出与特征处理。", "",
             "封存计划对读出的说明：`" + p["readout"]["context"] + "`。", "",
             "主数量指标采用贪心生成，在本轮 8 幅策略训练图之外的 55 幅训练图中，取真数量 1–4 的 52 幅图计算 B 措辞困难排序宏平均。"
             "四层图像分母依次为 3、34、12、3。B 措辞与这批图像均已有既往诊断记录，holdout 描述指本轮策略短训的图像留出。", "",
             "每策略评估记录包含 273 条贪心轨迹与 55 图 ×2 条随机轨迹，各 383 条、合计 1532 条。主检索警戒采用重复 token 正例口径，"
             "完整验证集为 210 图 /1050 句；严格索引及严格平局平均逐项报告。", "",
             f"数量 CI 以同图配对、按真数量 1–4 分层抽样；检索 CI 以验证图像及其 5 句描述为配对簇，图库固定。"
             f"bootstrap 为 {p['analysis']['bootstrap']} 次，随机种子 {p['analysis']['seed']}。新增 test 得分为 {a['new_test_scores']}；"
             f"读出优化器更新为 {a['readout_optimizer_updates']}。", "",
             "## 2. 正式训练与协议表现", "",
             table(["奖励组", "采样机会", "候选", "实际更新", "零方差跳过"],
                   [[LABELS[x], a["training"][x]["attempts"], a["training"][x]["samples"],
                     a["training"][x]["optimizer_updates"], a["training"][x]["zero_variance_skips"]] for x in ARMS]), ""]
    if a["adoption"]["checks"]["same32actual_updates"]:
        lines += ["三组正式训练均完成 32 次实际更新，达到预设的更新次数匹配门槛。", ""]
    else:
        lines += ["实际更新次数差异来自封存的零方差跳过规则；本次结果按共同采样机会预算解释，奖励差异与有效更新次数共同影响最终策略。", ""]
    lines += [table(["策略", "模式", "严格完成", "完成率", "自行生成结果标签", "标签比例", "读入反馈"],
                    [[LABELS[x], "贪心" if mode == "greedy" else "随机",
                      f"{a['evaluation'][x][mode]['strict_completed']}/{a['evaluation'][x][mode]['records']}",
                      pct(a["evaluation"][x][mode]["strict_rate"]), a["evaluation"][x][mode]["generated_results"],
                      pct(a["evaluation"][x][mode]["generated_result_rate"]), a["evaluation"][x][mode]["consumed_feedback"]]
                     for x in POLICIES for mode in ("greedy", "sampled")]), "",
              f"![真实更新次数与随机协议表现]({output / 'figures/03_updates_and_protocol.png'})", "",
              "## 3. 数量主结果与配对差值", "",
              table(["策略", "贪心：B 困难排序宏平均", "随机：B 困难排序宏平均"],
                    [[LABELS[x], *(pct(a["evaluation"][x][mode]["heldout_train55"]["B_hard_correct_macro1to4"])
                                  for mode in ("greedy", "sampled"))] for x in POLICIES]), "",
              table(["贪心配对对照", "宏平均差值 [95% CI]，百分点"],
                    [[f"{LABELS[x['policy']]} − {LABELS[x['reference']]}", pp_interval(x, "delta_macro1to4")]
                     for x in a["quantity_contrasts"] if x["mode"] == "greedy" and x["metric"] == "B_hard_correct"]), "",
              f"![主数量指标与配对差值]({output / 'figures/01_quantity_primary.png'})", "",
              "全部 72 项数量配对对照及各队列的 A/B 准确率、间隔保存在表格文件。以下列出贪心 all85 的 A/B 均值，"
              "该汇总覆盖 8 张训练图、55 张本轮留出训练图及 22 张验证数量图。", "",
              table(["策略", "措辞", "成对准确率", "困难准确率", "成对间隔", "困难间隔"],
                    [[LABELS[x], word, pct(v[word + "_pair_correct_mean"]), pct(v[word + "_hard_correct_mean"]),
                      number(v[word + "_pair_margin_mean"]), number(v[word + "_hard_margin_mean"])]
                     for x in POLICIES for v in [a["evaluation"][x]["greedy"]["all85"]] for word in ("A", "B")]), "",
              "## 4. 完整验证检索与主警戒", "",
              table(["策略", "口径", "I2T R@1", "I2T R@5", "I2T R@10", "T2I R@1", "T2I R@5", "T2I R@10", "mR"],
                    [[LABELS[x], LABELS[proto], *(pct(a["retrieval"][x][proto][key]) for key in
                       ("I2T_R1", "I2T_R5", "I2T_R10", "T2I_R1", "T2I_R5", "T2I_R10", "mR"))]
                     for proto in PROTOCOLS for x in POLICIES]), "",
              table(["口径", "困难数词 − 阶段起点 mR [95% CI]，百分点"],
                    [[LABELS[proto], pp_interval(retrieval_contrast(a, "quantity_hard", "baseline_continuation0", proto), "delta_mR")]
                     for proto in PROTOCOLS]), "",
              f"![完整验证检索与差值]({output / 'figures/02_retrieval_protocols.png'})", "",
              "全部 18 项检索配对对照均保存在 retrieval_contrasts.csv；主采纳门槛使用重复 token 正例口径。", "",
              "## 5. 随机两候选与训练描述统计", "",
              "下表在每策略的 55 幅图上比较两条随机候选的奖励差与 B 困难间隔差。可比图为奖励与效用均有差异的图，"
              "同向率使用该分母；其余图逐项保留平局类型。", "",
              table(["策略", "奖励", "图数", "可比", "同向", "反向", "仅奖励平局", "仅效用平局", "双平局", "同向率"],
                    [[LABELS[x], LABELS[arm], v["images"], v["comparable"], v["agree"], v["reverse"],
                      v["reward_tie_only"], v["utility_tie_only"], v["both_tie"],
                      "未定义" if v["agree_fraction_comparable"] is None else pct(v["agree_fraction_comparable"])]
                     for x in POLICIES for arm in ARMS for v in [a["sampled_rank_patterns"][x][arm]]]), "",
              "正式训练按同一次采样机会内的 8 候选计算奖励与 B 困难间隔的 Spearman 相关性，再汇总有定义的机会。"
              "这些训练机会随策略更新顺序产生，下表用于描述，区间字段保留为空。", "",
              table(["训练奖励组", "观察奖励", "机会数", "奖励有信息机会", "相关性有定义机会", "组内相关均值"],
                    [[LABELS[x], LABELS[arm], v["image_groups"], v["informative_groups"],
                      v["informative_correlation_groups"], number(v["mean_within_image_spearman"])]
                     for x in ARMS for arm, v in a["training_signals"][x].items()]), "",
              "## 6. 采纳门槛与结论范围", "",
              table(["封存门槛", "本次判定"], [[GATE_LABELS[key], "达到" if value else "未达到"]
                                            for key, value in a["adoption"]["checks"].items()]), ""]
    if a["adoption"]["passed"]:
        lines += ["本轮全部封存门槛达到，结论对应本次 UCM、单个继续训练种子、8 图短训与固定读出的条件。", ""]
    else:
        lines += ["本轮已完成预定比较；按封存门槛记录各项结果，保留最终检查点，并结束该预算。后续实验另行设计。", ""]
    lines += ["当前证据覆盖 UCM 单个继续训练种子，以及已做 32 次数量奖励更新的共同起点。冻结读出由验证集选择；"
              "B 措辞与相同的 55 张本轮图像留出集已有既往评估记录。"
              "主数量指标聚焦真数量 1–4，两个稀疏分层各含 3 图。完整验证检索的区间反映固定图库中的图像配对波动。"
              "原三数据集 RFPO 实验继续作为独立实验项记录。", "",
              "## 7. 首次失败与缓存重放修复证据", ""]
    observed = docs["cached_replay_addendum.json"]["observed_failure"]
    gate = docs["probability_gate_v2/result.json"]
    rounds = docs["probability_gate_v2/rounds.json"]
    lines += [f"原始整段概率重放在首组样本 {observed['sample']}、轮索引 {observed['round0']} 的 {observed['tokens']} 词回答上，"
              f"最大逐词 logprob 差为 {observed['fullsequence_max_logprob_gap']:.10f}，平均绝对差为 "
              f"{observed['fullsequence_mean_abs_gap']:.10f}，触发原定门槛；原始失败记录显示实际更新 "
              f"{docs['training/proxy_rank/failure.json']['optimizer_updates']} 次。原失败组的 8 条诊断候选另行记账。", "",
              f"修复采用原生 prefill 加逐词 KV 缓存的自动求导重放。扩大复核覆盖 {gate['checked_samples']} 条候选、"
              f"{gate['checked_rounds']} 轮，最大逐词差为 {max(x['max_token_logprob_gap'] for x in rounds):.10g}，"
              f"最大轮平均绝对差为 {max(x['mean_absolute_token_logprob_gap'] for x in rounds):.10g}，"
              f"末状态最小余弦为 {min(x['last_hidden_cosine'] for x in rounds):.10g}。"
              f"临时优化器更新 {gate['temporary_updates']} 次后，{gate['trainable_tensors']} 个可训练张量精确恢复至共同起点；"
              f"永久更新 {gate['permanent_Agent_updates']} 次。正式训练分别在 training_v2 下重新起步。", "",
              "保留证据：" + "、".join(link(name, raw / name) for name in
                  ("cached_replay_addendum.json", "continuation_job_result.json", "training/proxy_rank/failure.json",
                   "probability_gate_v2/result.json", "probability_gate_v2/rounds.json")) + "。", "",
              "## 8. 独立核验与交付索引", "",
              table(["核验", "结果", "重建范围"], [
                  [VERIFY_FILES[0], "通过", f"原始记录 {docs[VERIFY_FILES[0]]['records_rebuilt']} 条；生成轮 {docs[VERIFY_FILES[0]]['rounds_rebuilt']} 轮；状态、奖励、逐词证据"],
                  [VERIFY_FILES[1], "通过", f"三组奖励、advantage、loss、哈希链及检查点优化器；概率重放 {docs[VERIFY_FILES[1]]['probability_rounds_rebuilt']} 轮"],
                  [VERIFY_FILES[2], "通过", f"原始指标 {docs[VERIFY_FILES[2]]['raw_metric_values_rebuilt']} 项；数量区间 {docs[VERIFY_FILES[2]]['quantity_intervals_rebuilt']} 项；检索区间 {docs[VERIFY_FILES[2]]['retrieval_intervals_rebuilt']} 项；全部采纳门槛"],
              ]), "",
              "post_job_result 绑定 analysis 与三份独立核验的 SHA256。本交付的 source_manifest.json 进一步记录所有已读取证据文件的字节数及 SHA256；"
              "delivery_manifest.json 记录新生成文件的字节数及 SHA256。", "",
              "- " + link("结构化摘要", output / "final_summary.json"),
              "- " + link("输入证据清单", output / "source_manifest.json"),
              "- " + link("输出清单", output / "delivery_manifest.json"),
              "- " + link("全部数量配对对照", output / "tables/quantity_contrasts.csv"),
              "- " + link("全部检索配对对照", output / "tables/retrieval_contrasts.csv"),
              "- " + link("全部队列指标", output / "tables/quantity_cohorts.csv"),
              "- " + link("预先选定四图的原始回答对照", output / "定性样例原始回答.md"), "",
              f"图表字体：{graphics['font_name']}，`{graphics['font_path']}`；Matplotlib {graphics['matplotlib']}。"
              "PNG、PDF、SVG 三种格式均已生成。交付生成器负责读取与呈现已核验统计。", ""]
    with (output / "最终对照报告.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def export_qualitative(raw, output, docs):
    rows = docs["_qualitative_examples"]
    write_json(output / "qualitative_examples.json", rows)
    write_csv(output / "tables/qualitative_examples.csv", rows)
    lines = ["# 预先选定四图：四策略贪心原始回答对照", "",
             "按封存输入顺序，分别取真数量 1–4 的首张人工标注、本轮短训留出训练图；样例选择早于最终策略结果。"
             "直接读取已有贪心轨迹，共 4 图 ×4 策略；新增生成为 0。下列回答与标签来自保存轨迹和评分记录。", "",
             "选择依据：" + link("qualitative_case_lock.json", raw / "qualitative_case_lock.json") + "。", ""]
    for case in docs["qualitative_case_lock.json"]["images"]:
        lines += [f"## {case['image']}：{case['object']}，人工数量 {case['actual_count']}", "",
                  f"数据位置：`{case['split']}`，Arrow 行号 `{case['row']}`。", ""]
        for policy in POLICIES:
            row = unique(rows, image=case["image"], policy=policy)
            text = row["final_generated_text"]
            fence = "```"
            while fence in text:
                fence += "`"
            lines += [f"### {LABELS[policy]}", "",
                      table(["严格完成并读入工具", "反馈读入", "工具次数", "自行生成结果标签", "停止原因", "B 成对正确", "B 困难正确", "B 困难间隔"],
                            [[row["strict_completed_consumed_tool"], row["feedback_consumed"], row["tools_used"], row["generated_result_tag"],
                              row["stop_reason"], row["B_pair_correct"], row["B_hard_correct"], number(row["B_hard_margin"])]]), "",
                      fence + "text", text, fence, "",
                      "来源：" + link(row["trace_file"], raw / row["trace_file"]) + f"；SHA256 `{row['trace_sha256']}`。", ""]
    with (output / "定性样例原始回答.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, required=True, help="完整最终原始记录目录（只读）")
    parser.add_argument("--output-root", type=Path, required=True, help="原始记录树之外的新交付目录，已存在则停止")
    args = parser.parse_args()
    requested_output = args.output_root.expanduser().absolute()
    require(not requested_output.exists() and not requested_output.is_symlink(), f"输出路径已存在，已停止：{requested_output}")
    raw, output = args.raw_root.expanduser().resolve(), requested_output.resolve()
    require(raw.is_dir(), f"原始记录目录缺失：{raw}")
    require(not output.exists(), f"输出目录已存在，已停止：{output}")
    require(output != raw and raw not in output.parents, "请将交付目录设在原始记录目录树之外")
    docs, parsed_hashes = load_and_validate(raw)
    a = docs["analysis.json"]
    source_paths = [raw / name for name in parsed_hashes]
    for name in ("analysis_addendum_v2.json", "阶段运行说明_20261117.md", "code/analyze_comparison_v3.py"):
        if (raw / name).is_file():
            source_paths.append(raw / name)
    # The producing script is also bound, even when executed outside the raw tree.
    script = Path(__file__).resolve()
    if script not in source_paths:
        source_paths.append(script)
    sources = {"raw_root": str(raw), "files": [
        {"relative_path": str(path.relative_to(raw)) if path.is_relative_to(raw) else "delivery_generator:" + path.name,
         "absolute_path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in source_paths]}
    for item in sources["files"]:
        if item["relative_path"] in parsed_hashes:
            require(item["sha256"] == parsed_hashes[item["relative_path"]], "清单与读取内容哈希发生变化")
    output.mkdir(parents=True, exist_ok=False)
    try:
        graphics = make_figures(output, docs["analysis.json"])
        export_tables(output, docs["analysis.json"])
        write_json(output / "source_manifest.json", sources)
        a = docs["analysis.json"]
        summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "raw_root": str(raw),
                   "output_root": str(output), "source_analysis_sha256": sha256(raw / "analysis.json"),
                   "scope": a["scope"], "selected_prompt": a["selected_prompt"], "pilot_records": a["pilot_records"],
                   "common_start_history": {"optimizer_updates": docs["inputs/balanced_checkpoint_reference.json"]["optimizer_updates"],
                                            "group_size": docs["source_executed/rl/plan.json"]["training"]["group_size"],
                                            "reward": docs["source_executed/rl/plan.json"]["training"]["reward"],
                                            "same_training_images": True, "baseline_new_updates_this_stage": 0,
                                            "previously_evaluated_holdout_images": docs["source_executed/rl/plan.json"]["confirmation_train_holdout_images"]},
                   "training": a["training"], "primary_holdout_strata": a["primary_holdout_strata"],
                   "evaluation": a["evaluation"], "retrieval": a["retrieval"],
                   "quantity_contrasts": a["quantity_contrasts"], "retrieval_contrasts": a["retrieval_contrasts"],
                   "sampled_rank_patterns": a["sampled_rank_patterns"], "pilot_signals": a["pilot_signals"],
                   "training_signals": a["training_signals"], "adoption": a["adoption"],
                   "new_test_scores": a["new_test_scores"], "readout_optimizer_updates": a["readout_optimizer_updates"],
                   "Agent_optimizer_updates_this_stage": a["Agent_optimizer_updates"],
                   "independent_verification": {name: docs[name] for name in VERIFY_FILES},
                   "failed_original_stage": docs["training/proxy_rank/failure.json"],
                   "cached_replay_gate": docs["probability_gate_v2/result.json"],
                   "limitations": ["UCM 单个继续训练种子，8 图策略短训，起点已有 32 次数量奖励更新", "冻结读出由验证集选择",
                                   "B 措辞与 55 张本轮图像留出集存在既往诊断记录",
                                   "训练机会相关性仅作描述，区间字段为空", "工具请求后回放历史真实分类结果"],
                   "runtime": {"python": platform.python_version(), **graphics}}
        write_json(output / "final_summary.json", summary)
        export_qualitative(raw, output, docs)
        build_report(raw, output, docs, graphics)
        # Recheck every source after rendering to detect concurrent input changes.
        for item in sources["files"]:
            path = Path(item["absolute_path"])
            require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"],
                    f"交付生成期间输入发生变化：{path}")
        manifest = {"status": "complete", "raw_root": str(raw), "output_root": str(output),
                    "visual_review": "待人工检查导出 PNG 的中文、布局与坐标轴",
                    "files": [{"relative_path": str(path.relative_to(output)), "absolute_path": str(path),
                               "bytes": path.stat().st_size, "sha256": sha256(path)}
                              for path in sorted(output.rglob("*")) if path.is_file()]}
        write_json(output / "delivery_manifest.json", manifest)
    except Exception as error:
        write_json(output / "build_failure.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})
        raise
    print(json.dumps({"status": "complete", "output_root": str(output),
                      "report": str(output / "最终对照报告.md"), "adoption_passed": a["adoption"]["passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (DeliveryError, KeyError, OSError, TypeError, ImportError, json.JSONDecodeError) as error:
        print(f"交付生成停止：{error}", file=sys.stderr)
        raise SystemExit(2)
