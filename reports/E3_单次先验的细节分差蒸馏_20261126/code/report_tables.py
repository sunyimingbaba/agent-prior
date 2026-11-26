from local import ROOT, read


NAMES = {"retained": "保留 R2", "parent": "原筛图联合 RFPO", "ordinary": "只教普通分差",
         "guard": "普通分差加原 R2 细节下限", "teacher": "逐项取大，普通加细节",
         "reachable_ordinary": "单条候选，只教普通分差", "reachable_teacher": "单条候选，普通加细节",
         "coldR0": "训练前 R0", "warmordinary": "R2 起点，原普通奖励", "strict_coldordinary": "R0 起点，普通奖励",
         "strict_coldstrict": "R0 起点，加严联合奖励", "strict_warmstrict": "R2 起点，加严联合奖励"}
PRIMARY = [("official", "mR"), ("primary", "CDE"), ("shared", "CDE"), ("shared", "mrr")]
HEADERS = ["UCM 210 图 mR（%）", "126 图细节（%）", "106 图同一首句细节（%）", "106 图其他配文 MRR"]
HELD = [("hold_scene", "CDE"), ("hold_scene", "mrr"), ("hold_first", "CDE"), ("hold_first", "mrr")]


def table(headers, rows):
    return "\n".join("| " + " | ".join(row) + " |" for row in [headers, ["---"] * len(headers), *rows])


def number(x, metric, signed=False):
    precision = 4 if metric == "mrr" else 2
    factor = 1 if metric == "mrr" else 100
    return f"{factor * x:{'+' if signed else ''}.{precision}f}"


def setting(source, entries, arms):
    return [[NAMES[a], *[number(source[tag]["levels"][a][k]["mean"], k) for tag, k in entries]] for a in arms]


def delta(source, key, ref, tag, metric):
    data = source[tag][key][ref][metric]
    def fmt(value): return number(value, metric, True)
    return f"{fmt(data['mean'])} [{fmt(data['interval'][0])}, {fmt(data['interval'][1])}]"


def main():
    reach = read(ROOT / "reachable/results/score.json"); strict = read(ROOT / "strict_rfpo/results/score.json")
    base = read(ROOT / "results/score.json")
    txt = ["# 单次先验实测数据", "两份固定随机种子等权。细节为检索器把真实细节句排在相反句之前的比例。"
           "mR 是图检文与文检图 R@1、R@5、R@10 六项的均值。MRR 是正确配文排名的倒数均值。"
           "标为“差”的百分数以百分点计。方括号为 20,000 次配对重采样的 95% 区间。",
           "## 两个正式终点及同一次首句输出"]
    rows = setting(reach, PRIMARY, list(reach["official"]["levels"]))
    rows += setting(strict, PRIMARY, [a for a in strict["official"]["levels"] if a not in reach["official"]["levels"]])
    txt.append(table(["权重", *HEADERS], rows))
    txt.append("## 事先指定的配对比较")
    rows = []
    for source, group, key, target, refs in (
        (base, "逐项取大", "teacher_minus", "teacher", ("retained", "parent", "ordinary", "guard")),
        (reach, "单条候选", "teacher_minus", "reachable_teacher", ("retained", "parent", "reachable_ordinary", "teacher")),
        (strict, "加严 RFPO", "teacher_minus", "strict_coldstrict", ("coldR0", "strict_coldordinary", "retained")),
        (strict, "加严 RFPO", "warm_minus", "strict_warmstrict", ("parent", "warmordinary", "retained"))):
        for ref in refs:
            rows.append([group, NAMES[target] + " 减 " + NAMES[ref],
                         *[delta(source, key, ref, t, m) for t, m in PRIMARY]])
    txt.append(table(["实验", "比较", "210 图 mR 差", "126 图细节差", "106 图同一输出细节差", "106 图其他配文 MRR 差"], rows))
    txt.append("## 本轮留出的 64 张训练划分图片")
    rows = setting(reach, HELD, list(reach["hold_first"]["levels"]))
    rows += setting(strict, HELD, [a for a in strict["hold_first"]["levels"] if a not in reach["hold_first"]["levels"]])
    txt.append(table(["权重", "场景输入细节（%）", "场景输入其他配文 MRR", "首句输入细节（%）", "首句输入其他配文 MRR"], rows))
    txt.append("## RSITMD 普通检索")
    txt.append("沿用原 448 图子集、真实首句输入及 UCM 冻结检索器。首列使用完整原图库，次列去掉与输入相同的配文。具体图片数和配文数由图库的去重与过滤规则决定。")
    rows = []; delta_rows = []; seen = set()
    for stage, target, key, refs in [
        (ROOT, "teacher", "teacher_minus", ("retained", "parent")),
        (ROOT/"reachable", "reachable_teacher", "teacher_minus", ("retained", "teacher")),
        (ROOT/"strict_rfpo", "strict_coldstrict", "teacher_minus", ("retained", "strict_coldordinary")),
        (ROOT/"strict_rfpo", "strict_warmstrict", "warm_minus", ("retained", "parent"))]:
        src = read(stage/"results/cross_scores.json")
        for a in src["native"]["levels"]:
            if a in seen: continue
            seen.add(a)
            rows.append([NAMES[a], *[number(src[k]["levels"][a]["mR"]["mean"], "mR") for k in ("native","alternatives")]])
        for ref in refs:
            values = []
            for k in ("native","alternatives"):
                e = src[k][key][ref]["scores"]["mR"]
                values.append(f"{100*e['mean']:+.2f} [{100*e['interval'][0]:+.2f}, {100*e['interval'][1]:+.2f}]")
            delta_rows.append([NAMES[target]+" 减 "+NAMES[ref], *values])
    txt.append(table(["权重", "全部配文 mR（%）", "排除输入配文 mR（%）"], rows))
    txt.append(table(["配对比较", "全部配文 mR 差", "排除输入配文 mR 差"], delta_rows))
    (ROOT / "实测数据表.md").write_text("\n\n".join(txt) + "\n")


if __name__ == "__main__":
    main()
