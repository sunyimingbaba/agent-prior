"""Compare one and two hard-negative phrasings with a third phrasing held out."""
import csv
import itertools
import json
import runpy
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "E3-1_UCM固定候选奖励调参_20261123"
SOURCE = PRIOR / "逐候选分项.csv"
WEIGHTS = (25, 50, 75, 100)
PHRASINGS = "ABC"
METRICS = ("correct", "margin", "native_rr", "native_margin", "eos")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    shared = runpy.run_path(str(PRIOR / "analyze.py"))
    digest = shared["digest"]
    load_csv = shared["load_csv"]
    paired_stats = shared["paired_stats"]
    relative_rank = shared["relative_rank"]
    selected = shared["selected"]
    prior_report = json.loads((PRIOR / "汇总.json").read_text())
    source = sorted(load_csv(SOURCE), key=lambda r: (r["mode"], r["image"], int(r["k"])))
    assert len(source) == 512
    output = []
    grouped = {}

    for (mode, image), source_group in itertools.groupby(source, key=lambda r: (r["mode"], r["image"])):
        rows = list(source_group)
        assert [int(r["k"]) for r in rows] == list(range(8))
        pair = rows[0]["pair"]
        assert all(r["pair"] == pair for r in rows)
        c = np.array([float(r["caption_margin"]) for r in rows])
        caption_units = np.rint(relative_rank(c) * 14).astype(int)
        assert np.array_equal(caption_units, np.rint([float(r["caption_rank"]) * 14 for r in rows]))
        directions = {p: np.array([float(r["detail_" + p]) for r in rows]) for p in PHRASINGS}
        detail_units = {p: np.rint(relative_rank(values) * 14).astype(int) for p, values in directions.items()}
        native_rr = np.array([float(r["native_rr"]) for r in rows])
        native_margin = np.array([float(r["native_margin"]) for r in rows])
        eos = np.array([r["immediate_eos"] == "True" for r in rows])
        old = np.array([float(r["old_reward"]) for r in rows])
        for heldout in PHRASINGS:
            allowed = [p for p in PHRASINGS if p != heldout]
            left, right = allowed
            methods = {"random": np.zeros(8), "original_rfpo": old, "caption_only": caption_units}
            for weight in WEIGHTS:
                # Integer half-rank units keep equally scored candidates tied.
                content = (100 - weight) * 2 * caption_units
                methods[f"{left}_{weight}"] = content + weight * 2 * detail_units[left]
                methods[f"{right}_{weight}"] = content + weight * 2 * detail_units[right]
                methods[f"{left}{right}_{weight}"] = content + weight * (detail_units[left] + detail_units[right])
            metrics = dict(correct=directions[heldout] > 0, margin=directions[heldout],
                           native_rr=native_rr, native_margin=native_margin, eos=eos)
            group = dict(pair=pair, image=image, good_candidates=int(metrics["correct"].sum()))
            for method, scores in methods.items():
                winners = selected(scores)
                result = {name: float(values[winners].mean()) for name, values in metrics.items()}
                group.update({method + "/" + key: value for key, value in result.items()})
                output.append(dict(mode=mode, image=image, pair=pair, heldout=heldout, method=method,
                                   winners=" ".join(map(str, winners)), **result))
            grouped.setdefault((mode, heldout), []).append(group)

    summary = {}
    for (mode, heldout), groups in grouped.items():
        assert len(groups) == 32 and len({g["pair"] for g in groups}) == 16
        left, right = [p for p in PHRASINGS if p != heldout]
        methods = ["random", "original_rfpo", "caption_only"]
        methods += [f"{source}_{weight}" for weight in WEIGHTS for source in (left, right, left + right)]
        mixed = [g for g in groups if 0 < g["good_candidates"] < 8]
        result = {}
        for method in methods:
            stats = {metric: paired_stats(groups, method + "/" + metric) for metric in METRICS}
            stats["mixed_correct"] = paired_stats(mixed, method + "/correct")
            comparisons = []
            if method not in ("random", "original_rfpo", "caption_only"):
                comparisons.append("caption_only")
                weight = int(method.split("_")[1])
                if method.startswith(left + right):
                    comparisons.extend((f"{left}_{weight}", f"{right}_{weight}"))
            for other in comparisons:
                stats["vs_" + other] = {metric: paired_stats(groups, method + "/" + metric, other + "/" + metric)
                                         for metric in METRICS}
            result[method] = stats
        summary.setdefault(mode, {})[heldout] = {
            "mixed_images": len(mixed),
            "oracle_n": sum(g["good_candidates"] > 0 for g in groups),
            "methods": result,
        }
        # The known A-only choices and all shared references must replay identically.
        previous = prior_report["modes"][mode]["methods"]
        reference_metrics = [("native_rr", "native_rr"), ("native_margin", "native_margin")]
        if heldout != "A":
            reference_metrics.extend((("correct", heldout), ("margin", heldout + "_margin")))
        for method, old in (("random", "random"), ("original_rfpo", "original"), ("caption_only", "detail0+format0")):
            for current, archived in reference_metrics:
                assert abs(result[method][current]["mean"] - previous[old][archived]["mean"]) < 1e-14
        if heldout != "A":
            for weight in WEIGHTS:
                assert abs(result[f"A_{weight}"]["correct"]["mean"] -
                           previous[f"detail{weight / 100:g}+format0"][heldout]["mean"]) < 1e-14

    checks = {}
    for mode in summary:
        primary = summary[mode]["C"]["methods"]["AB_25"]
        checks[mode] = {
            "C_better_than_A": primary["vs_A_25"]["correct"]["mean"] > 0,
            "C_better_than_B": primary["vs_B_25"]["correct"]["mean"] > 0,
            "native_rr_at_least_caption_only": primary["vs_caption_only"]["native_rr"]["mean"] >= 0,
            "native_margin_at_least_caption_only": primary["vs_caption_only"]["native_margin"]["mean"] >= 0,
        }
    write_csv(OUT / "逐图措辞选择.csv", output)
    (OUT / "汇总.json").write_text(json.dumps({
        "seed": 20261123, "bootstrap_draws": 20000, "candidates": len(source), "screens": checks, "modes": summary,
        "sha256": {p.name: digest(p) for p in
                   (SOURCE, PRIOR / "汇总.json", PRIOR / "analyze.py", OUT / "实验口径.md")},
        "analysis_sha256": digest(Path(__file__)),
    }, ensure_ascii=False, indent=2) + "\n")
    print("Replayed 512 candidates in 3 held-out wording conditions and verified shared reference results.")
    print(checks)


if __name__ == "__main__":
    main()
