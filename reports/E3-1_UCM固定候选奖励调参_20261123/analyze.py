"""Replay reward choices on frozen UCM candidates. Requires NumPy, SciPy and PyTorch."""
import csv
import hashlib
import itertools
import json
import platform
import re
import runpy
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCE = OUT.parent / "E3-1_奖励一致性分析_20261112" / "空间关系候选奖励核验_20261120"
PREVIOUS = OUT.parent / "E3-2_UCM同起点三奖励短训_20261120" / "results/eos_shared_candidates.csv"
REWARD_PATH = ROOT / "CUP/agent/grpo/reward.py"
WEIGHTS = (0, .125, .25, .5, .75, 1)
BONUSES = (0, .05, .1, .2)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_rank(scores):
    return (rankdata(scores) - 1) / (len(scores) - 1)


def key(name, bonus):
    return f"{name}+format{bonus:g}"


def selected(scores):
    return np.flatnonzero(scores == scores.max())


def paired_stats(groups, metric, reference=None):
    pairs = sorted({g["pair"] for g in groups})
    if not pairs:
        return None
    pooled = {p: [] for p in pairs}
    for group in groups:
        value = group[metric] if reference is None else group[metric] - group[reference]
        pooled[group["pair"]].append(value)
    sums = np.array([sum(pooled[p]) for p in pairs])
    sizes = np.array([len(pooled[p]) for p in pairs])
    indices = np.random.default_rng(20261123).integers(len(pairs), size=(20000, len(pairs)))
    limits = np.quantile(sums[indices].sum(axis=1) / sizes[indices].sum(axis=1), [.025, .975])
    return {"mean": float(sums.sum() / sizes.sum()), "lo": float(limits[0]), "hi": float(limits[1]),
            "images": len(groups), "pairs": len(pairs)}


def main():
    sc = SOURCE / "scoring"
    nat = SOURCE / "native"
    source_rows = sc / "rows.json"
    third_path = SOURCE / "第三措辞逐候选.csv"
    native_csv = SOURCE / "训练原描述逐候选.csv"
    binding = read(sc / "binding.json")
    index = read(nat / "index.json")
    assert digest(sc / "vectors.pt") == binding["bank"]
    assert digest(source_rows) == binding["rows"]
    assert digest(nat / "features.pt") == index["sha256"]
    rows = read(source_rows)
    vectors = torch.load(sc / "vectors.pt", weights_only=True, map_location="cpu")["visual"].double().numpy()
    captions = torch.load(nat / "features.pt", weights_only=True, map_location="cpu")
    similarity = vectors @ captions["features"].double().numpy().T
    assert similarity.shape == (576, 1635) and np.isfinite(similarity).all()
    even = np.array([int(hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[-1], 16) % 2 == 0
                     for ids in captions["tokens"].tolist()])
    assert len(set(map(tuple, captions["tokens"].tolist()))) == len(even)
    get_format = runpy.run_path(str(REWARD_PATH))["r_format"]
    third = {(r["mode"], r["image"], int(r["k"])): r for r in load_csv(third_path)}
    native_check = {(r["mode"], r["image"], int(r["k"])): r for r in load_csv(native_csv)
                    if r["gallery"] == "drop_first"}
    previous = {(r["image"], int(r["k"])): r for r in load_csv(PREVIOUS)}
    groups = {}
    raw = []
    selections = []
    shifts = []
    observed_previous = 0

    for (mode, image), candidates in itertools.groupby(
        sorted((r["mode"], r["image"], r["k"], i) for i, r in enumerate(rows) if r["k"] >= 0),
        key=lambda item: item[:2],
    ):
        entries = list(candidates)
        assert [k for _, _, k, _ in entries] == list(range(8))
        ids = [i for _, _, _, i in entries]
        samples = [rows[i] for i in ids]
        pair = samples[0]["pair"]
        assert all(r["pair"] == pair for r in samples)
        img = {"mode": mode, "image": image, "pair": pair}
        m = index["maps"][image]
        s = similarity[ids]
        positive = m["positive"]
        first = m["first"]
        assert first in positive
        other_positive = [i for i in positive if i != first]
        assert other_positive
        neg = np.ones(len(even), dtype=bool)
        neg[positive] = False
        reward_neg = neg & even
        eval_neg = neg & ~even
        assert not np.any(reward_neg & eval_neg)
        a = np.array([r["A"] for r in samples])
        b = np.array([r["B"] for r in samples])
        c = np.array([float(third[(mode, image, k)]["C"]) for k in range(8)])
        for k in range(8):
            c_row = third[(mode, image, k)]
            assert c_row["pair"] == pair and float(c_row["A"]) == a[k] and float(c_row["B"]) == b[k]
            assert np.isclose(float(c_row["C_pos"]) - float(c_row["C_neg"]), c[k], atol=1e-14, rtol=0)
        caption_margin = s[:, first] - s[:, reward_neg].max(axis=1)
        prior = np.array([r["soft"] for r in samples])
        old = np.array([r["reward"] for r in samples])
        format_value = np.array([get_format(r["text"]) for r in samples])
        assert np.array_equal(format_value, [r["format"] for r in samples])
        assert np.array_equal(prior, [1 / (1 + r["rank0"]) for r in samples])
        assert np.array_equal(old, prior + format_value)
        fmt = format_value > 0
        empty = np.array([any(not content.strip() for content in re.findall(
            r"<answer>(.*?)</answer>", r["text"], flags=re.DOTALL)) for r in samples])
        eos = []
        for k, row in enumerate(samples):
            trace = read(SOURCE / "samples" / mode / Path(image).stem / f"sample{k}.json")
            assert (trace["image"], trace["mode"], trace["k"], trace["vector_sha256"]) == (
                image, mode, k, row["native_tensor"])
            eos.append(trace["rounds"][0]["ids"] in ([151645], [151643]))
        eos = np.array(eos)
        cap_rank, detail_rank = relative_rank(caption_margin), relative_rank(a)
        for k in range(8):
            if mode == "scene" and (image, k) in previous:
                row = previous[(image, k)]
                assert row["pair"] == pair
                assert float(row["old"]) == old[k]
                assert float(row["caption"]) == rankdata(caption_margin)[k]
                assert float(row["hard"]) == ((rankdata(caption_margin) + rankdata(a)) / 2)[k]
                assert (row["eos"] == "True") == eos[k]
                observed_previous += 1

        best = s[:, other_positive].max(axis=1)
        native_margin = best - s[:, eval_neg].max(axis=1)
        native_rr = 1 / (1 + (s[:, eval_neg] > best[:, None]).sum(axis=1))
        full_margin = best - s[:, neg].max(axis=1)
        full_rr = 1 / (1 + (s[:, neg] > best[:, None]).sum(axis=1))
        for k in range(8):
            archived = native_check[(mode, image, k)]
            assert float(archived["rr"]) == full_rr[k]
            assert np.isclose(float(archived["margin"]), full_margin[k], atol=1e-14, rtol=0)
            raw.append(dict(**img, k=k, format=format_value[k], empty_answer=bool(empty[k]),
                            immediate_eos=bool(eos[k]), proxy_rank=int(samples[k]["rank0"]) + 1,
                            proxy_score=prior[k], old_reward=old[k], caption_margin=caption_margin[k],
                            caption_rank=cap_rank[k], detail_A=a[k], detail_rank=detail_rank[k],
                            detail_B=b[k], detail_C=c[k], native_rr=native_rr[k],
                            native_margin=native_margin[k], full_rr=full_rr[k], full_margin=full_margin[k]))
        # Use integer average-rank units and dyadic weight numerators before scaling.
        # This preserves mathematical ties across components.
        caption_units = np.rint(cap_rank * 14).astype(int)
        detail_units = np.rint(detail_rank * 14).astype(int)
        bases = {"proxy_rank": np.rint(relative_rank(prior) * 14).astype(int) * 8}
        for weight in WEIGHTS:
            detail_weight = int(weight * 8)
            bases[f"detail{weight:g}"] = (8 - detail_weight) * caption_units + detail_weight * detail_units
        methods = {"random": np.zeros(8), "original": old, "proxy_raw": prior}
        for name, values in bases.items():
            for bonus in BONUSES:
                methods[key(name, bonus)] = values * 20 + int(bonus * 20) * 112 * fmt
        metrics = dict(B=b > 0, B_margin=b, C=c > 0, BC=(b > 0) & (c > 0), C_margin=c,
                       native_rr=native_rr, native_margin=native_margin, full_rr=full_rr,
                       full_margin=full_margin, eos=eos, format=fmt)
        group = dict(**img, coverage_B=int((b > 0).sum()), coverage_BC=int(((b > 0) & (c > 0)).sum()),
                     format_candidates=int(fmt.sum()), empty_answers=int(empty.sum()),
                     proxy_span=float(np.ptp(prior)), caption_span=float(np.ptp(caption_margin)))
        for name, values in methods.items():
            winners = selected(values)
            expected = {metric: float(values[winners].mean()) for metric, values in metrics.items()}
            group.update({name + "/" + metric: value for metric, value in expected.items()})
            selections.append(dict(**img, method=name, winners=" ".join(map(str, winners)), **expected))

        checks = [("proxy_raw", "original")]
        checks.extend((key(base, 0), key(base, bonus)) for base in bases for bonus in BONUSES if bonus)
        for base, changed in checks:
            before, after = selected(methods[base]), selected(methods[changed])
            shifts.append(dict(**img, base=base, method=changed,
                               before=" ".join(map(str, before)), after=" ".join(map(str, after)),
                               changed=not np.array_equal(before, after), disjoint=not np.intersect1d(before, after).size,
                               **{metric: group[changed + "/" + metric] - group[base + "/" + metric]
                                  for metric in ("B", "C", "native_rr", "native_margin", "eos")}))
        groups.setdefault(mode, []).append(group)

    assert observed_previous == len(previous) == 192
    assert len(raw) == 512 and len(third) == 576
    report = {}
    for mode, all_images in groups.items():
        assert len(all_images) == 32 and len({g["pair"] for g in all_images}) == 16
        for pair in {g["pair"] for g in all_images}:
            assert sum(g["pair"] == pair for g in all_images) == 2
        mixed = [g for g in all_images if 0 < g["coverage_B"] < 8]
        summary = {}
        for method in methods:
            result = {metric: paired_stats(all_images, method + "/" + metric) for metric in metrics}
            for reference in ("random", "original", key("detail0", 0)):
                if reference == method:
                    continue
                result["vs_" + reference] = {
                    metric: paired_stats(all_images, method + "/" + metric, reference + "/" + metric)
                    for metric in ("B", "C", "BC", "native_rr", "native_margin", "eos")}
            result["mixed_B"] = paired_stats(mixed, method + "/B")
            summary[method] = result

        format_summary = {}
        for _, name in checks:
            affected = [r for r in shifts if r["mode"] == mode and r["method"] == name and r["changed"]]
            format_summary[name] = dict(changed=len(affected), disjoint=sum(r["disjoint"] for r in affected),
                                        improved_B=sum(r["B"] > 0 for r in affected),
                                        worse_B=sum(r["B"] < 0 for r in affected),
                                        total_B_change=sum(r["B"] for r in affected) / len(all_images))
        report[mode] = dict(images=len(all_images), mixed_B=len(mixed),
                            oracle_B=sum(g["coverage_B"] > 0 for g in all_images),
                            oracle_BC=sum(g["coverage_BC"] > 0 for g in all_images),
                            format_mixed_images=sum(0 < g["format_candidates"] < 8 for g in all_images),
                            format_candidates=sum(g["format_candidates"] for g in all_images),
                            empty_answers=sum(g["empty_answers"] for g in all_images),
                            proxy_span_median=float(np.median([g["proxy_span"] for g in all_images])),
                            caption_span_median=float(np.median([g["caption_span"] for g in all_images])),
                            format_shifts=format_summary, methods=summary)

    primary = report["scene"]["methods"]
    passed = []
    for weight in WEIGHTS[1:-1]:
        method = key(f"detail{weight:g}", 0)
        row = primary[method]
        vs_caption = row["vs_" + key("detail0", 0)]
        if (row["vs_original"]["B"]["mean"] > 0 and vs_caption["B"]["mean"] > 0
                and vs_caption["native_rr"]["mean"] >= 0 and vs_caption["native_margin"]["mean"] >= 0):
            passed.append(method)
    write_csv("逐候选分项.csv", raw)
    write_csv("逐图奖励选择.csv", selections)
    write_csv("格式项改选.csv", shifts)
    write_csv("逐图候选覆盖.csv", [{k: v for k, v in g.items() if "/" not in k}
                                  for all_images in groups.values() for g in all_images])
    result = dict(modes=report, exploratory_passes=passed, verified_previous_candidates=observed_previous,
                  verified_native_candidates=len(raw), seed=20261123, bootstrap_draws=20000,
                  sha256={str(p.relative_to(ROOT)): digest(p) for p in
                          (source_rows, sc / "vectors.pt", nat / "features.pt", nat / "index.json",
                           third_path, native_csv, PREVIOUS, REWARD_PATH, OUT / "实验口径.md", Path(__file__))},
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                scipy=scipy.__version__, torch=torch.__version__))
    (OUT / "汇总.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Replayed {len(raw)} candidates, checked {observed_previous} previous scores, passed: {passed}")


if __name__ == "__main__":
    main()
