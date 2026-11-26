from local import ROOT, read, dump, sha


def main():
    inventory = {}
    for stage, expected_models, expected_eval, expected_cross in ((ROOT, 6, 3040, 1792), (ROOT/"reachable", 4, 1856, 896),
                                                                  (ROOT/"strict_rfpo", 6, 2784, 2688)):
        core = read(stage / "results/backup.json")
        cross = read(stage / "results/cross_backup.json")
        actual = sum(sum(r["images"].values()) for r in core["archive"].values())
        external = core["external_models"]
        cp = {p.parent.name: read(p)["final"] for p in (stage / "models").glob("*/complete.json")}
        assert len(cp) == expected_models and actual == expected_eval and sum(r["images"] for r in cross.values()) == expected_cross
        proofs = [stage/"results"/(n+".json") for n in ("backup", "cross_backup", "score", "cross_scores")]
        name = stage.name if stage != ROOT else "base"
        inventory[name] = dict(adapters=cp, external=external, ucm_generated=actual, rsitmd_generated=expected_cross,
                               sha256={f.stem: sha(f) for f in proofs})
    dump(ROOT/"results/inventory.json", inventory)


if __name__ == "__main__":
    main()
