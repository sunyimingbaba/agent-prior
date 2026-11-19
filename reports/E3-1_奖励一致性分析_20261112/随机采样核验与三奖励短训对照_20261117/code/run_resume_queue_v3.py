"""Resume one paused arm, then run the remaining fixed-budget experiment."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import traceback

from stage_io import dump, read, sha

root = Path(__file__).resolve().parent.parent
run = root / "resume_20261119"
continuation = root / "continuation_job_result_v3.json"
post = root / "post_job_result_v3.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def execute(script, *args):
    entry = [script, *map(str, args)]
    log = run / (script.removesuffix(".py") + "".join("_" + str(a) for a in args if a != root) + ".log")
    if (root / "PAUSE_REQUESTED").exists():
        raise SystemExit(75)
    with log.open("xb") as stream:
        with subprocess.Popen([sys.executable, "-u", root / "code" / script, *map(str, args)],
                              cwd=root, env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"},
                              stdout=stream, stderr=subprocess.STDOUT) as child:
            dump(run / "progress.json", {"entry": entry, "pid": child.pid, "started": now(), "log": log.name})
            code = child.wait()
    if code:
        raise SystemExit(code)


def main():
    assert not continuation.exists() and not post.exists()
    with (run / "launch.json").open("x") as f:
        import json
        json.dump({"pid": os.getpid(), "utc": now(), "manifest": sha(run/"code_hashes.json")}, f)
    for manifest in ["pilot_code_hashes_v2.json", "continuation_code_hashes.json",
                     "cached_continuation_hashes.json", "post_code_hashes_v2.json",
                     "resume_20261119/code_hashes.json"]:
        for name, expected in read(root / manifest).items():
            assert sha(root / name) == expected, name
    assert read(run / "preflight.json")["ok"]
    assert read(root / "continuation_job_result_v2.json")["status"] == "paused"
    assert read(root / "post_job_result.json")["status"] == "prior_stage_stopped"
    assert read(root / "training_v2/proxy_rank/result.json")["attempts"] == 32
    assert read(root / "training_v2/quantity_pair/paused.json")["attempts_completed"] == 12
    assert not (root / "training_v2/quantity_hard").exists() and not (root / "evaluation").exists()
    marker = root / "PAUSE_REQUESTED"
    assert sha(marker) == read(run / "pause_inventory.json")["PAUSE_REQUESTED"]["sha256"]
    assert not (run / "original_pause_marker.json").exists()
    marker.rename(run / "original_pause_marker.json")
    plan = read(root / "plan.json")
    target = continuation
    try:
        execute("train_reward_arm_resume_v3.py", "quantity_pair")
        execute("train_reward_arm_v2.py", "quantity_hard")
        policies = plan["evaluation"]["policies"]
        for policy in policies:
            execute("evaluate_policy_v2.py", policy)
        dump(continuation, {"status": "generation_training_evaluation_complete",
             "arms": {a: read(root/"training_v2"/a/"result.json") for a in plan["reward_arms"]},
             "policies": {p: read(root/"evaluation"/p/"result.json") for p in policies},
             "resume_proof_sha256": sha(run/"restore.json")})
        target = post
        execute("analyze_comparison_v4.py")
        execute("verify_raw_cpu_v2.py", root)
        for arm in plan["reward_arms"]:
            if not (root/"training_v2"/arm/"independent_cuda_normalization.json").exists():
                execute("verify_normalization_cuda.py", root, arm)
        execute("verify_training_cpu_v2.py", root)
        execute("verify_metrics_cpu.py", root)
        execute("verify_assets_final_v2.py", root)
        dump(post, {"status": "complete", "analysis_sha256": sha(root/"analysis.json"),
             "independent_raw_sha256": sha(root/"independent_raw_verification.json"),
             "independent_training_sha256": sha(root/"independent_training_verification.json"),
             "independent_metrics_sha256": sha(root/"independent_metric_verification.json"),
             "final_asset_sha256": sha(root/"final_asset_verification.json"), "utc": now()})
    except BaseException as exc:
        dump(target, {"status": "paused" if isinstance(exc, SystemExit) and exc.code == 75 else "failed",
                      "last_entry": read(run/"progress.json") if (run/"progress.json").exists() else None,
                      "error": traceback.format_exc(), "utc": now()})
        raise


if __name__ == "__main__":
    main()
