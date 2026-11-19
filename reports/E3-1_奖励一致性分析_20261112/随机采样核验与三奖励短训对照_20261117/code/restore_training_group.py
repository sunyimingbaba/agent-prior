"""Restore the exact state at a committed group boundary before sampling again."""
import json
import random

import numpy as np
import torch

from checkpoint_io import restore_adapter_exact
from milestone_common import lora_hash
from stage_io import dump, read, sha


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def nested_tuple(value):
    return tuple(map(nested_tuple, value)) if isinstance(value, list) else value


def restore_rng(folder):
    rng = torch.load(folder / "torch_rng.pt", map_location="cpu", weights_only=True)
    assert len(rng["torch_cuda"]) == torch.cuda.device_count()
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state_all(rng["torch_cuda"])
    assert torch.equal(torch.get_rng_state(), rng["torch_cpu"])
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), rng["torch_cuda"]))

    saved = read(folder / "python_numpy_rng.json")
    py_state = nested_tuple(saved["python_random_state"])
    random.setstate(py_state)
    assert random.getstate() == py_state
    ns = saved["numpy_random_state"]
    state = (ns["kind"], np.asarray(ns["keys"], dtype=np.uint32), ns["position"],
             ns["has_gauss"], ns["cached_gaussian"])
    np.random.set_state(state)
    actual = np.random.get_state()
    assert actual[0] == state[0] and np.array_equal(actual[1], state[1]) and actual[2:] == state[2:]


def restore_group(root, out, model, optimizer, scorer, plan, prompt):
    arm = out.name
    run = root / "resume_20261119"
    assert read(run / "preflight.json")["ok"] and not (run / "restore.json").exists()
    pointer = read(out / "paused.json")
    n, count = pointer["attempts_completed"], pointer["optimizer_updates"]
    manifest = read(out / f"paused_{n}_checkpoint_manifest.json")
    folder = out / f"paused_{n}" / "checkpoint"
    assert manifest["arm"] == arm and manifest["attempts_completed"] == n and manifest["optimizer_updates"] == count
    assert {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()} == set(manifest["files"])
    for name, meta in manifest["files"].items():
        path = folder / name
        assert path.stat().st_size == meta["bytes"] and sha(path) == meta["sha256"], name
    state = read(folder / "training_state.json")
    assert state["arm"] == arm and state["attempts_completed"] == n and state["optimizer_updates"] == count
    assert state["plan_sha256"] == sha(root / "plan.json")
    assert state["prompt_lock_sha256"] == sha(root / "prompt_selection_lock.json")
    initial = plan["policy"]["lora_hash"]
    settings = read(out / "settings.json")
    assert settings["config"] == plan["training"] and settings["prompt"] == prompt
    assert settings["readout_frozen"] == scorer.verify_frozen()
    assert state["initial_policy_hash"] == settings["initial_policy_hash"] == initial
    rows, updates = read_rows(out / "samples.jsonl"), read_rows(out / "updates.jsonl")
    group_size = plan["training"]["group_size"]
    assert len(updates) == n and len(rows) == n * group_size
    order = plan["training"]["image_order"]
    previous, total = initial, 0
    for index, update in enumerate(updates):
        assert (update["attempt"], update["image"], update["policy_hash_before"]) == (index + 1, order[index], previous)
        for sample, row in enumerate(rows[index * group_size:(index + 1) * group_size]):
            assert (row["attempt"], row["sample"], row["image"], row["policy_hash_before"]) == (index + 1, sample, order[index], previous)
            assert row["seed"] == 2026111800 + 100 * index + sample
        total += int(update["optimizer_update"])
        assert total == update["optimizer_updates_so_far"]
        previous = update["policy_hash_after"]
    assert previous == state["policy_hash"] == manifest["policy_hash"] and total == count
    result = restore_adapter_exact(model, folder)
    assert lora_hash(model) == previous
    saved = torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=True)
    groups = optimizer.state_dict()["param_groups"]
    assert saved["param_groups"] == groups
    optimizer.load_state_dict(saved)
    param_ids = groups[0]["params"]
    params = optimizer.param_groups[0]["params"]
    assert len(saved["state"]) == len(params) == len(param_ids) == 584
    for ident, param in zip(param_ids, params):
        actual = optimizer.state[param]
        expected = saved["state"][ident]
        assert actual.keys() == expected.keys() and actual["step"].item() == count
        for key, tensor in actual.items():
            assert tensor.dtype == expected[key].dtype and torch.equal(tensor.cpu(), expected[key]), (ident, key)
    del saved
    restore_rng(folder)
    progress = read(root / "training_progress.json")
    assert progress["arm"] == arm and progress["attempts_completed"] == n and progress["optimizer_updates"] == count
    dump(run / "restore.json", dict(ok=True, arm=arm, prior_groups=n, updates=count, prefix_samples=len(rows),
         adapter_verification=result, policy_hash=previous, optimizer_states_exact=len(params),
         rng_restored=["torch_cpu", "torch_cuda_all", "python", "numpy"],
         next_group=n+1, next_image=order[n], next_seed=2026111800+100*n,
         samples_prefix_sha256=sha(out/"samples.jsonl"), updates_prefix_sha256=sha(out/"updates.jsonl")))
    return rows, updates, count, previous, progress["elapsed_seconds"]
