# School A100 — legacy main-method smoke specification

Date: 2026-10-26. SSH root@172.18.113.4 port 1024. No password stored.

## Scope and isolation

- Test the pre-Stage-A main method, not Stage A and not a full training run.
- Frozen source: tag `agse-pre-stage-a-20261008`, commit `2f4dde543c8c16991003e35dd8bdb7ac01fe22c8`.
- New task root: `/public/cup-legacy-smoke-20261026`.
- Preserve base environment, original models, data and experiment results.
- Host reports Ubuntu 22.04, x86_64, Python 3.12.6, torch 2.4.1+cu124,
  torchvision 0.19.1+cu124; one idle A100-SXM4-80GB, MIG disabled.
- Isolated venv with base system packages (to reuse CUDA torch); install missing
  legacy dependencies into the venv only. No global pip installation.
- Legacy target versions: transformers 4.57.2, tokenizers 0.22.1,
  pytorch-lightning 1.4.0, torchmetrics 0.7.2, timm 0.5.4, peft 0.10.0.
- Pin NumPy 1.26.4; use the preserved Linux CPython 3.12 wheelhouse.
- Compatibility shims and third-party environment patches must be recorded;
  do not alter the legacy model, reward or data protocol to make the test pass.
- Models: local Qwen2.5-VL-7B-Instruct and CLIP ViT-L/14 backups.
- Tiny UCM smoke inputs only; preserve original caption-conditioned behavior.

## Gates

1. Seed-0 CUDA 16x16 matrix forward/backward: finite loss and gradients.
2. Load Qwen plus saved main-method LoRA; run bounded generation and tool check.
3. Generate a small real prior cache; verify shape and finite values.
4. Real retrieval forward/backward and optimizer update with agent prior.
5. Bounded legacy RL/LoRA optimizer step if all required scoring inputs exist.
6. Independent agent follows the documented smoke invocation verbatim.

Readiness is not established by imports alone. Smoke results are not retrieval
benchmark results and do not resolve the legacy caption/validation/reward issues.
No automatic shutdown and no full experiment queue.

## Preflight evidence

Base CUDA witness passed: loss 35.21732711791992, finite_grad=true.
The run-experiment skill's referenced shared compute-env-contract.md is absent;
this explicit specification and the witness/doc replay checks are used instead.

Further commands and measured outcomes will be appended after verification.

## Actual setup and first smoke results

- Copied frozen source, wheelhouse, Qwen/CLIP weights, UCM arrows, saved main
  LoRA, scene classifier, scorer checkpoint and feature cache into task root.
- Virtual environment: `/public/cup-legacy-smoke-20261026/env`.
- `pip check`: no broken requirements. Base torch and all original assets kept.
- The environment sitecustomize reuses the preserved shim and adds AdamW alias.
- Qwen metadata compatibility: removed `transformers_version` from the task
  copy only; config.original.json retained alongside it. No weight edits.
- Added a CLIP cache symlink only when the destination did not already exist.
- First complete invocation: `bash /public/cup-legacy-smoke-20261026/tests/run_smoke.sh`.
- Log `logs/smoke_retry1.log`: ALL_LEGACY_SMOKE_PASSED at 11:30:19 +08:00.
- Rollout: three real training images, finite [3584] priors, two genuine scene
  tool injections. Text quality anomaly: one empty response and unrelated
  `addCriterion` after tool results on two samples. Execution is not quality.
- Retrieval: real inherited three losses and backward/AdamW update passed;
  agent projection gradient finite/nonzero and parameters changed; finite 3x3
  score matrix. Tiny batch=3 allocated peak 2.893 GiB (not a full-workload peak).
- Legacy RL script completed exactly one bounded step (group8/sample4) and
  wrote a separate smoke LoRA. This does not fix the known reward protocol.
- Original full run_frozen Trainer wiring, full batch32/accumulation8, formal
  metrics and long-run stability are not certified by this smoke.
- Independent doc replay is performed separately and recorded in logs/doc_replay.log.

Spec artifacts SHA256:

- compat_sitecustomize.py: `361dc8e5685ce7315950f1130d5f2b64f13ddac4e1694ecbf7b842f3d403722c`
- run_smoke.sh: `892609a1b9294d8eb0c2b72fb30ecb9c4589b4a989e1bb3e70d69db9bd3f10bb`
- retrieval_smoke.py: `6be1a731555dbc26af208d8b42a72dc258dadda2bf16025f119b0d17ffb86d8f`
- Qwen patched config: `1a1306a1b2a9b144837681fbe4b2fe7cd85f105234d767a3af75ba5beae7c190`

Independent doc replay: exit 0, ALL_LEGACY_SMOKE_PASSED at 11:35:48 +08:00.
No material invocation divergence. It reused cached priors, completed retrieval
update and one RL step with eight finite reward rows. GPU afterward idle at
4MiB / 0%. Text-quality anomaly and untested full-Trainer/full-batch boundaries
remain. No full training was launched and no shutdown performed.
