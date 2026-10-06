"""Export matched CCB-versus-Agent text-to-image Top-5 retrieval examples.

This is an evaluation-only utility.  It loads two existing CUP checkpoints,
encodes the same test split, identifies queries whose rank is improved by the
agent prior, and writes a portable JSON manifest plus thumbnail assets.  The
manifest is consumed by ``figs/render_qualitative_comparison.py``.
"""

import argparse
import functools
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vilt.datamodules.multitask_datamodule import MTDataModule
from vilt.modules.vilt_module import ViLTransformerSS


class _MyBackboneFinetuningStub:
    """Compatibility shim for historical Lightning callback pickles."""

    def __init__(self, *args, **kwargs):
        pass


# Some checkpoints pickle the callback as ``__main__.MyBackboneFinetuning``.
# Evaluation only needs the state dict, so a no-op class is sufficient.
import __main__  # noqa: E402
__main__.MyBackboneFinetuning = _MyBackboneFinetuningStub


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]["config"]
    model = ViLTransformerSS(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval()
    return model, config


def make_datasets(config):
    datamodule = MTDataModule(config, dist=False)
    datamodule.setup("test")
    base = datamodule.dms[0]
    return base.make_no_false_test_dset(), base.make_no_false_test_dset(image_only=True)


@torch.no_grad()
def encode_checkpoint(checkpoint_path, device, agent_cache_dir=None):
    model, config = load_model(checkpoint_path, device)
    # Historical checkpoints sometimes retain the cache path used at training
    # time, while test-split hidden states were saved in a sibling cache.
    # Allow an explicit evaluation-only override without touching the model.
    if agent_cache_dir:
        config["agent_cache_dir"] = agent_cache_dir
    text_dataset, image_dataset = make_datasets(config)
    loader_args = dict(
        batch_size=64,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=functools.partial(text_dataset.collate, mlm_collator=None),
    )
    text_loader = DataLoader(text_dataset, **loader_args)
    image_loader = DataLoader(
        image_dataset, batch_size=64, num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=functools.partial(image_dataset.collate, mlm_collator=None),
    )

    text_features, query_image_ids, queries = [], [], []
    for batch in tqdm(text_loader, desc=f"text: {Path(checkpoint_path).parent.parent.name}"):
        features = model.txt_embeds(batch["clip_text_token"].to(device))["text_feats"]
        text_features.append(features.cpu())
        query_image_ids.extend(int(x) for x in batch["img_index"])
        queries.extend(batch["clip_text_txt"])

    image_features, image_ids = [], []
    for batch in tqdm(image_loader, desc=f"image: {Path(checkpoint_path).parent.parent.name}"):
        hidden = batch.get("agent_hidden")
        if hidden is not None:
            hidden = hidden.to(device)
        features = model.img_embeds(batch["clip_img"].to(device), agent_hidden=hidden)["image_feats"]
        image_features.append(features.cpu())
        image_ids.extend(int(x) for x in batch["img_index"])

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        torch.cat(text_features), torch.cat(image_features), query_image_ids,
        image_ids, queries, image_dataset,
    )


def ranks_and_topk(text_features, image_features, query_image_ids, image_ids):
    text_features = torch.nn.functional.normalize(text_features.float(), dim=1)
    image_features = torch.nn.functional.normalize(image_features.float(), dim=1)
    scores = text_features @ image_features.T
    index_for_image_id = {image_id: index for index, image_id in enumerate(image_ids)}
    gt_columns = torch.tensor([index_for_image_id[i] for i in query_image_ids])
    descending = scores.argsort(dim=1, descending=True)
    ranks = (descending == gt_columns.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    return scores, descending[:, :5], ranks


def export_images(dataset, indices, destination):
    destination.mkdir(parents=True, exist_ok=True)
    output = {}
    for image_index in sorted(set(indices)):
        path = destination / f"image_{image_index}.png"
        if not path.exists():
            dataset.get_raw_image(image_index).save(path)
        output[image_index] = str(path.resolve())
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccb_ckpt", required=True)
    parser.add_argument("--agent_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_cases", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--agent_cache_dir",
        help="Optional Agent hidden-state cache override for the Agent checkpoint.",
    )
    args = parser.parse_args()

    ccb = encode_checkpoint(args.ccb_ckpt, args.device)
    agent = encode_checkpoint(args.agent_ckpt, args.device, args.agent_cache_dir)
    ccb_text, ccb_images, ccb_targets, ccb_ids, queries, _ = ccb
    agent_text, agent_images, agent_targets, agent_ids, _, asset_dataset = agent
    if ccb_targets != agent_targets or ccb_ids != agent_ids:
        raise RuntimeError("checkpoint test splits are not aligned")

    ccb_scores, ccb_top5, ccb_ranks = ranks_and_topk(ccb_text, ccb_images, ccb_targets, ccb_ids)
    agent_scores, agent_top5, agent_ranks = ranks_and_topk(agent_text, agent_images, agent_targets, agent_ids)
    improvements = ccb_ranks - agent_ranks

    # Prefer visually self-contained corrections: the Agent retrieves the
    # ground truth in its displayed Top-5 but CCB does not.  Sort those by
    # improvement, then fall back to other improvements if needed.
    all_rows = torch.argsort(improvements, descending=True).tolist()
    candidates = [
        row for row in all_rows
        if agent_ranks[row] < 5 and ccb_ranks[row] >= 5
    ]
    candidates.extend(row for row in all_rows if row not in candidates)
    selected = []
    used_queries = set()
    for row in candidates:
        query_key = queries[row].strip().lower()
        if query_key in used_queries or improvements[row] <= 0:
            continue
        selected.append(row)
        used_queries.add(query_key)
        if len(selected) == args.num_cases:
            break
    if len(selected) < args.num_cases:
        raise RuntimeError("not enough improved queries to form the requested qualitative cases")

    needed_image_ids = []
    for row in selected:
        needed_image_ids.append(ccb_targets[row])
        needed_image_ids.extend(ccb_ids[col] for col in ccb_top5[row].tolist())
        needed_image_ids.extend(agent_ids[col] for col in agent_top5[row].tolist())
    out_dir = Path(args.out_dir)
    paths = export_images(asset_dataset, needed_image_ids, out_dir / "assets")

    records = []
    for row in selected:
        def ranked(top5, scores, ids):
            return [
                {"path": paths[ids[column]], "score": float(scores[row, column])}
                for column in top5[row].tolist()
            ]
        records.append({
            "query": queries[row],
            "ground_truth": paths[ccb_targets[row]],
            "ccb": ranked(ccb_top5, ccb_scores, ccb_ids),
            "agent": ranked(agent_top5, agent_scores, agent_ids),
            "ccb_rank": int(ccb_ranks[row]) + 1,
            "agent_rank": int(agent_ranks[row]) + 1,
        })
    manifest = out_dir / "qualitative_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Saved {manifest}")
    for i, record in enumerate(records, 1):
        print(f"case {i}: CCB rank {record['ccb_rank']} -> Agent rank {record['agent_rank']}")


if __name__ == "__main__":
    main()
