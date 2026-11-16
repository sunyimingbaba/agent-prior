"""Re-hash original source, model assets, historical checkpoints and parent evidence."""
from pathlib import Path
from milestone_common import ROOT, PLAN, BASE, ASSETS
from stage_io import read, dump, sha

parent = Path(PLAN['parent_root'])
source = read(ROOT/'source_snapshot/inventory_before.json')['files']
for n,h in source.items():
    assert sha(BASE/'CUP'/n)==h,n
qwen = read(parent/'inputs/parent_preflight.json')['model_asset_hashes']
for n,v in qwen.items():
    assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==v['sha256'],n
assert sha(ASSETS/'ucm_scene_cls.pt')==PLAN['model_hashes']['scene_head_sha256']
assert sha(ASSETS/'ucm_main_domain_tool.ckpt')==PLAN['utility_model_sha256']
assert sha(ASSETS/'models/ViT-L-14.pt')==PLAN['model_hashes']['clip_sha256']
oldplan = read(parent/'plan.json')
checkpoints = 0
for policy in oldplan['policies']:
    manifest = read(Path(oldplan['parent_permutation_root'])/policy['checkpoint_manifest'])
    for n,v in manifest['files'].items():
        assert sha(Path(policy['checkpoint'])/n)==v['sha256'],n
        checkpoints += 1
for split,v in read(ROOT/'cache/dataset_sources.json').items():
    assert sha(ASSETS/f'ucm/ucm_caption_karpathy_label_{split}.arrow')==v['arrow_sha256']
previous = read(parent/'file_checksums.json')
for n,v in previous.items():
    assert sha(parent/n)==v['sha256'],n
dump(ROOT/'final_source_audit.json',{'ok':True,'original_python_files_unchanged':len(source),
    'Qwen_base_files_unchanged':len(qwen),'old_LoRA_checkpoint_files_unchanged':checkpoints,
    'original_CLIP_AGSE_and_scene_head_unchanged':True,'all_three_arrow_files_unchanged':True,
    'previous_phase_evidence_files_unchanged':len(previous)})
print('ALL_ORIGINAL_SOURCES_AND_MODELS_PRESERVED',flush=True)
