"""Replay the packaged CPU analysis in a separate temporary output tree."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
from common import ROOT,read,dump,sha
out=Path(tempfile.mkdtemp(prefix='agse-multineg-replay-'))
for name in ['code','source_data','case_traces','inputs']:shutil.copytree(ROOT/name,out/name,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['plan.json','采样覆盖与先验置换预案_20261115.json']:shutil.copy2(ROOT/name,out/name)
for script in ['analyze_multineg.py','build_case_review.py','verify_multineg.py']:
    result=subprocess.run([sys.executable,str(out/'code'/script)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
left=read(ROOT/'analysis.json');right=read(out/'analysis.json');left.pop('elapsed_seconds');right.pop('elapsed_seconds');assert left==right
names=['candidate_scores.json','per_image_selections.json','candidate_scores.csv','quantity_score_matrix.csv','per_image_selections.csv','paired_contrasts.csv','candidate_mechanics.csv','training_reward_replay.csv','training_group_mechanics.csv','own_shared_state_sensitivity.csv','count_confusion.csv','selector_changes.csv','decisive_case_2035.json','独立核验.json']
for name in names:assert (ROOT/name).read_bytes()==(out/name).read_bytes(),name
dump(ROOT/'打包材料重放核验.json',{'ok':True,'fresh_output_root':str(out),'analysis_numerical_tree_identical':True,'byte_identical_outputs':len(names),'independent_reconstruction_passed':True,'parent_directory_needed':False,'new_optimizer_updates':0})
print('PACKAGED_CPU_REPLAY_PASSED',len(names))
