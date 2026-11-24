"""Index the frozen inputs, complete training and evaluation stages, and returned evidence."""
from pathlib import Path
import hashlib, json
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    verification=read('results/return_verification.json')
    assert sum(verification['traces'].values())==5320 and len(verification['adapters'])==10
    assert read('inputs/nomination_bounded.json')['eligible']
    assert read('results/check_dev.json')['recalls']==72 and max(read('results/blind.json')['baseline_replay_max_difference'].values())==0
    assert read('results/recheck_independent.json')['candidate_positions']==4544
    assert read('results/recheck_fresh.json')['candidate_positions']==3904
    assert read('results/replica.json')['stratified_detail_pooled']['pairs']==145
    assert read('results/swap_independent.json')['all']['images']==142
    assert read('results/relative_independent.json')['images']==142
    assert digest(ROOT/'records/run_logs.tar.gz')==(ROOT/'records/run_logs.sha256').read_text().split()[0]
    path=ROOT/'results/final.json'
    inputs={str(p.relative_to(ROOT)):digest(p) for root in ('inputs','reward_confirm/inputs','confirm/inputs','guard/inputs') for p in (ROOT/root).glob('*.json')}
    results={p.name:digest(p) for p in sorted((ROOT/'results').glob('*.json')) if p!=path and not p.name.startswith('dev_joint_tmp')}
    path.write_text(json.dumps(dict(adapters=len(verification['adapters']),returned_greedy_traces=sum(verification['traces'].values()),
        inputs=inputs,results=results,logs=digest(ROOT/'records/run_logs.tar.gz')),indent=2)+'\n')
if __name__=='__main__':main()
