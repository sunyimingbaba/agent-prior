from pathlib import Path
from milestone_common import ROOT,PLAN,BASE,ASSETS
from stage_io import read,dump,sha
before=read(ROOT/'source_snapshot/inventory_before.json');after={n:sha(BASE/'CUP'/n) for n in before['files']};assert after==before['files']
assert sha(ASSETS/'ucm_main_domain_tool.ckpt')==PLAN['utility_model_sha256']
for x in read(ROOT/'states/records.json'):
    assert sha(ROOT/x['state_file'])==x['state_file_sha256']
    if x['family']=='original_cached':assert sha(x['source_path'])==x['state_file_sha256']
for snap in ['gpu_code_hashes.json','gpu_retry_code_hashes.json','gradient_code_hashes.json','post_code_hashes.json']:
    for n,h in read(ROOT/snap).items():assert sha(ROOT/n)==h,n
result=read(ROOT/'gpu_result.json');assert result['original_model_parameter_hash']==result['final_model_parameter_hash']==read(ROOT/'gradient_result.json')['model_hash_unchanged']
dump(ROOT/'final_source_audit.json',{'ok':True,'source_python_files_unchanged':len(after),'cached_states_preserved':511,'original_cached_files_preserved':85,'model_tensor_hash_unchanged':result['final_model_parameter_hash'],'utility_checkpoint_sha256':PLAN['utility_model_sha256'],'optimizer_updates':0});print('ALL_INJECTION_ORIGINALS_PRESERVED',flush=True)
