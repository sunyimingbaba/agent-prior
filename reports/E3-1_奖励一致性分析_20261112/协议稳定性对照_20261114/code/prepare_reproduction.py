"""Prepare a fresh control run from the verified six-record legacy prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
p.add_argument('--dest',type=Path,required=True);a=p.parse_args()
assert a.dest.name.startswith('protocol_stability_20261114_reproduction_') and not a.dest.exists()
meta=json.loads(a.metadata.read_text(encoding='utf-8'));h=hashlib.sha256()
with a.archive.open('rb') as f:
    for b in iter(lambda:f.read(8388608),b''):h.update(b)
assert h.hexdigest()==meta['archive_sha256'] and a.archive.stat().st_size==meta['archive_bytes']
prefixes=('code/','inputs/','features/','baseline_evidence/','baseline_reproduction/','failed_attempt_01/','failed_attempt_02/')
names={'plan.json','实验方案.json','preflight.json','policy_settings.json','generation_code_hashes.json',
 'generation_retry_code_hashes.json','generation_v3_code_hashes.json','post_code_hashes.json',
 'generation_pipeline_result.json','generation_pipeline_result_v2.json'}
selected={}
with tarfile.open(a.archive) as t:
    members=t.getmembers();assert len(members)==meta['archive_files'] and len({x.name for x in members})==len(members)
    manifest=json.loads(t.extractfile('file_checksums.json').read())
    for m in members:
        assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        data=t.extractfile(m).read()
        if m.name!='file_checksums.json':
            assert len(data)==manifest[m.name]['bytes'] and hashlib.sha256(data).hexdigest()==manifest[m.name]['sha256'],m.name
        if m.name in names or m.name.startswith(prefixes):selected[m.name]=data
assert names<=set(selected)
assert len(selected['baseline_reproduction/records.jsonl'].splitlines())==6
assert not any(n.startswith(('pilot/','confirmation/','scoring/','generation_complete/')) for n in selected)
a.dest.mkdir(parents=True)
for name,data in selected.items():
    out=a.dest/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
(a.dest/'reproduction_source.json').write_text(json.dumps({'ok':True,'source_archive_sha256':meta['archive_sha256'],
 'copied_files':len(selected),'inherited_legacy_reproduction_records':6,'new_control_records_to_generate':620,
 'historical_failure_metadata':'preserved as provenance of inherited prefix, not outcomes of this new run',
 'next_steps':['check_reproduction_environment.py','run_generation_v3.py','run_post_protocol.py']},indent=2)+'\n',encoding='utf-8')
print('FRESH_REPRODUCTION_DIRECTORY_PREPARED',len(selected))
