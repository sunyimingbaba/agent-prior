import argparse
import datetime
from pathlib import Path
import tarfile
from common import ROOT,read,dump,sha
p=argparse.ArgumentParser();p.add_argument('--name',default='multineg_quantity_offline_evidence_20261115.tar.gz');a=p.parse_args()
assert read(ROOT/'独立核验.json')['ok'] and read(ROOT/'打包材料重放核验.json')['ok']
for n,h in read(ROOT/'code_hashes.json').items():assert sha(ROOT/n)==h,n
skip={'file_checksums.json','备份校验.json'}
paths=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not set(p.relative_to(ROOT).parts)&{'scratch','__pycache__'} and p.relative_to(ROOT).parts[0]!='原始运行记录' and str(p.relative_to(ROOT)) not in skip and not p.name.endswith('.tar.gz')]
required={'source_data/'+n for n in read(ROOT/'plan.json')['input_files'] if not n.endswith('.tar.gz')}
assert required <= {str(p.relative_to(ROOT)) for p in paths},'Missing frozen source inputs'
manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths};dump(ROOT/'file_checksums.json',manifest)
archive=ROOT/a.name;assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=6) as t:
    for p in paths+[ROOT/'file_checksums.json']:t.add(p,arcname=str(p.relative_to(ROOT)))
# Read the archive back and validate every included source and derived output.
with tarfile.open(archive) as t:
    members=t.getmembers();assert len(members)==len(manifest)+1 and len({x.name for x in members})==len(members)
    for m in members:
        assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        data=t.extractfile(m).read()
        assert data==(ROOT/m.name).read_bytes(),m.name
backup={'archive_name':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'archive_files':len(manifest)+1,'manifest_files':len(manifest),'local_archive_verification':'passed','remote_backup_status':'pending','new_optimizer_updates':0,'new_generated_trajectories':0,'parent_a100_evidence_sha256':read(ROOT/'plan.json')['parent_evidence_sha256'],'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dump(ROOT/'备份校验.json',backup);print('OFFLINE_ARCHIVE_VERIFIED',len(manifest),archive.stat().st_size)
