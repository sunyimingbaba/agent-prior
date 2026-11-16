"""Back up the completed immutable cache while readout training continues."""
from pathlib import Path
import datetime
import tarfile
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
assert str(r).startswith('/private/AGSE_E3_1_rebuild_20261112/image_only_matched_readout_20261116')
assert read(r/'cache_audit.json')['ok'] and read(r/'cache_result.json')['states']==2100
for n,h in read(r/'cache_export_code_hashes.json').items():assert sha(r/n)==h,n
paths=set(p for p in (r/'cache').rglob('*') if p.is_file())
paths.update(r/n for n in ['cache_plan.json','cache_result.json','cache_audit.json','cache_pipeline_result.json',
                         'cache_code_hashes.json','code/verify_cache.py'])
paths.update(r/n for n in read(r/'cache_code_hashes.json'))
paths=sorted(paths)
j={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}
dump(r/'cache_export_checksums.json',j)
a=r.parent/'image_only_cache_2100_evidence_20261116.tar.gz'
assert not a.exists()
with tarfile.open(a,'w:gz',compresslevel=3) as t:
    for p in paths+[r/'cache_export_checksums.json']:t.add(p,arcname=str(p.relative_to(r)))
metadata={'archive_name':a.name,'remote_archive':str(a),'archive_bytes':a.stat().st_size,
          'archive_sha256':sha(a),'archive_files':len(paths)+1,'manifest_files':len(paths),
          'cache_states':2100,'readout_training_continues_separately':True,
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dump(r.parent/(a.name+'.json'),metadata)
print('COMPLETED_CACHE_EVIDENCE_EXPORTED',len(paths),a.stat().st_size,flush=True)
