"""Validate completed frozen-policy records and preserve a resumable pause."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import torch
torch.set_num_threads(2)
r=Path(__file__).resolve().parent.parent;assert str(r).startswith('/private/')
assert json.loads((r/'pause_status.json').read_text())['status']=='paused'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def tensor_sha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

plan=json.loads((r/'plan.json').read_text());cases=json.loads((r/'inputs/cases.json').read_text())
expected=[]
for i,item in enumerate(cases):
    expected.append(f"all__{item['split']}_{item['row']}_greedy_0")
    expected += [f"all__{item['split']}_{item['row']}_candidate_{800000+i*4+g}" for g in range(4)]
assert len(expected)==500
out=r/'pause_snapshot';out.mkdir(exist_ok=False)
records=[];truncated=[];checked=0;registered=set()
for mode in ['generation','smoke']:
    path=r/mode/'records.jsonl';lines=path.read_bytes().splitlines(keepends=True);rows=[]
    for i,line in enumerate(lines):
        try:x=json.loads(line)
        except json.JSONDecodeError:
            assert i==len(lines)-1
            (out/(mode+'_incomplete_tail.bin')).write_bytes(line);truncated.append(mode);continue
        h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True)
        assert h.dtype==torch.bfloat16 and h.shape==(3584,) and torch.isfinite(h).all()
        assert sha(r/x['state_file'])==x['state_file_sha256']
        assert tensor_sha(h)==x['generation_state_sha256']==x['cache_state_sha256']
        assert sha(r/x['trace_file'])==x['trace_sha256']
        trace=json.loads((r/x['trace_file']).read_text());assert trace['state_tensor_sha256']==x['generation_state_sha256']
        assert x['quality_usable']==trace['quality_usable'];rows.append(x);checked+=1
        registered.update([x['state_file'],x['trace_file']])
    if mode=='generation':
        assert [x['id'] for x in rows]==expected[:len(rows)];records=rows
        (out/'validated_completed_records.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
assert len(records)<500
orphans=[str(p.relative_to(r)) for d in ['generation/priors','generation/traces'] for p in (r/d).glob('*') if p.is_file() and str(p.relative_to(r)) not in registered]
ref=plan['policy'];checkpoint=Path(ref['checkpoint'])
assert sha(checkpoint/'adapter_model.safetensors')==ref['adapter_sha256']
assert sha(Path(ref['source_remote_archive']))==ref['source_archive_sha256']
for name in ['gate_code_hashes.json','pipeline_code_hashes.json','post_code_hashes.json']:
    for file,h in json.loads((r/name).read_text()).items():assert sha(r/file)==h
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
status=json.loads((r/'pause_status.json').read_text());status.update(completed_full_run_records=len(records),planned_full_run_records=500,
    remaining_full_run_records=500-len(records),next_record_id=expected[len(records)],completed_image_groups=len(records)//5,
    partial_next_image_records=len(records)%5,cache_and_trace_records_verified=checked,truncated_jsonl_tail_modes=truncated,
    uncommitted_orphan_files=orphans,plan_sha256=sha(r/'plan.json'),gpu_after_stop=gpu)
dump(r/'pause_status.json',status)
dump(out/'integrity.json',{'ok':True,'completed_records_form_contiguous_expected_prefix':True,'cache_and_trace_records_checked':checked,
    'completed_generation_records':len(records),'uncommitted_orphans':orphans,'warmstart_checkpoint_and_source_archive_verified':True,
    'all_executed_code_hashes_unchanged':True,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
note=f'''# 数量冲突奖励诊断：暂停与续跑

用户请求暂停，生成任务及后处理等待进程已停止，服务器由用户自行关闭。

- 已完成新轨迹：{len(records)}/500；剩余：{500-len(records)}。
- 完整图像组：{len(records)//5}；下一图像已有记录：{len(records)%5}/5。
- 下一条编号：`{expected[len(records)]}`。
- 本阶段策略权重冻结，优化器更新数为 0。
- 每条轨迹按固定种子独立生成；使用原始方案、同一 FP32 适配器恢复与同一共享工具流程续跑。
- 完整方案摘要：`{sha(r/'plan.json')}`。

## 续跑顺序

1. 用户开启服务器后，核对 `/private` 挂载、现有缓存与 `pause_snapshot/integrity.json`。
2. 原始 `generation/records.jsonl` 保持原文件；完整有效前缀另存于 `pause_snapshot/validated_completed_records.jsonl`。
3. 编写独立续跑入口，按既定 500 条编号列表跳过已保存编号，按原 image_order 和种子补齐剩余轨迹。原生成入口采用新建目录检查，直接重启会触发已有目录保护。
4. 对未提交缓存和可能的 JSONL 尾部保留备份后处理。孤立文件列表与尾部完整性记录保存在 `pause_status.json`。
5. 补齐后运行原 `score_quantity.py`，再依次执行奖励重放、统计和独立核验。重新启动后处理等待进程。

此前 100 图的图像哈希、400 个历史数量分数、文本特征与软件检查均已通过。当前阶段的正式数量统计和训练建议留待全部轨迹完成后给出。

原策略模型、数据和环境保留在既有私有目录；格式模型原始归档在上阶段已完成本机与服务器备份。模型位置和哈希见 `inputs/model_reference.json`。
'''
(r/'暂停与续跑说明_20261114.md').write_text(note)
paths=[p for p in sorted(r.rglob('*')) if p.is_file() and p.name!='pause_file_checksums.json' and not any(x in p.relative_to(r).parts for x in ['scratch','__pycache__'])]
manifest={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths};dump(r/'pause_file_checksums.json',manifest)
archive=r.parent/'quantity_pause_evidence_20261114.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
    for p in paths+[r/'pause_file_checksums.json']:t.add(p,arcname=str(p.relative_to(r)))
backup={'archive_name':archive.name,'remote_archive':str(archive),'remote_root':str(r),'archive_sha256':sha(archive),
    'archive_bytes':archive.stat().st_size,'archive_files':len(paths)+1,'manifest_files':len(paths),
    'completed_generation_records':len(records),'remaining_generation_records':500-len(records),
    'source_policy_archive_sha256':ref['source_archive_sha256'],'status':'paused','gpu_after_stop':gpu,
    'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dump(r.parent/'quantity_pause_backup_20261114.json',backup)
print(json.dumps({'pause':status,'backup':backup},ensure_ascii=False))
