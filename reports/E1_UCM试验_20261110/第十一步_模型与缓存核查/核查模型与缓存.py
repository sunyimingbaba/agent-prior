#!/usr/bin/env python3
"""Read-only audit of image/Arrow/prior alignment for the user-reviewed sample set."""
import argparse
import collections
import csv
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import pyarrow as pa
import torch
from PIL import Image

parser = argparse.ArgumentParser()
for name in ['manifest', 'arrow-dir', 'cache-dir', 'images-dir', 'archive', 'agse-model', 'clip-model', 'output-dir']:
    parser.add_argument('--'+name, type=Path, required=True)
a = parser.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=True)

def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rgb_digest(image):
    return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()

with a.manifest.open(encoding='utf-8-sig', newline='') as f:
    samples = list(csv.DictReader(f))
assert len(samples) == 100 and len({r['图像编号'] for r in samples}) == 100
rows_by_split = {}
tables = {}
for split in ('train', 'val'):
    path = a.arrow_dir / f'ucm_caption_karpathy_label_{split}.arrow'
    table = pa.ipc.open_file(str(path)).read_all()
    names = table['path'].to_pylist()
    assert len(names) == len(set(names)), f'duplicate Arrow paths in {split}'
    rows_by_split[split] = {name: i for i, name in enumerate(names)}
    tables[split] = table

word = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven'}
results = []
with tarfile.open(a.archive, mode='r:gz') as archive:
    for row in samples:
        name = row['图像编号']
        split = row['数据来源'].split()[-1]
        idx = rows_by_split[split][name]
        table = tables[split]
        caption = table['caption'][idx].as_py()[0]
        image_bytes = table['image'][idx].as_py()
        with Image.open(io.BytesIO(image_bytes)) as arrow_image, Image.open(a.images_dir / name) as tiff_image:
            img_sha = rgb_digest(arrow_image)
            img_match = img_sha == rgb_digest(tiff_image)
        prior_name = f'ucm_caption_karpathy_label_{split}/{idx}.pt'
        prior_path = a.cache_dir / prior_name
        prior = torch.load(prior_path, map_location='cpu', weights_only=True)
        tensor_ok = isinstance(prior, torch.Tensor) and prior.numel() == 3584
        finite_ok = bool(torch.isfinite(prior).all()) if tensor_ok else False
        member = archive.getmember('agent_cache_domain_tool/' + prior_name)
        archive_bytes = archive.extractfile(member).read()
        prior_sha = digest_file(prior_path)
        count = int(row['用户确认数量'])
        num_match = bool(re.search(r'\b'+word[count]+r'\b', caption, re.IGNORECASE))
        results.append({'图像编号':name, '来源':split, 'Arrow行号':idx, '图像像素匹配':img_match,
                        '第一条描述匹配清单':caption == row['原始第一条描述'],
                        '先验文件':prior_name, '张量形状':str(tuple(prior.shape)),
                        '张量dtype':str(prior.dtype), '张量3584维':tensor_ok, '张量数值有限':finite_ok,
                        '先验与传输归档一致':prior_sha == hashlib.sha256(archive_bytes).hexdigest(),
                        '先验SHA256':prior_sha, '图像RGB_SHA256':img_sha,
                        '人工确认数量':count, '第一条描述出现对应英文数词':num_match,
                        '第一条描述':caption})
with (a.output_dir/'100图像与先验核查.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
checks = ['图像像素匹配','第一条描述匹配清单','张量3584维','张量数值有限','先验与传输归档一致']
summary = {'sample_count':len(results),
           'split_counts':dict(collections.Counter(r['来源'] for r in results)),
           'arrow_total_rows':{k:v.num_rows for k,v in tables.items()},
           'checks':{name:sum(bool(r[name]) for r in results) for name in checks},
           'shapes':dict(collections.Counter(r['张量形状'] for r in results)),
           'dtypes':dict(collections.Counter(r['张量dtype'] for r in results)),
           'first_caption_contains_true_number_word':sum(r['第一条描述出现对应英文数词'] for r in results),
           'model_files':{str(path):{'bytes':path.stat().st_size,'sha256':digest_file(path)} for path in (a.agse_model,a.clip_model)},
           'archive_sha256':digest_file(a.archive)}
(a.output_dir/'核查汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
