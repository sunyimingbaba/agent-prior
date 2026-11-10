import argparse, ast, csv, hashlib
from pathlib import Path
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Verify matching image regions and image hashes.')
parser.add_argument('--images-dir', type=Path, required=True)
parser.add_argument('--evidence', type=Path, default=Path(__file__).with_name('重叠证据.csv'))
args = parser.parse_args()
with args.evidence.open(encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        a = args.images_dir / row['图像A']
        b = args.images_dir / row['图像B']
        assert hashlib.sha256(a.read_bytes()).hexdigest() == row['A图SHA256']
        assert hashlib.sha256(b.read_bytes()).hexdigest() == row['B图SHA256']
        with Image.open(a) as ia, Image.open(b) as ib:
            aa = np.asarray(ia.convert('RGB').crop(ast.literal_eval(row['A坐标_左上右下'])))
            bb = np.asarray(ib.convert('RGB').crop(ast.literal_eval(row['B坐标_左上右下'])))
            assert np.array_equal(aa, bb), (row['图像A'], row['图像B'])
        print(f"{row['图像A']} / {row['图像B']}: {aa.shape[1]}×{aa.shape[0]} pixels identical")
