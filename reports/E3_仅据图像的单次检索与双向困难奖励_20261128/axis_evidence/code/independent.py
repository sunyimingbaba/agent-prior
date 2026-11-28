"""Independently construct Gaussian derivative kernels and use OpenCV separable filtering."""
from pathlib import Path
import hashlib,io
from collections import defaultdict
import cv2
import numpy as np
import pyarrow as pa
from PIL import Image
from prepare import ROOT,read,dump,sha
def image_metric(im):
    im=np.asarray(im,dtype=np.float64)
    gray=np.einsum('ijk,k->ij',im,np.array([.299,.587,.114]))/255.
    values=[]
    for s in [1,2,4]:
        t=np.arange(-4*s,4*s+1,dtype=float)
        smooth=np.exp(-.5*(t/s)**2);smooth/=smooth.sum();deriv=t*smooth/s**2
        gx=cv2.sepFilter2D(gray,cv2.CV_64F,deriv,smooth,borderType=cv2.BORDER_REFLECT)
        gy=cv2.sepFilter2D(gray,cv2.CV_64F,smooth,deriv,borderType=cv2.BORDER_REFLECT)
        h=float(np.sum(gy**2));v=float(np.sum(gx**2))
        values.append((h-v)/(h+v) if h+v else 0.)
    return float(np.mean(values))
def main():
    cv2.setNumThreads(1)
    rows=read(ROOT/'results/逐图.json');summary=read(ROOT/'results/方向核验.json')
    tables={};max_err=0.;max_rot=0.;check=defaultdict(list)
    for r in rows:
        s=r['split']
        if s not in tables:
            p=Path('/Users/botlv/cup-deploy-assets/arrow/ucm')/('ucm_caption_karpathy_label_'+s+'.arrow')
            assert sha(p)==summary['arrow_sha256'][s]
            tables[s]=pa.ipc.open_file(pa.memory_map(str(p))).read_all()
        tbl=tables[s]; ix=r['row']
        im=Image.open(io.BytesIO(tbl['image'][ix].as_py())).convert('RGB')
        assert tbl['path'][ix].as_py()==r['image'] and hashlib.sha256(im.tobytes()).hexdigest()==r['rgb']
        pred=image_metric(im); reverse=image_metric(im.transpose(Image.Transpose.ROTATE_90))
        max_err=max(max_err,abs(pred-r['margin']),abs(reverse-r['rotated_margin']));max_rot=max(max_rot,abs(pred+reverse))
        assert abs(pred)>1e-12 and (pred>0)==(r['axis']=='horizontal') and pred*reverse<0 and max_err<1e-12 and max_rot<1e-12
        check[r['cohort'],r['pair']].append(float((pred>0)==(r['label']=='horizontal')))
    for name,reported in summary['summary'].items():
        y=np.array([np.mean(v) for (n,_),v in check.items() if n==name]); n=len(y);rng=np.random.default_rng(281128)
        draws=rng.integers(0,n,size=(20000,n));freq=np.stack([np.bincount(i,minlength=n) for i in draws])
        assert reported['balanced_image_accuracy']['mean']==float(y.mean()) and np.array_equal(np.quantile((freq@y)/n,[.025,.975]),reported['balanced_image_accuracy']['interval'])
    dump(ROOT/'results/独立复算.json',dict(backend='OpenCV sepFilter2D',opencv=cv2.__version__,observations=len(rows),grouped_pairs=len(check),
          max_margin_abs_error=max_err,max_rotation_abs_error=max_rot,all_predictions_identical=True,all_pair_intervals_replicated=True))
if __name__=="__main__":main()
