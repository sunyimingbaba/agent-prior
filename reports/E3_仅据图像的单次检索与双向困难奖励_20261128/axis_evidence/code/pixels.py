"""Spatial structure-tensor orientation, independent of the image identity and the task labels."""
import numpy as np
from scipy.ndimage import gaussian_filter

def measure(rgb):
    x=np.asarray(rgb,dtype=np.float64)
    if x.ndim!=3 or x.shape[2]!=3:raise ValueError("expected RGB")
    gray=(0.299*x[:,:,0]+0.587*x[:,:,1]+0.114*x[:,:,2])/255.
    scales=[]
    for sigma in (1.,2.,4.):
        gx=gaussian_filter(gray,sigma,order=(0,1),mode="reflect",truncate=4.)
        gy=gaussian_filter(gray,sigma,order=(1,0),mode="reflect",truncate=4.)
        jxx=np.square(gx).mean();jyy=np.square(gy).mean()
        scales.append(float((jyy-jxx)/(jyy+jxx)) if (jyy+jxx)>0 else 0.)
    s=float(np.mean(scales))
    axis="tie" if abs(s)<=1e-12 else ("horizontal" if s>0 else "vertical")
    return dict(margin=s,scale_margins=scales,axis=axis)
