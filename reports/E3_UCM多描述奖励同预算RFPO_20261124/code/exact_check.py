"""Independent integer-lattice reconstruction of grouped reciprocal ranks."""
import math
import numpy as np
from scipy.stats import rankdata
def exact(order,ranks):
    denom=[int(x)+1 for x in ranks];lcm=math.lcm(*denom)
    ticks=sorted(lcm//x for x in denom)
    loc=rankdata(order,method='min').astype(int)-1
    out=np.empty(len(order))
    for start in sorted(set(loc)):
        mask=loc==start;n=int(mask.sum());out[mask]=sum(ticks[start:start+n])/(n*lcm)
    return out
