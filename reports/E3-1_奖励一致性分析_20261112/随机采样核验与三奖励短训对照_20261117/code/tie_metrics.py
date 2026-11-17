"""Expected strict hits under uniform selection within each exact float32 score tie."""
import numpy as np

def expected_hits(scores,positive):
 result=[]
 for k in [1,5,10]:
  threshold=np.partition(scores,-k,axis=1)[:,-k,None]
  equal=scores==threshold;above=scores>threshold
  total=equal.sum(1);good=(positive & equal).sum(1);take=k-above.sum(1)
  miss=np.ones(scores.shape[0])
  for i in range(k):
   miss*=np.where(i<take,np.maximum(total-good-i,0)/np.maximum(total-i,1),1.)
  result.append(np.where((positive & above).any(1),1.,1.-miss))
 return np.stack(result,1)
