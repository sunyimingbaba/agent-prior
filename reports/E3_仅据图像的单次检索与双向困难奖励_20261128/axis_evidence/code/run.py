"""Serialize NumPy integer counts while retaining the prespecified estimators unchanged."""
import json
import numpy as np
import check
def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def write(path,data):
    path.write_text(json.dumps(data,default=scalar,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
if __name__=="__main__":
    check.dump=write
    check.main()
