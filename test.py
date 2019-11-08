import xlrd
import pandas as pd
import numpy as np
"""
a=np.load("logits.npy")
np.set_printoptions(precision=3)
#np.savetxt("results.txt",a)
a_e=np.exp(a)
a_sum=np.sum(a_e,axis=-1)
a_sum=a_sum[:,np.newaxis]
a_softmax=a_e/a_sum
np.savetxt("softmax_logits.txt",a_softmax,fmt="%.3f,%.3f")
"""
"""
a=np.load("results.npy")
print(a)
np.savetxt("results_digits.txt",a,fmt="%1d")
"""
import json
a=[{"answer":"我爱中国","logits":np.array([[1.0,2.0],[3.0,4.0]]).tolist(),"iscorrect":True},{"answer":"我爱中国","logits":np.array([[1.0,2.0],[3.0,4.0]]).tolist(),"iscorrect":True}]
with open("test.json","w",encoding="utf8") as f:
    for b in a:
        f.write(json.dumps(b,ensure_ascii=False)+"\n")