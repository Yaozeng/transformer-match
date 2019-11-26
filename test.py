import xlrd
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
from modeling_tf_roberta import TFRobertaForSequenceClassification
from configuration_roberta import RobertaConfig
from tokenization_roberta import RobertaTokenizer
import h5py
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
"""
import json
a=[{"answer":"我爱中国","logits":np.array([[1.0,2.0],[3.0,4.0]]).tolist(),"iscorrect":str(True)},{"answer":"我爱中国","logits":np.array([[1.0,2.0],[3.0,4.0]]).tolist(),"iscorrect":str(True)}]
with open("test.json","w",encoding="utf8") as f:
    for b in a:
        f.write(json.dumps(b,ensure_ascii=False)+"\n")
        """
"""
a=np.array([1,0,1])
b=np.array([1,1,1])
print(f1_score(a,b))
"""
import csv
import sys
lines=[]
with open("data/sts-testset-20191112-fix.tsv", "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    for line in reader:
        if sys.version_info[0] == 2:
            line = list(unicode(cell, 'utf-8') for cell in line)
        lines.append(line)
examples = []
for (i, line) in enumerate(lines):
    if i == 0:
        continue
    guid = "%s-%s" % ("dev", i)
    text_a = line[3]
    text_b = line[4]
    label = line[5]
    print(text_a)
    print(text_b)
    print(label)
