import json
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
"""
correct_one=0
correct_all=0
all_sample=0
for line in open(r"D:\代码\服务器代码中转\transformer\output\results_roberta.json", 'r'):
    a=json.loads(line)
    all_sample+=1
    logits=a["logits"]
    label=a["label"]
    logits_one=logits[0]
    print(logits_one)
    labels=[label]*len(logits)
    correct_all_result=np.argmax(np.array(logits),axis=-1)
    correct_one_result = np.argmax(np.array(logits_one), axis=-1)
    print(correct_one_result)
    correct_all+=int(np.any(correct_all_result==np.array(labels)))
    correct_one+=int(correct_one_result==np.array(label))
print(correct_one/all_sample)
print(correct_all/all_sample)
"""


