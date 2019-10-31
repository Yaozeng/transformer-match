import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import sys

lines = []
with open("./deliver_qa.out.fix.tsv", "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    for i,line in enumerate(reader):
        if(i==0):
            continue
        example = []
        if sys.version_info[0] == 2:
            line = list(unicode(cell, 'utf-8') for cell in line)
        example.append(line[3])
        example.append(line[4])
        example.append(line[5])
        lines.append(example)
with open("./badcase.to20190711.txt", "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    for i, line in enumerate(reader):
        if (i == 0):
            continue
        example=[]
        if sys.version_info[0] == 2:
            line = list(unicode(cell, 'utf-8') for cell in line)
        example.append(line[0])
        example.append(line[1])
        example.append(line[2])
        lines.append(example)

df=pd.DataFrame(lines,columns=list("ABC"))
X_train, X_test=train_test_split(df,test_size=0.2,stratify=df['C'],shuffle=True)
X_train.to_csv("train.tsv",header=False,index=False,sep='\t')
X_test.to_csv("dev.tsv",header=False,index=False,sep='\t')
print(X_train["A"])
print(X_train)
print(len(X_train))
print(len(X_train[X_train["C"]=="0"])/len(X_train[X_train["C"]=="1"]))