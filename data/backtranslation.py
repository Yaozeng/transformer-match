import pandas as pd
import csv
import sys
df1=pd.read_csv(r"D:\代码\服务器代码中转\transformer\data\train.tsv",header=None,sep='\t')
lines = []
with open(r"D:\代码\服务器代码中转\transformer\data\train_new.txt", "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    for line in reader:
        if sys.version_info[0] == 2:
            line = list(unicode(cell, 'utf-8') for cell in line)
        lines.append(line)
print(len(lines))
new_datas=[]
for i in range(len(df1)):
    new_data = []
    sentence1 = df1.iloc[i, 1]
    sentence2=lines[i][0]
    print(sentence1)
    print(sentence2)
    if sentence1.lower() == sentence2.lower():
        continue
    new_data.append(df1.iloc[i, 0])
    new_data.append(sentence2)
    new_data.append(df1.iloc[i, 2])
    new_datas.append(new_data)
print(new_datas)
df_new=pd.DataFrame(new_datas,columns=list("ABC"))
df_new.to_csv("train_bt.tsv",header=False,index=False,sep='\t')