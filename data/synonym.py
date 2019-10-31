import pandas as pd
import random
from eda import *
df=pd.read_csv(r"D:\代码\服务器代码中转\transformer\data\train.tsv",sep='\t',header=None)
new_datas=[]
for i in range(len(df)):
    new_data=[]
    sentence=df.iloc[i,1]
    print(sentence)
    augment_sentences=eda2(sentence,0.1)
    augment_sentence=random.choice(augment_sentences[0:-1])
    print(augment_sentence)
    if augment_sentence.lower()==sentence.lower():
        continue
    new_data.append(df.iloc[i,0])
    new_data.append(augment_sentence)
    new_data.append(df.iloc[i, 2])
    new_datas.append(new_data)
df_new=pd.DataFrame(new_datas,columns=list("ABC"))
df_new.to_csv("train_synonym.tsv",header=False,index=False,sep='\t')
