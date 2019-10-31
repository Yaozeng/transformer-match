import pandas as pd
df1=pd.read_csv(r"D:\代码\服务器代码中转\transformer\data\train_bt.tsv",header=None,sep='\t')
df2=pd.read_csv(r"D:\代码\服务器代码中转\transformer\data\train_synonym.tsv",header=None,sep='\t')
df3=pd.read_csv(r"D:\代码\服务器代码中转\transformer\data\train.tsv",header=None,sep='\t')

df=pd.concat([df1,df2,df3],ignore_index=True)
df.to_csv("train_augment.tsv",header=False,index=False,sep='\t')