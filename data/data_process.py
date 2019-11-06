import xlrd
import pandas as pd
from sklearn.model_selection import train_test_split
data=xlrd.open_workbook(r"D:\代码\服务器代码中转\transformer\data\langying_20191031_score_list_part1.result.xlsx")
text=pd.read_csv(r"D:\代码\服务器代码中转\transformer\data\langying_20191031_ref_list_part1.format.txt",sep="\t")
text=text.values
table = data.sheets()[0]
nrows = table.nrows
process_datas=[]
for i,t in enumerate(text):
    for j in range(1,nrows):
        if str(t[0]) in table.row_values(j)[6]:
            refs=table.row_values(j)[5].split("|")
            for ref in refs:
                process_data = []
                process_data.append(t[1])
                process_data.append(table.row_values(j)[4].lower())
                process_data.append(ref.lower())
                process_data.append(int(table.row_values(j)[2]>=0.5))
                print(process_data)
                process_datas.append(process_data)
df_new=pd.DataFrame(process_datas)
df_new.to_csv("train_real.tsv",header=False,index=False,sep='\t')
df_new=pd.read_csv(r"D:\代码\服务器代码中转\transformer\data\train_real.tsv",header=None,sep="\t")
X_train, X_test=train_test_split(df_new,test_size=0.2,shuffle=True,random_state=0)
X_train.to_csv("train_real_split.tsv",header=False,index=False,sep='\t')
X_test.to_csv("dev_real_split.tsv",header=False,index=False,sep='\t')