import pandas as pd
import numpy as np
from natsort import index_natsorted, natsorted, ns
import json
import seaborn as sns
import plotly.express as px


cellData = pd.read_csv('cellData.tsv', sep='\t')
geneData = pd.read_csv('geneData.tsv', sep='\t')
gene_panel = np.unique(geneData.Gene.values)


mylist = []
for index, row in cellData.iterrows():
    arr = json.loads(row.Prob)
    idx = np.argmax(arr)
    ClassName = eval(row.ClassName)
    mylist.append(ClassName[idx])
# out
# mylist = sorted(mylist)
df = pd.DataFrame(columns=gene_panel, index=mylist)
for i in range(df.shape[0]):
    cols = eval(cellData.iloc[i].Genenames)
    vals = eval(cellData.iloc[i].CellGeneCount)
    df.iloc[i][cols] = vals

df = df.fillna(0)
df = df.groupby(axis=0, level=0).mean()
df = df.T
df = df.sort_index(axis=1,
                   key=lambda x: np.argsort(index_natsorted(df.columns.values))
                   )

sns.heatmap(df, annot=True)

fig = px.imshow(df)
fig.show()
print(cellData)