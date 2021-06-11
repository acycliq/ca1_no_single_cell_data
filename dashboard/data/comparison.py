"""
compares the data calling results with these obtained when scRNAseq data were available.
"""
import pandas as pd
import numpy as np
import json
import os

CA1_ROOT = 'D:\\Home\\Dimitris\\OneDrive - University College London\\dev\\Python\\ca1\\viewer\\data'

def run():
    ca1_data_str = os.path.join(CA1_ROOT, 'cellData.tsv')
    ca1_cellData = pd.read_csv(ca1_data_str, sep='\t')
    ca1_geneData = pd.read_csv('geneData.tsv', sep='\t')
    cellData = pd.read_csv('cellData.tsv', sep='\t')

    # get the gene panel (sorted)
    genes = np.unique(ca1_geneData.Gene)
    class_names = [d+'_class' for d in genes]
    class_names.append('Zero')
    m = cellData.shape[0]
    n = len(class_names)
    out = pd.DataFrame(np.zeros([m, n]), columns=class_names)


    actual_list = []
    for i, row in ca1_cellData.iterrows():
        # 1. Find the most likely class as this is given by the ca1 data with proper scRNAseq
        arr = json.loads(row.Prob)
        argmax = np.argmax(arr)
        ca1_class = eval(row.ClassName)
        actual = ca1_class[argmax]
        actual_list.append(actual)

        # 2 For the same cell find the cell call results under unknown scRNAseq data
        # Do some sanity checking first
        assert row['X'] == cellData.X_0[i]
        assert row['Y'] == cellData.Y_0[i]

        cols = eval(cellData.ClassName[i])
        vals = eval(cellData.Prob[i])
        out.iloc[i][cols] = vals

    # make an extra column with the actual classes
    out['actual'] = actual_list
    # set it as the index of the df
    out = out.set_index('actual')
    # group by actual class
    res = out.groupby(out.index.values).agg('mean')
    # save the confusion matrix to csv
    res.to_csv('confusion_matrix.csv')
    print('Done')


if __name__ == "__main__":
    run()