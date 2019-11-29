import pprint

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

warnings.filterwarnings("ignore")
import math
import statistics

# np.random.seed(123)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    xy_sum = 0
    x2_sum = 0
    y2_sum = 0
    x_sum = 0
    y_sum = 0
    for idx in range(n):
        xy_sum += x[idx] * y[idx]
        x2_sum += pow(x[idx], 2)
        y2_sum += pow(y[idx], 2)
        x_sum += x[idx]
        y_sum += y[idx]
    rxy = ((n * xy_sum) - ((x_sum) * (y_sum))) / (
                (math.sqrt((n * x2_sum) - pow(x_sum, 2))) * (math.sqrt((n * y2_sum) - pow(y_sum, 2))))
    return rxy


data = pd.read_csv('inputdata/data.csv')

with open('inputdata/data.csv') as f:
    first_line_head = f.readline().replace("\n","").split(",")


corr_matrix = []
print("Loop over header: ")
for i in(range(len(first_line_head))):
    row = []
    for j in range(len(first_line_head)):
        #print(first_line_head[i]+","+first_line_head[j])
        if j == i:
            row.append(1)
        elif i > j:
            row.append(None)
        else:
            row.append(pearson_def(data[first_line_head[i]],data[first_line_head[j]]))
    corr_matrix.append(row)

print("Custom correlation matrix:")

corr_matrix_df = pd.DataFrame(corr_matrix)
corr_matrix_df.columns=first_line_head


corr_matrix_df.insert(0,"column_name", first_line_head,True)


corr_matrix_df.set_index('column_name',inplace=True)

print("Correlation matrix DF")
print(corr_matrix_df)


corr= data.corr()

print("Standard correlation matrix is type")
print(corr)

sns.heatmap(corr)

columns = np.full(corr.shape[0], True, dtype=bool)

for i in range(corr_matrix_df.shape[0]):
    for j in range(i + 1, corr_matrix_df.shape[0]):
        if corr_matrix_df.iloc[i, j] >= 0.9:
            if columns[j]:
                columns[j] = False

selected_columns = data.columns[columns]

data = data[selected_columns]

print("Subset of data:")
print(data)
