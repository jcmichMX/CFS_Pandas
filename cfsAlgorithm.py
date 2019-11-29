import pprint

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

np.random.seed(123)

data = pd.read_csv('inputdata/data.csv')

data = data.iloc[:,1:-1]

label_encoder = LabelEncoder()
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')


corr= data.corr()

sns.heatmap(corr)

columns = np.full(corr.shape[0], True,dtype=bool)

for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.9:
                if columns[j]:
                    columns[j]= False

selected_columns = data.columns[columns]

data = data[selected_columns]

print("Selected columns:")
print(selected_columns)


print("Subset of data:")
print(data)