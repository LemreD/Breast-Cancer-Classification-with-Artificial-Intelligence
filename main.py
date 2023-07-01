import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis,LocalOutlierFactor
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("data.csv")
""" fazla kısmın çıkarılması"""
data.drop(['Unnamed: 32','id'],inplace= True,axis = 1) 
data = data.rename(columns= {"diagnosis":"target"})
"""iyi ve kötü huylu grafiksel gösterim"""
sns.countplot(data["target"])
print(data.target.value_counts())

data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]

print(len(data))
print("data shape",data.shape)
"""
standardlaştırma

"""
"""string değerler için"""
corr_matrix = data.corr() 
"""ağaç gösterimi"""
sns.clustermap(corr_matrix, annot = True, fmt= ".2f")
plt.title("correlation between features")
plt.show()
