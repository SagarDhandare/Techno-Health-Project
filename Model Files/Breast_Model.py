# Importing Libraries:
import pandas as pd
import numpy as np
import pickle

# for displaying all feature from dataset:
pd.pandas.set_option('display.max_columns', None)

# Reading Dataset:
dataset = pd.read_csv("Breast_data.csv")

# Dropping 'id' and 'Unnamed: 32' features:
dataset = dataset.drop(['id','Unnamed: 32'], axis=1)

# Encoding on target feature:
dataset['diagnosis'] = np.where(dataset['diagnosis']=='M', 1, 0)

# Splitting Independent and Dependent Feature:
X = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

# Feature Importance:
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model=model.fit(X,y)

# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

# Listing out highly correlated features:
correlated_features = list(correlation(dataset, 0.7))

# Dropping highly correlated features:
X = X.drop(correlated_features, axis=1)

# Train Test Split:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

# Creating a pickle file for the classifier
filename = 'Breast.pkl'
pickle.dump(RandomForest, open(filename, 'wb'))