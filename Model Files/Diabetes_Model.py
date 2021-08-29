# Importing Libraries:
import pandas as pd
import numpy as np
import pickle

# for displaying all feature from dataset:
pd.pandas.set_option('display.max_columns', None)

# Reading Dataset:
dataset = pd.read_csv("Diabetes_data.csv")

# Replacing Zero values with Median:
dataset['Glucose'] = np.where(dataset['Glucose']==0, dataset['Glucose'].mean(), dataset['Glucose'])
dataset['BloodPressure'] = np.where(dataset['BloodPressure']==0, dataset['BloodPressure'].mean(), dataset['BloodPressure'])
dataset['SkinThickness'] = np.where(dataset['SkinThickness']==0, dataset['SkinThickness'].mean(), dataset['SkinThickness'])
dataset['Insulin'] = np.where(dataset['Insulin']==0, dataset['Insulin'].mean(), dataset['Insulin'])
dataset['BMI'] = np.where(dataset['BMI']==0, dataset['BMI'].mean(), dataset['BMI'])

# Independent and Dependent Feature:
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Train Test Split:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

# Creating a pickle file for the classifier
filename = 'Diabetes.pkl'
pickle.dump(RandomForest, open(filename, 'wb'))