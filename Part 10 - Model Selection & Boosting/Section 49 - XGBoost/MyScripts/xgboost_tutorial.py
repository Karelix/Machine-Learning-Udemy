# XGBoost

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
genderencoder = LabelEncoder()
x[:,2] = genderencoder.fit_transform(x[:,2])
columnTransformer = ColumnTransformer([('lel',OneHotEncoder(),[1])],
                                       remainder = 'passthrough')
x = columnTransformer.fit_transform(x).astype(float)

# Avoiding Dummy Variable Trap
x = x[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,
                                                    random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)

# Predicting the Test set Results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()