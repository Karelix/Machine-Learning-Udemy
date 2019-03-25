# Artificial Neural Networks (ANN)

# Part 1 - Data Preprocessing

# Importing the libraries
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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Part 2 - Making of Ann

# Importing Keras 
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding Input layer and the first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform', activation='relu',input_shape=(11,)))

# Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(x=x_train,y=y_train,batch_size=10,epochs=100)

# Predicting the Test set Results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)