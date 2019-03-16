# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
#np.set_printoptions(threshold = np.nan)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#labelEncoder_x = LabelEncoder()
#x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])

# Create dummy variables for the countries
columnTransformer = ColumnTransformer([('lel',OneHotEncoder(),[0])],
                                       remainder = 'passthrough')
x = columnTransformer.fit_transform(x).astype(float)

# Encoding dependent variable
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)