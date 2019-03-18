# Multiple Linear Regression
import statsmodels.formula.api as sm
def backwardElimination(x,y,sl):
  numVars = len(x[0])
  temp = np.zeros((50,6),dtype = int)
  for i in range(0,numVars):
    regressor = sm.OLS(y,x).fit()
    pval = regressor.pvalues.astype(float).tolist()
    maxVar = max(pval)
    adjR_before = regressor.rsquared_adj.astype(float)
    if maxVar > sl:
      j = pval.index(maxVar)
      temp[:,j] = x[:,j]
      x = np.delete(x,j,1)
      tmp_regressor = sm.OLS(y,x).fit()
      adjR_after = tmp_regressor.rsquared_adj.astype(float)
      if(adjR_before >= adjR_after):
        x_rollback = np.hstack((x,temp[:,[0,j]]))
        x_rollback = np.delete(x_rollback,j,1)
        print(regressor.summary())
        return x_rollback
      else:
        continue
    else:
      print(regressor.summary())
      return x
  regressor.summary()
  return x
  

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create dummy variables for the States
columnTransformer = ColumnTransformer([('lel',OneHotEncoder(),[3])],
                                       remainder = 'passthrough')
x = columnTransformer.fit_transform(x).astype(float)

# Avoiding the Dummy Variable Trap
x = x[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#x_train = sc_x.fit_transform(x_train)
#x_test = sc_x.transform(x_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Building the optimal model using Backward Elimination
#import statsmodels.formula.api as sm
# Appending column of ones for b0
x = np.append(arr = np.ones((50,1)).astype(int), values = x, 
              axis = 1)

# Initialise x_opt
x_opt = x[:, [0,1,2,3,4,5]] # x_opt = x
SL = 0.05
X_Modeled = backwardElimination(x_opt,y,SL)
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit() #Part 2
#regressor_OLS.summary()
#x_opt = x[:, [0,1,3,4,5]] # x_opt = x
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit() #Part 2
#regressor_OLS.summary()
#x_opt = x_opt[:, [0,2,3,4]] # x_opt = x
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit() #Part 2
#regressor_OLS.summary()
#x_opt = x_opt[:, [0,1,3]] # x_opt = x
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit() #Part 2
#regressor_OLS.summary()
#x_opt = x_opt[:, [0,1]] # x_opt = x
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit() #Part 2
#regressor_OLS.summary()
#
#x_opt_train, x_opt_test, y_opt_train, y_opt_test = train_test_split(x_opt,y,test_size = 0.2,random_state = 0)
#regressor_be = LinearRegression()

#regressor.fit(x_opt_train, y_opt_train)
#
## Predicting the Test set results
#y_opt_pred = regressor.predict(x_opt_test)