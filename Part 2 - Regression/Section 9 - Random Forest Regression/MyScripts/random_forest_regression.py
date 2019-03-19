# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Position_Salaries.csv')
plt.figure()
plt.scatter(dataset.iloc[:,1].values,dataset.iloc[:,2].values,color = 'magenta')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
#                                                    random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#x_train = sc_x.fit_transform(x_train)
#x_test = sc_x.transform(x_test)

# Fitting Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
rf_regressor.fit(x,y)

# Predict a new result with Random Forest Regression
#level = 6.5
y_pred = rf_regressor.predict([[6.5]])

# Visualising the Regression results
x_grid = np.arange(min(x),max(x),0.001)
x_grid = x_grid.reshape((len(x_grid),1))
plt.figure()
plt.scatter(x,y,color='red')
plt.plot(x_grid,rf_regressor.predict(x_grid),color='magenta')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()