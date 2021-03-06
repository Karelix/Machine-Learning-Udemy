# Decision Tree Regression

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

# Fitting Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state = 0)
dt_regressor.fit(x,y)

# Predict a new result with Polynomial Regression
#level = 6.5
y_pred = dt_regressor.predict([[6.5]])

# Visualising the Regression results NOT GOOD
plt.figure()
plt.scatter(x,y,color='red')
plt.plot(x,dt_regressor.predict(x),color='magenta')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (Smoother)
x_grid = np.arange(min(x),max(x),0.0001)
x_grid = x_grid.reshape((len(x_grid),1))
plt.figure()
plt.scatter(x,y,color='red')
plt.plot(x_grid,dt_regressor.predict(x_grid),color='magenta')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()