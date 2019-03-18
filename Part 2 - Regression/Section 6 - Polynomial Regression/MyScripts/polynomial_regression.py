# Polynomial Regression

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

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# Visualising the Linear Regression vs the Polynomial Regression results
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.figure()
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='magenta')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff (Linear Regression vs Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

level = 6.5
# Predicting a new result with Linear Regression
lin_reg.predict([[level]])

# Predict a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[level]]))