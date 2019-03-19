# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Position_Salaries.csv')
#plt.figure()
#plt.scatter(dataset.iloc[:,1].values,dataset.iloc[:,2].values,color = 'magenta')
#plt.title('Salary vs Level')
#plt.xlabel('Level')
#plt.ylabel('Salary')
#plt.show()


x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
#                                                    random_state = 0)

# Feature Scaling ## WE NEED IT FOR SVR!!!
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1))
y = y.reshape(len(y[:,0]))

# Fitting SVR to the dataset
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf',gamma = 'auto',degree = 5)
svr_regressor.fit(x,y)


# Predict a new result with SVR
#level = 6.5
y_pred = sc_y.inverse_transform(svr_regressor.predict(sc_x.transform(np.array([[6.5]]))))

# Visualising the SVRn results
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.figure()
plt.scatter(x,y,color='red')
plt.plot(x_grid,svr_regressor.predict(x_grid),color='magenta')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()