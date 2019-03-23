# Hierarchical Clustering

#%reset -f

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Mall_Customers.csv')
x = dataset.iloc[:, 3:].values
