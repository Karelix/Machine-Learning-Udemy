# Upper confidence Bound

# Importying the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Ads_CTR_Optimisation.csv')

# Implement UCB
import math
d = 10
ads_selected = []
N = [0]*d
R = [0]*d
total_reward = 0
for n in range(0,len(dataset.values[:,0])):
  max_upper_bound = 0
  ad = 0
  for i in range(0,d):
    if(N[i]>0):
      r = R[i]/N[i]
      delta_i = math.sqrt(3/2*math.log(n+1)/N[i])
      upper_bound = r + delta_i
    else:
      upper_bound = 1e400
    if upper_bound > max_upper_bound:
      max_upper_bound = upper_bound
      ad = i
  ads_selected.append(ad)
  N[ad] += 1
  reward = dataset.values[n,ad]
  R[ad] += reward
  total_reward += reward
  
# Visualising the results
plt.hist(ads_selected)
plt.title('Times an ad was selected')
plt.xlabel('Ads')
plt.ylabel('Times')
plt.show()