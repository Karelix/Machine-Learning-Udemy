# Thompson Sampling

# Importying the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
d = 10
ads_selected = []
N1 = [0]*d
N0 = [0]*d
total_reward = 0
for n in range(0,len(dataset.values[:,0])):
  max_random = 0
  ad = 0
  for i in range(0,d):
    random_beta = random.betavariate(N1[i]+1,N0[i]+1)
    if random_beta > max_random:
      max_random = random_beta
      ad = i
  ads_selected.append(ad)
  reward = dataset.values[n,ad]
  if(reward):
    N1[ad] += 1
  else:
    N0[ad] += 1
  total_reward += reward
  
# Visualising the results
plt.hist(ads_selected)
plt.title('Times an ad was selected')
plt.xlabel('Ads')
plt.ylabel('Times')
plt.show()