# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def clean_string(string,stopWords):
  review = re.sub('[^a-zA-Z]',' ',string) #only letters
  review = review.lower() #make lowercase
  review = review.split() #make list from the words
  ps = PorterStemmer() #object for stemming
  #purge all not necessary words and perform stemming
  review = [ps.stem(word) for word in review if not word in stopWords]
  review = ' '.join(review)
  return review

# Importing the dataset
dataset = pd.read_csv('../Restaurant_Reviews.tsv',delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
unwanted = ('no','not')
stopWords = set([word for word in set(stopwords.words('english')) if word not in unwanted])
for i in range(0,len(dataset['Review'])):
  corpus.append(clean_string(dataset['Review'][i],stopWords))
  
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1,
                                                    random_state = 0)

#Naive Bayes
# Fitting the Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

# Predicting the Test set Results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Decision Tree Classification
# Fitting the Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
tree_classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
tree_classifier.fit(x_train,y_train)

# Predicting the Test set Results
y_pred_tree = tree_classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_tree = confusion_matrix(y_test,y_pred_tree)

# Random Forest Classification
# Fitting the Decision Forest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
forest_classifier.fit(x_train,y_train)

# Predicting the Test set Results
y_pred_forest = forest_classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_forest = confusion_matrix(y_test,y_pred_forest)