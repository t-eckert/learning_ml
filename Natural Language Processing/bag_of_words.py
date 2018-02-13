import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

ps = PorterStemmer()

corpus = []

for i in range(len(dataset['Review'])):
  review = re.sub("[^a-zA-Z]", " ", dataset['Review'][i]).lower().split()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)