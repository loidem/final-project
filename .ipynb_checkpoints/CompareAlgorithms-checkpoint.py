import os
import sys
import re
import random
import math
import matplotlib
import pandas as pd
import numpy as np
import ipaddress as ip
from os.path import split
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import sklearn.ensemble as ek
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import model_selection, tree, linear_model
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import tree, DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
#matplotlib inline


#custom tokenizer for URLs. 
#first split - "/"
#second split - "-"
#third split - "."
#remove ".com" (also "http://", but we dont have "http://" in our dataset)
def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens

#function to remove "http://" from URL
def trim(url):
    return re.match(r'(?:\w*://)?(?:.*\.)?([a-zA-Z-1-9]*\.[a-zA-Z]{1,}).*', url).groups()[0]

df = pd.read_csv(r'C:\Users\Kephas\Desktop\FinalProject\dataset.csv',',',error_bad_lines=False)
df = df.sample(frac=1)
df = df.sample(frac=1).reset_index(drop=True)
df.head()

 #displaying 5 records

len(df)


 #data['url'].values
x = df.iloc[:,0:1].values
y = df.iloc[:,1].values

#convert it into numpy array and shuffle the dataset
df = np.array(df)
random.shuffle(df)

#convert text data into numerical data for machine learning models
y = [d[1] for d in df]
corpus = [d[0] for d in df]
vectorizer = TfidfVectorizer(tokenizer=getTokens)
X = vectorizer.fit_transform(corpus)

#split the data set inot train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


#prepare the model
lgr = LogisticRegression()
lgr.fit(X_train, y_train)


#make model prediction for the testing class

y_pred_class = lgr.predict(X_test)
print("Accuracy for LRG",lgr.score(X_test, y_test))


Dt = DecisionTreeClassifier()
Dt.fit(X_train, y_train)

y_pred_class = Dt.predict(X_test)
print("Accuracy for DT ",Dt.score(X_test, y_test))

#prepare the model
NB = MultinomialNB()
NB.fit(X_train, y_train)

y_pred_class = NB.predict(X_test)
print("Accuracy  for MNB",NB.score(X_test, y_test))

predicted = lgr.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print(cm)

print("False positive rate : %.2f %%" % ((cm[0][1] / float(sum(cm[0])))*100))
print('False negative rate : %.2f %%' % ( (cm[1][0] / float(sum(cm[1]))*100)))


report = classification_report(y_test, predicted)
print(report)

# Plot with Labels
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
plt.title('Confusion Matrix for Logistic Regression ')

#sns.heatmap(matrix,annot=True,fmt="d")
# Set x-axis label
classNames = ['Negative','Positive']
plt.xlabel('Predicted label')
# Set y-axis label
plt.ylabel('True label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

predicted = Dt.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print(cm)

print("False positive rate : %.2f %%" % ((cm[0][1] / float(sum(cm[0])))*100))
print('False negative rate : %.2f %%' % ( (cm[1][0] / float(sum(cm[1]))*100)))


report = classification_report(y_test, predicted)
print(report)

# Plot with Labels
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
plt.title('Confusion Matrix for Decision Tree ')

#sns.heatmap(matrix,annot=True,fmt="d")
# Set x-axis label
classNames = ['Negative','Positive']
plt.xlabel('Predicted label')
# Set y-axis label
plt.ylabel('True label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

predicted = NB.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print(cm)

print("False positive rate : %.2f %%" % ((cm[0][1] / float(sum(cm[0])))*100))
print('False negative rate : %.2f %%' % ( (cm[1][0] / float(sum(cm[1]))*100)))

report = classification_report(y_test, predicted)
print(report)

# Plot with Labels
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
plt.title('Confusion Matrix for Decision Tree ')

#sns.heatmap(matrix,annot=True,fmt="d")
# Set x-axis label
classNames = ['Negative','Positive']
plt.xlabel('Predicted label')
# Set y-axis label
plt.ylabel('True label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

model = { "DecisionTree":tree.DecisionTreeClassifier(max_depth=10),
         "MNB":MultinomialNB(),
         "LogisticRegression":LogisticRegression()   
}

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y ,test_size=0.5)

results = {}
for algo in model:
    clf = model[algo]
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    print ("%s : %.3f " % (algo, score))
    results[algo] = score

winner = max(results, key=results.get)
print("Best algorithm: ", winner)