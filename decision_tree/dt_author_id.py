#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# Find number of features in data
print "\n\nNumber of Features: ", len(features_train[0])
print "Number of Data Points: ", len(features_train)

from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split = 40)

t0 = time()
clf.fit(features_train,labels_train)
print "\n\nTraining Time: ", round(time()-t0, 3), "s"

t0 = time()
score = clf.score(features_test,labels_test)
print "Testing Time: ", round(time() - t0, 3), "s"

print "\n Accuracy: ", round(score*100,2), "%"

#########################################################


