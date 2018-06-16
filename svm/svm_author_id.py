#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

from sklearn import svm

# Comment the following lines for full data( takes too long to train!)
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]



#clf = svm.SVC(kernel = 'linear')
clf = svm.SVC(kernel = 'rbf', C=10000)

t0 = time()
clf.fit(features_train,labels_train)
print "\n\nTraining Time: ", round(time() - t0, 3), "s"

t0 = time()
score = clf.score(features_test,labels_test)
print "Testing Time: ", round(time() - t0, 3), "s"

print "\nAccuracy: ", score*100 , "%"

# Predictions for 10th, 26th and 50th of Test Set

pred = clf.predict([features_test[10],features_test[26],features_test[50]])
print "Predictions for 10th, 26th and 50th element are respectively : ", pred

# How many from Test Set are predicted to be in Chris(1) class 

pred = clf.predict(features_test)
count = 0
for i in pred:
	if(i==1):
		count+=1
print "\n>> ", count, " out of test set are predicted to be in Chris(1) class"


#########################################################


