#!/usr/bin/python

""" 
    This is the code to accompany the mini-project. 

    Use a AdaBoost Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

    t0 = time()
	< your clf.fit() line of code >
	print "training time:", round(time()-t0, 3), "s"
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels

#######################################################################

from sklearn.ensemble import AdaBoostClassifier

features_train, features_test, labels_train, labels_test = preprocess()
clf = AdaBoostClassifier(n_estimators = 500)
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
score = clf.score(features_test,labels_test)
print "testing time:", round(time()-t0, 3), "s"

print "accuracy:",score

########################################################################
