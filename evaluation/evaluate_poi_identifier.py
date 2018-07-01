#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

### Train-Test Split
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)

### Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print "Accuracy:", clf.score(features_test, labels_test)


### Code For Evaluation Metrics
nPOIsTest = len([i for i in labels_test if i == 1])
print "Number of POIs in the Test Set:", nPOIsTest
nTotalTest = len(labels_test)
print "Number of People in the Test Set:", nTotalTest
labels_pred = clf.predict(features_test)
#for i,j in zip(labels_test, labels_pred):
#	print int(i), "\t", int(j)

### Precision and Recall
from sklearn.metrics import precision_score, recall_score

precision = precision_score(labels_test, labels_pred)
print "Precision: ", precision

recall =  recall_score(labels_test, labels_pred)
print "Recall: ", recall


### Practice Stuff (Since we will be optimizing the POI Identifier in Final Project)
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print "\n\n#################################################","\n\t\t Practice Stuff","\n#################################################"

print "Predictions:", predictions
print "True Labels:", true_labels
true_pos = len([i for i,j in zip(predictions,true_labels) if i==j and i==1])
print "True Positives:", true_pos
true_neg = len([i for i,j in zip(predictions,true_labels) if i==j and i==0])
print "True Negitives:", true_neg
false_pos = len([i for i,j in zip(predictions,true_labels) if i!=j and i==1])
print "False Positives:", false_pos
false_neg = len([i for i,j in zip(predictions,true_labels) if i!=j and i==0])
print "False Negitives:", false_neg
prec_score = float(true_pos)/float(true_pos+false_pos)
print "Precision:", prec_score
rec_score = float(true_pos)/float(true_pos+false_neg)
print "Recall:", rec_score




