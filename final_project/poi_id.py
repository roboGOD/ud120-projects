#!/usr/bin/python

import sys
import pickle
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### You will need to use more features

#features_list = ['poi', 'salary', 'bonus', 'deferral_payments','total_payments','exercised_stock_options',
#'restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances',
#'from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person']

### Best Features 
features_list = ['poi', 'salary', 'bonus', 'total_payments', 'exercised_stock_options',
'expenses', 'director_fees','restricted_stock_deferred',
'from_this_person_to_poi','deferred_income','long_term_incentive','from_poi_to_this_person']



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)

### Task 3: Create new feature(s)


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


n_comp = 7
classifier = GaussianNB()
classifier_name = 'gnb'
estimators = [('reduce_dim', PCA(n_components=n_comp)), (classifier_name, classifier)]
pipe = Pipeline(estimators)


### Parameters for MLPClassifier
#pipe.set_params(mlp__alpha=1e-5, mlp__hidden_layer_sizes=(5, 2), mlp__max_iter=10000)

### Parameters for RandomForestClassifier
#pipe.set_params(rfc__min_samples_split=50, rfc__n_estimators=100)

### Parameters for DecisionTreeClassifier
#pipe.set_params(dtc__min_samples_split=20)

### Parameters for SVC
#param_grid = dict(svc__C=[1,10,100,1000,10000],svc__gamma=[0.0001,0.001,0.01,0.1,0.5,1])
#clf = GridSearchCV(pipe, param_grid)

### Parameters for AdaBoostClassifier
#pipe.set_params(adb__n_estimators=50)

clf = pipe



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

t0 = time()
clf.fit(features_train, labels_train)
print "Time Taken For Training:", round(time()-t0,3)
t0 = time()
predictions = clf.predict(features_test)
print "Time Taken For Predictions:", round(time()-t0,3),"\n"
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for prediction, truth in zip(predictions, labels_test):
    if prediction == 0 and truth == 0:
        true_negatives += 1
    elif prediction == 0 and truth == 1:
        false_negatives += 1
    elif prediction == 1 and truth == 0:
        false_positives += 1
    elif prediction == 1 and truth == 1:
        true_positives += 1
    else:
        print "Warning: Found a predicted label not == 0 or 1."
        print "All predictions should take value 0 or 1."
        print "Evaluating performance for processed predictions:"
        break

PERF_FORMAT_STRING = "\
\nAccuracy: {:>0.{display_precision}f}\nPrecision: {:>0.{display_precision}f}\n\
Recall: {:>0.{display_precision}f}\nF1: {:>0.{display_precision}f}\nF2: {:>0.{display_precision}f}"
#RESULTS_FORMAT_STRING = "\nTotal predictions: {:4d}\nTrue positives: {:4d}\nFalse positives: {:4d}\n\
#False negatives: {:4d}\nTrue negatives: {:4d}"
try:
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    print clf
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
    #print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    print ""
except:
    print "Got a divide by zero when trying out:", clf
    print "Precision or recall may be undefined due to a lack of true positive predicitons."

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)