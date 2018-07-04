#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from operator import itemgetter


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL",0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
### Visualize Bonus and Salary
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus , color = "b", marker = "o")

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()



### Printing Top 4 Persons with respect to Bonus and Salary
num = 4
listMax = []
for person in data_dict:
	#if data_dict[person]["bonus"] > 5000000 and data_dict[person]["salary"] > 1000000 \ 
	#and data_dict[person]["bonus"] != "NaN" and data_dict[person]["salary"] != "NaN":
	if data_dict[person]["bonus"] != "NaN" and data_dict[person]["salary"] != "NaN":
		listMax.append((person, data_dict[person]["bonus"], data_dict[person]["salary"]))

### Sorting By Bonus
listMax = sorted(listMax, key= itemgetter(1), reverse = True)
print "List of", num, "Persons (Person Name, Bonus, Salary)"
for t in listMax[:num]:
	print t
