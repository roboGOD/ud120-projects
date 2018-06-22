#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "r", "g", "c", "m"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place green x over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="m", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)

for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

from sklearn.cluster import KMeans

km = KMeans(n_clusters = 2)
km.fit(finance_features)
pred = km.labels_


### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file

try:
    Draw(pred, finance_features, poi, mark_poi=False, name="tempclusters1.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"


### Sneak Preview
### For exercised_stock_options
minValStock = 0
maxValStock = 0
for person in data_dict:
	if data_dict[person]["exercised_stock_options"] != "NaN":
		if minValStock == 0 and maxValStock == 0:
			minValStock = data_dict[person]["exercised_stock_options"]
			maxValStock = data_dict[person]["exercised_stock_options"]
		elif minValStock > data_dict[person]["exercised_stock_options"]:
			minValStock = data_dict[person]["exercised_stock_options"]
		elif maxValStock < data_dict[person]["exercised_stock_options"] :
			maxValStock = data_dict[person]["exercised_stock_options"]

print "exercised_stock_options -> minVal =", minValStock, " & maxVal =", maxValStock

### For salary
minValSal = 0
maxValSal = 0
for person in data_dict:
	if data_dict[person]["salary"] != "NaN":
		if minValSal == 0 and maxValSal == 0:
			minValSal = data_dict[person]["salary"]
			maxValSal = data_dict[person]["salary"]
		elif minValSal > data_dict[person]["salary"]:
			minValSal = data_dict[person]["salary"]
		elif maxValSal < data_dict[person]["salary"] :
			maxValSal = data_dict[person]["salary"]

print "salary -> minVal =", minValSal, " & maxVal =", maxValSal

### Feature Scaling

from sklearn.preprocessing import MinMaxScaler

salArray = []
stockArray = []
for f1,f2,_ in finance_features:
	### Ignoring Zero Items
	#if f1 != 0.0 and f2 != 0.0:
	#	salArray.append(float(f1))
	#	stockArray.append(float(f2))
	salArray.append(float(f1))
	stockArray.append(float(f2))

salArray = numpy.array([salArray])
stockArray = numpy.array([stockArray])
salArray = salArray.reshape((salArray.size,1))
stockArray = stockArray.reshape((salArray.size,1))

salScale = MinMaxScaler()
stockScale = MinMaxScaler()

salArray = salScale.fit_transform(salArray)
stockArray = stockScale.fit_transform(stockArray)

print "New values for Sal = $200,000 and Stock = $1,000,000: ", salScale.transform([[200000]]), " and ", stockScale.transform([[1000000]])

rescaled_finance_features = numpy.array([salArray.ravel(), stockArray.ravel()]).reshape((len(salArray), 2), order = 'F')

for f1,f2 in rescaled_finance_features:
	plt.scatter(f1,f2)
plt.show()

km = KMeans(n_clusters = 2)
km.fit(rescaled_finance_features)
pred = km.labels_

Draw(pred, rescaled_finance_features , poi, mark_poi=False, name="tempclusters2.pdf", f1_name=feature_1, f2_name=feature_2)
