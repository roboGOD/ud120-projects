#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "\nNumber of Data Points(Employees): ", len(enron_data)
print "Number of features for each person: ", len(enron_data["SKILLING JEFFREY K"])

# Finding Persons of Interest (POIs) and list of total Persons
listPOI = []
listPersons = []

for person_name in enron_data:
	listPersons.append(person_name)
	if enron_data[person_name]["poi"] == 1 :
		listPOI.append(person_name)
print "Number of POIs in the dataset: ", len(listPOI)

# Finding List of all features
listFeatures = []
for feature in enron_data["PRENTICE JAMES"]:
	listFeatures.append(feature)

#print listFeatures

print "Total value of stock belonging to James Prentice: ", enron_data["PRENTICE JAMES"]["total_stock_value"]
print "Number of Emails from Wesley Colwell to POIs: ", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "Value of stock options exercised by Jeffrey K Skilling: ", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print "Ken Lay-> ", "Total Payments: ", enron_data["LAY KENNETH L"]["total_payments"]
print "\nThe feature without a value for a particular person is represented as 'NaN' or 'Not a Number'\n"

countSal = 0
countEmail = 0
for person_name in enron_data:
	if enron_data[person_name]["salary"] != "NaN":
		countSal += 1
	if enron_data[person_name]["email_address"] != "NaN":
		countEmail += 1

print "Number of persons with quantified salary: ", countSal
print "Number of persons with know Email: ", countEmail

count_Total_Pay_NaN = 0
for person_name in enron_data:
	if enron_data[person_name]["total_payments"] == "NaN":
		count_Total_Pay_NaN += 1
float_count = float(count_Total_Pay_NaN)
float_len = float(len(enron_data))
percentage = round(float_count/float_len*100,2)
print "Percentage of number of persons with 'NaN' as total payments: ", percentage, "%"

count_POI_Pay_NaN = 0
for person_name in listPOI:
	if enron_data[person_name]["total_payments"] == "NaN":
		count_POI_Pay_NaN += 1
float_count = float(count_POI_Pay_NaN)
float_len = float(len(enron_data))
percentage = round(float_count/float_len*100,2)
print "Percentage of number of POIs with 'NaN' as total payments: ", percentage, "%"

# If we add 10 POIs with "NaN" as total_payments
print "New Number of Persons after adding 10 more POIs: ", len(enron_data)+10
print "New Number of Persons with NaN as total_payments: ", count_Total_Pay_NaN+10
print "New Number of POIs: ", len(listPOI)+10
print "New Number of POIs with NaN as total_payments: ", count_POI_Pay_NaN+10

'''
Once the new data points are added, do you think a supervised classification 
algorithm might interpret “NaN” for total_payments as a clue that someone is a POI?

Yes Ofcourse!

Adding in the new POI’s in this example, none of whom we have financial information for, 
has introduced a subtle problem, that our lack of financial information about them can be 
picked up by an algorithm as a clue that they’re POIs. Another way to think about this is 
that there’s now a difference in how we generated the data for our two classes--non-POIs 
all come from the financial spreadsheet, while many POIs get added in by hand afterwards. 
That difference can trick us into thinking we have better performance than we do--suppose 
you use your POI detector to decide whether a new, unseen person is a POI, and that person 
isn’t on the spreadsheet. Then all their financial data would contain “NaN” but the person 
is very likely not a POI (there are many more non-POIs than POIs in the world, and even at 
Enron)--you’d be likely to accidentally identify them as a POI, though!
This goes to say that, when generating or augmenting a dataset, you should be exceptionally 
careful if your data are coming from different sources for different classes. It can easily 
lead to the type of bias or mistake that we showed here. There are ways to deal with this, 
for example, you wouldn’t have to worry about this problem if you used only email data--
in that case, discrepancies in the financial data wouldn’t matter because financial features 
aren’t being used. There are also more sophisticated ways of estimating how much of an effect 
these biases can have on your final answer; those are beyond the scope of this course.
For now, the takeaway message is to be very careful about introducing features that come from 
different sources depending on the class! It’s a classic way to accidentally introduce biases 
and mistakes.
'''

print ""