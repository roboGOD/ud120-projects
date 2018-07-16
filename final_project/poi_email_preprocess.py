
import sys
from poi_email_addresses import poiEmails
sys.path.append("../tools")
from parse_out_email_text import parseOutText
import pickle


### dumpEmailData: Operations Performed:
###		Get the email_address of POIs and Non POIs
###		Go to emails_by_address directory
### 	Open files "to" and "from" that email_address(if they exist)
###		Go to the enron mail directory and the specified paths in the file.
###		Fetch the content of the emails
### 	Store it in a dictionary(word_data) with the person name as key
###		After all the emails are done, dump the word_data to pickle file. 
def dump_email_data(data_dict):

	directory = "emails_by_address/"
	counter = 0
	word_data = {}


	ls = poiEmails()
	for key in ls:
		email = ls[key]
		path1 = directory + "from_" + email + ".txt"
		path2 = directory + "to_" + email + ".txt"
		words = ""
		
		try:
			f1 = open(path1, "r")
			ls1 = f1.readlines()
			f1.close()
			for path in ls1:
				path = "../" + path[:-1]
				f2 = open(path, "r")
				words = words + " " + (parseOutText(f2))
				f2.close()

		except Exception:
			pass

		try:
			f1 = open(path2, "r")
			ls1 = f1.readlines()
			f1.close()
			for path in ls1:
				path = "../" + path[:-1]
				f2 = open(path, "r")
				words = words + " " + (parseOutText(f2))
				f2.close()

		except Exception:
			pass

		if words != "":
			if key in word_data:
				word_data[key] = word_data[key] + " " + words
			else:
				word_data[key] = words
			del words

	for key in data_dict:
		email = data_dict[key]['email_address']
		path1 = directory + "from_" + email + ".txt"
		path2 = directory + "to_" + email + ".txt"
		words = ""

		try:
			f1 = open(path1, "r")
			ls = f1.readlines()
			f1.close()
			for path in ls:
				path = "../" + path[:-1]
				f2 = open(path, "r")
				words = words + " " + (parseOutText(f2))
				f2.close()
		except Exception:
			pass

		try:
			f1 = open(path2, "r")
			ls = f1.readlines()
			f1.close()
			for path in l1:
				path = "../" + path[:-1]
				f2 = open(path, "r")
				words = words + " " + (parseOutText(f2))
				f2.close()
		except Exception:
			pass

		if words != "":
			if key in word_data:
				word_data[key] = word_data[key] + " " + words
			else:
				word_data[key] = words
			del words
		
		counter += 1
		print counter

	pickle_out = open("email_data.pickle", "wb")
	pickle.dump(word_data, pickle_out)
	pickle_out.close()



