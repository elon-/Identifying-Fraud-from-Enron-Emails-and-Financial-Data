#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot 
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from  numpy  import *
from tester import test_classifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV 
from sklearn.pipeline import Pipeline
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
features_list=["poi","other", "expenses","fraction_to_poi",'shared_receipt_with_poi',"exercised_stock_options"]
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",None)


### Task 3: Create new feature(s)
my_dataset = data_dict
def fraction (total_message,message_with_poi):
    if total_message!="NaN"and message_with_poi!="NaN":
            fraction=(float(message_with_poi)/float(total_message))
    else:
         fraction=0.        
    return fraction 
new_dataset={}
for name in my_dataset:
    data_point=my_dataset[name]
    #print name
    data_point["fraction_from_poi"]=fraction(data_point["to_messages"],data_point['from_poi_to_this_person'])
    data_point["fraction_to_poi"]=fraction(data_point["from_messages"],data_point['from_this_person_to_poi'])
    #print data_point
    new_dataset[name]=data_point

### Store to my_dataset for easy export below.
my_dataset = new_dataset


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels,features=targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=0.4, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
dec_tree = DecisionTreeClassifier()
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
param_grid1 ={"dec_tree__criterion":("gini","entropy"),
              "dec_tree__min_samples_split":[1,2,3,4],
              "dec_tree__max_features":(None,"auto","log2")}
### using our testing script. Check the tester.py script in the final project
pipeline=Pipeline([("std",StandardScaler()),
           ("dec_tree",DecisionTreeClassifier())])
           
clf=GridSearchCV(estimator=pipeline,param_grid=param_grid1,cv=5,scoring="precision")
clf.fit(features_train,labels_train)
clf = clf.best_estimator_
score=clf.score(features_test,labels_test)
pre=clf.predict(features_test)
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
test_classifier(clf, my_dataset, features_list)
# Example starting point. Try investigating other evaluation techniques!


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)