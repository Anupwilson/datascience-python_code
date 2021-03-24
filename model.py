# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:56:40 2021

@author: Anup w
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('final_data.csv')


X = dataset.loc[:,['caller_id','open_by','loc','category']]



y = dataset.loc[:, ['i_impact']]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

#Fitting model with trainig data
classifier.fit(X, y)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2403,397,165,215]]))