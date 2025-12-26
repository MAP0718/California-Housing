#Imports

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

#constants
MODEL_FILE='model.pkl'
PIPELINE_FILE='pipeline.pkl'

#Pipeline Function

def build_pipeline(num_attr,cat_attr):
	
	num_pipeline=Pipeline([
		('impute',SimpleImputer(strategy='median')),
		('scaler',StandardScaler())
		])
	
	cat_pipeline=Pipeline([
		('encoder',OneHotEncoder(handle_unknown='ignore'))
		])

	full_pipeline=ColumnTransformer([
		('num',num_pipeline,num_attr),
		('cat',cat_pipeline,cat_attr)
		])
	
	return full_pipeline

#If The Model Is Not Trained We Train And Save The Model And Pipeline

if not os.path.exists(MODEL_FILE):

	#load the data
	data=pd.read_csv("housing.csv")

	# Separate the data Test and Train
	
	data['income_cat']=pd.cut(data['median_income'],
				bins=[0,1.5,3.0,4.5,6.0,np.inf],
				labels=[1,2,3,4,5] )

	split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

	for train_index,test_index in split.split(data,data['income_cat']):
	
		test_set=data.loc[test_index].drop('income_cat',axis=1)
		housing=data.loc[train_index].drop('income_cat',axis=1)

	test_set.to_csv('input.csv') #Saving Test Set to input.csv

	housing_features=housing.drop('median_house_value',axis=1)
	housing_labels=housing['median_house_value']

	num_attr=housing_features.drop('ocean_proximity',axis=1).columns.tolist()
	cat_attr=['ocean_proximity']

	pipeline=build_pipeline(num_attr,cat_attr)

	housing_prepared=pipeline.fit_transform(housing_features)
	
	model=RandomForestRegressor()
	model.fit(housing_prepared,housing_labels)

	joblib.dump(model,MODEL_FILE)
	joblib.dump(pipeline,PIPELINE_FILE)

	# model and pipeline are saved and model is trained 

#Inference Phase
	
else:
	
	model=joblib.load(MODEL_FILE)
	pipeline=joblib.load(PIPELINE_FILE)

	test_data=pd.read_csv('input.csv')

	preprocess_test_data=pipeline.transform(test_data)

	predicted_data=model.predict(preprocess_test_data)
	
	test_data['median_house_value']=predicted_data
	test_data.to_csv('output.csv',index=False)
	
	# Save the data to output.csv
