# %%


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pickle

# %%
# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart.csv')

# %%


# %%
# print first 5 rows of the dataset
heart_data.head()

# %%
# print last 5 rows of the dataset
heart_data.tail()

# %%
# number of rows and columns in the dataset
heart_data.shape

# %%
# getting some info about the data
heart_data.info()

# %%
# checking for missing values
heart_data.isnull().sum()

# %%
# statistical measures about the data
heart_data.describe()

# %%
# checking the distribution of Target Variable
heart_data['target'].value_counts()

# %% [markdown]
# 1 --> Defective Heart
# 
# 0 --> Healthy Heart

# %%


# %%


# %%
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# %%
print(X)

# %%
print(Y)

# %%


# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# %%
print(X.shape, X_train.shape, X_test.shape)

# %%


# %%


# %%
#model = LogisticRegression()

# %%
# training the LogisticRegression model with Training data
#model.fit(X_train, Y_train)

# %%
# accuracy on training data logistic
#X_train_prediction = model.predict(X_train)
#training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# %%
#print('Accuracy on Training data : ', training_data_accuracy)

# %%
# accuracy on test data logistic
#X_test_prediction = model.predict(X_test)
#test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# %%
#print('Accuracy on Test data : ', test_data_accuracy)

# %%
# Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10,  random_state=42)
# training the Random Forest Model with Training data
random_forest_model.fit(X_train, Y_train)

# %%
# accuracy on training data random forest
random_forest_predictions = random_forest_model.predict(X_train)
training_data_accuracy_Forest = accuracy_score(random_forest_predictions, Y_train)

# %%
print('Accuracy on Training data for random rainforest: ', training_data_accuracy_Forest )

# %%
# accuracy on test data for rainforest
X_test_prediction_forest = random_forest_model.predict(X_test)
test_data_accuracy_forest = accuracy_score(X_test_prediction_forest, Y_test)
pickle.dump(random_forest_model, open('heart3.pkl', 'wb'))
# %%
print('Accuracy on Test data for random rainforest: ', test_data_accuracy_forest )

# %%
# AdaBoost model
adaboost_model = AdaBoostClassifier()
# training the Ada boost model  with Training data (creating relationship)
adaboost_model.fit(X_train, Y_train)

#pickle.dump(adaboost_model, open('heart1.pkl', 'wb'))

# %%
# accuracy on training data for adaboost
adaboost_predictions = adaboost_model.predict(X_train)
training_data_accuracy_adaboost = accuracy_score(adaboost_predictions, Y_train)


print(X_train)
adaboost_predictions

# %%
print('Accuracy on Training data for Adaboost: ', training_data_accuracy_adaboost )

# %%
# accuracy on test data for adaboost
X_test_prediction_adaboost = adaboost_model.predict(X_test)
test_data_accuracy_Adaboost = accuracy_score(X_test_prediction_adaboost, Y_test)

# %%
print('Accuracy on Test data for Adaboost: ', test_data_accuracy_Adaboost)

# %%
# XG 
xg_model=XGBClassifier(n_estimators=100,max_depth=10,min_child_weight=10,random_state=42)
# training the XG Boost model with Training data
xg_model.fit(X_train, Y_train)

# %%
# accuracy on training data for XGboost
xg_predictions = xg_model.predict(X_train)
training_data_accuracy_xg = accuracy_score(xg_predictions, Y_train)

# %%
print('Accuracy on Training data for Xg: ', training_data_accuracy_xg)

# %%
# accuracy on test data for xg
X_test_prediction_Xg = xg_model.predict(X_test)
test_data_accuracy_Xg = accuracy_score(X_test_prediction_Xg, Y_test)
#pickle.dump(xg_model, open('heart2.pkl', 'wb'))

# %%
print('Accuracy on Test data for Xg Boost: ', test_data_accuracy_Xg)

# %%

# %%
#input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
#input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
#input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#prediction = model.predict(input_data_reshaped)
#print(prediction)

#if (prediction[0]== 0):
  #print('The Person does not have a Heart Disease')
#else:
  #print('The Person has Heart Disease')

# %%



