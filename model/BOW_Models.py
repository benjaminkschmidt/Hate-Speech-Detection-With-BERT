import pandas as pd
import numpy as np
import nltk
import os
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore') 

# Import the necessary models
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load the Dataframe
cwd = os.path.abspath(os.getcwd()) # Get the current working directory
pkl_path = os.path.join(cwd, "BOW_data_preprocessed.pkl") # Join the paths
df = pd.DataFrame(pd.read_pickle(pkl_path))

# Extract the necessary data/columns for training purposes
X = []
for i in range(0,len(df)):
    X.append(df.iloc[i]["BOW_representation"].tolist())

X = np.asarray(X)
y = df["class"]

# Form the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Shape of training set : " + str(X_train.shape))
print("Shape of testing set : " + str(X_test.shape))

# -------------------------------------------------- Gaussian Naive Bayes Model --------------------------------------------------

# Initialize the model
clf_NB = GaussianNB()

print("\n------------------------- Running the GaussianNB Model on BOW data -------------------------")

# Fit the model on the training data
clf_NB.fit(X_train, y_train)

# Make the predictions
y_pred = clf_NB.predict(X_test)

# Get the overall model performance metrics on the testing set
print("---------- Model Performance Metrics with Gaussian Naive Bayes Model ----------")
print("Accuracy : " + str(accuracy_score(y_test, y_pred, )*100))
print("Precision : " + str(precision_score(y_test, y_pred, average='macro')*100))
print("Recall : " + str(recall_score(y_test, y_pred, average='macro')*100))

# Get the confusion matrix
print(confusion_matrix(y_test, y_pred))


# -------------------------------------------------- Linear SVC Model --------------------------------------------------

# Initialize the model
clf_SVC = SVC(gamma='auto')

print("\n------------------------- Running the SVC Model on BOW data -------------------------")

# Fit the model on the training data
clf_SVC.fit(X_train, y_train)

# Make the predictions
y_pred = clf_SVC.predict(X_test)

# Get the overall model performance metrics on the testing set
print("---------- Model Performance Metrics with SVC Model ----------")
print("Accuracy : " + str(accuracy_score(y_test, y_pred, )*100))
print("Precision : " + str(precision_score(y_test, y_pred, average='macro')*100))
print("Recall : " + str(recall_score(y_test, y_pred, average='macro')*100))

# Get the confusion matrix
print(confusion_matrix(y_test, y_pred))


# -------------------------------------------------- Random Forest Model --------------------------------------------------

# Initialize the model
clf_rf = RandomForestClassifier()

print("\n------------------------- Running the Random Forest Model on BOW data -------------------------")

# Fit the model on the training data
clf_rf.fit(X_train, y_train)

# Make the predictions
y_pred = clf_rf.predict(X_test)

# Get the overall model performance metrics on the testing set
print("---------- Model Performance Metrics with Random Forest Model ----------")
print("Accuracy : " + str(accuracy_score(y_test, y_pred, )*100))
print("Precision : " + str(precision_score(y_test, y_pred, average='macro')*100))
print("Recall : " + str(recall_score(y_test, y_pred, average='macro')*100))

# Get the confusion matrix
print(confusion_matrix(y_test, y_pred))