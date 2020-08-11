import pandas as pd
import nltk
import numpy as np
import os
from nltk.corpus import stopwords, wordnet
from collections import Counter
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Import the necessary models
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore') 

# Form the set of stopwords
stop_words = set(stopwords.words('english')) 

# Function for preprocessing the data using BOW 
def preprocess_data(filename):

    # Load the data from the csv file
    df = pd.DataFrame(pd.read_csv(filename))

    print("The size of the dataframe is : {}".format(len(df)))

    # Extract the necessary columns
    df = df[["tweet", "class"]]

    # Convert the columns to lowercase
    df['tweet'] = df['tweet'].str.lower()

    # print(df.head(20))

    word_corpus = []
    tweet_word_lists = []

    # ---------------------------------------- BOW Model ----------------------------------------

    # Get the tokenized words for each tweet
    print("Length of the Dataframe : " + str(len(df)))
    for i in range(0,len(df)):
        # print(i)
        l = []
        tweet_words = nltk.word_tokenize(df.iloc[i]["tweet"])   # Tokenize the tweet
        for word in tweet_words:
            # The word is in the english dictionary and not a stop word
            if wordnet.synsets(word) and word not in stop_words: 
                word_corpus.append(word)
                l.append(word)
        tweet_word_lists.append(l)

    # Get the 20 most common words from all_words
    most_frequent_words = []
    counter = Counter(word_corpus).most_common(20)
    for i in range(0,len(counter)):
        most_frequent_words.append(counter[i][0])

    # Remove the duplicate words from the most frequent words
    most_frequent_words = list(set(most_frequent_words))
    most_frequent_words_len = len(most_frequent_words) 
    print("Length of most_frequent_words : " + str(most_frequent_words_len))
    print("Most Frequent Words : " + str(most_frequent_words))

    word_count_list = []

    # Using the most frequent words for the Bag of Words model
    for i in range(0,len(tweet_word_lists)):
        l = np.zeros(most_frequent_words_len)
        for j in range(0,len(most_frequent_words)):
            count = tweet_word_lists[i].count(most_frequent_words[j])
            l[j] = count
        # print(len(l))
        word_count_list.append(l)

    # Form the new representation with the BOW representation
    df["BOW_representation"] = word_count_list

    return most_frequent_words_len, most_frequent_words, df


print("\n-------------------- Preprocessing on the Labeled Dataset --------------------")

# Get the sample data path
cwd = os.path.abspath(os.getcwd()) # Get the current working directory
data_path = os.path.join(cwd, "labeled_data.csv") # Join the paths
print(data_path)

# Call the preprocess_data function to preprocess the data
most_frequent_words_len, most_frequent_words, df_labeled = preprocess_data(data_path)

# Extract the necessary data/columns for training purposes
X = []
for i in range(0,len(df_labeled)):
    X.append(df_labeled.iloc[i]["BOW_representation"].tolist())

X = np.asarray(X)
y = df_labeled["class"]

# Form the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Shape of training set : " + str(X_train.shape))
print("Shape of testing set : " + str(X_test.shape))


# -------------------------------------------------- Gaussian Naive Bayes Model without Grid Search --------------------------------------------------

# Initialize the model
clf_NB = GaussianNB()

print("\n\n------------------------- Running the GaussianNB Model without Grid Search on BOW data -------------------------")

# Fit the model on the training data
clf_NB.fit(X_train, y_train)

# Make the predictions
y_pred = clf_NB.predict(X_test)

# Get the overall model performance metrics on the testing set
print("---------- Model Performance Metrics with Gaussian Naive Bayes Model ----------")
print("Accuracy : " + str(accuracy_score(y_test, y_pred, )*100))
print("Precision : " + str(precision_score(y_test, y_pred, average='macro')*100))
print("Recall : " + str(recall_score(y_test, y_pred, average='macro')*100))
print("F1 Score : " + str(f1_score(y_test, y_pred, average='macro')))

# Get the confusion matrix
print(confusion_matrix(y_test, y_pred))


# -------------------------------------------------- Linear SVC Model without Grid Search--------------------------------------------------

# Initialize the model
clf_SVC = SVC(gamma='auto')

print("\n\n------------------------- Running the SVC Model without Grid Search on BOW data -------------------------")

# Fit the model on the training data
clf_SVC.fit(X_train, y_train)

# Make the predictions
y_pred = clf_SVC.predict(X_test)

# Get the overall model performance metrics on the testing set
print("---------- Model Performance Metrics with SVC Model ----------")
print("Accuracy : " + str(accuracy_score(y_test, y_pred, )*100))
print("Precision : " + str(precision_score(y_test, y_pred, average='macro')*100))
print("Recall : " + str(recall_score(y_test, y_pred, average='macro')*100))
print("F1 Score : " + str(f1_score(y_test, y_pred, average='macro')))

# Get the confusion matrix
print(confusion_matrix(y_test, y_pred))


# -------------------------------------------------- Random Forest Model without Grid Search --------------------------------------------------

# Initialize the model
clf_rf = RandomForestClassifier()

print("\n\n------------------------- Running the Random Forest Model without Grid Search on BOW data -------------------------")

# Fit the model on the training data
clf_rf.fit(X_train, y_train)

# Make the predictions
y_pred = clf_rf.predict(X_test)

# Get the overall model performance metrics on the testing set
print("---------- Model Performance Metrics with Random Forest Model ----------")
print("Accuracy : " + str(accuracy_score(y_test, y_pred, )*100))
print("Precision : " + str(precision_score(y_test, y_pred, average='macro')*100))
print("Recall : " + str(recall_score(y_test, y_pred, average='macro')*100))
print("F1 Score : " + str(f1_score(y_test, y_pred, average='macro')))

# Get the confusion matrix
print(confusion_matrix(y_test, y_pred))


# # -------------------------------------------------- Linear SVC Model with Grid Search --------------------------------------------------

# # Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10]},
#                     {'kernel': ['linear'], 'C': [1, 10]}]

# scores = ['precision', 'recall']

# print("\n\n------------------------- Running SVC Model with Grid Search on BOW data -------------------------")

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(
#         SVC(), tuned_parameters, scoring='%s_macro' % score
#     )
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()


# # -------------------------------------------------- Random Forest Model with Grid Search --------------------------------------------------

# # Set the parameters by cross-validation
# param_grid = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt'],
#     'max_depth' : [4,5],
#     'criterion' :['gini', 'entropy']
# }

# scores = ['precision', 'recall']

# print("\n\n------------------------- Running Random Forest Model with Grid Search on BOW data -------------------------")

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(
#         RandomForestClassifier(), param_grid=param_grid, scoring='%s_macro' % score
#     )
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()

# ------------------------- Preprocessing the Unlabeled Data using BOW -------------------------

# Load the data from the csv file
cwd = os.path.abspath(os.getcwd()) # Get the current working directory
file_path = os.path.join(cwd, "cleaned_2 - 2k_anti_racist_tweets.csv") # Join the paths
df_unlabeled = pd.read_csv(file_path, encoding = "ISO-8859-1", engine='python')

print("The size of the dataframe is : {}".format(len(df_unlabeled)))

# Convert the columns to lowercase
df_unlabeled['tweet'] = df_unlabeled['tweet'].str.lower()

# print(df_unlabeled.head(10))

word_corpus = []
tweet_word_lists = []

# Get the tokenized words for each tweet
print("Length of the Dataframe : " + str(len(df_unlabeled)))
for i in range(0,len(df_unlabeled)):
    # print(i)
    l = []
    tweet_words = nltk.word_tokenize(df_unlabeled.iloc[i]["tweet"])   # Tokenize the tweet
    for word in tweet_words:
        # The word is in the english dictionary and not a stop word
        if wordnet.synsets(word) and word not in stop_words: 
            word_corpus.append(word)
            l.append(word)
    tweet_word_lists.append(l)

word_count_list = []

# Using the most frequent words for the Bag of Words model
for i in range(0,len(tweet_word_lists)):
    l = np.zeros(most_frequent_words_len)
    for j in range(0,len(most_frequent_words)):
        count = tweet_word_lists[i].count(most_frequent_words[j])
        l[j] = count
    # print(len(l))
    word_count_list.append(l)

# Form the new representation with the BOW representation
df_unlabeled["BOW_representation"] = word_count_list


# Run the pretrained models to make predictions on the unlabeled BOW data

# -------------------------------------------------- Gaussian Naive Bayes Model --------------------------------------------------

# Extract the necessary column for prediction purpose
X_unlabeled = []
for i in range(0,len(df_unlabeled)):
    X_unlabeled.append(df_unlabeled.iloc[i]["BOW_representation"].tolist())

# Make the Predictions using the pretrained Gaussian NB Model
y_pred_unlabeled = clf_NB.predict(X_unlabeled)

# Output the Predicitions
print("\n---------- Predictions on Unlabeled Dataset with Gaussian Naive Bayes Model ----------")
# for i in range(0,len(df_unlabeled)):
#     print("Tweet : {} , label : {}".format(df_unlabeled.iloc[i]["tweet"], y_pred_unlabeled[i]))

# Frequency Distribution Plot

count_0 = list(y_pred_unlabeled).count(0)
count_1 = list(y_pred_unlabeled).count(1)
count_2 = list(y_pred_unlabeled).count(2)

print("Total tweets classified as Hate Speech (Label 0) = {}".format(count_0))
print("Total tweets classified as Offensive Language (Label 1) = {}".format(count_1))
print("Total tweets classified as Neither (Label 2) = {}".format(count_2))

print("\nPercentage correct classification = {}".format((count_2/len(y_pred_unlabeled)) * 100))


x_labels = np.array(['Hate Speech', 'Offensive Language', 'Neither'], dtype='|S13')
y_values = np.array([count_0, count_1, count_2])

# creating the bar plot 
plt.bar(x_labels, y_values) 
  
plt.xlabel("Tweet Type") 
plt.ylabel("Frequency") 
plt.title("Frequency Distribution of various types of tweets using Gaussian Naive Bayes") 
plt.show()


# -------------------------------------------------- Linear SVC Model --------------------------------------------------

# Extract the necessary column for prediction purpose
X_unlabeled = []
for i in range(0,len(df_unlabeled)):
    X_unlabeled.append(df_unlabeled.iloc[i]["BOW_representation"].tolist())

# Make the Predictions using the pretrained Linear SVC Model
y_pred_unlabeled = clf_SVC.predict(X_unlabeled)

# Output the Predicitions
print("\n---------- Predictions on Unlabeled Dataset with Linear SVC Model ----------")
# for i in range(0,len(df_unlabeled)):
#     print("Tweet : {} , label : {}".format(df_unlabeled.iloc[i]["tweet"], y_pred_unlabeled[i]))


# Frequency Distribution Plot

count_0 = list(y_pred_unlabeled).count(0)
count_1 = list(y_pred_unlabeled).count(1)
count_2 = list(y_pred_unlabeled).count(2)

print("Total tweets classified as Hate Speech (Label 0) = {}".format(count_0))
print("Total tweets classified as Offensive Language (Label 1) = {}".format(count_1))
print("Total tweets classified as Neither (Label 2) = {}".format(count_2))

print("\nPercentage correct classification = {}".format((count_2/len(y_pred_unlabeled)) * 100))

x_labels = np.array(['Hate Speech', 'Offensive Language', 'Neither'], dtype='|S13')
y_values = np.array([count_0, count_1, count_2])

# Creating the bar plot 
plt.bar(x_labels, y_values) 
  
plt.xlabel("Tweet Type") 
plt.ylabel("Frequency") 
plt.title("Frequency Distribution of various types of tweets using Linear SVC") 
plt.show()


# -------------------------------------------------- Random Forest Model --------------------------------------------------

# Extract the necessary column for prediction purpose
X_unlabeled = []
for i in range(0,len(df_unlabeled)):
    X_unlabeled.append(df_unlabeled.iloc[i]["BOW_representation"].tolist())

# Make the Predictions using the pretrained Random Forest Model
y_pred_unlabeled = clf_rf.predict(X_unlabeled)

# Output the Predicitions
print("\n---------- Predictions on Unlabeled Dataset with Linear SVC Model ----------")
# for i in range(0,len(df_unlabeled)):
#     print("Tweet : {} , label : {}".format(df_unlabeled.iloc[i]["tweet"], y_pred_unlabeled[i]))

# Frequency Distribution Plot

count_0 = list(y_pred_unlabeled).count(0)
count_1 = list(y_pred_unlabeled).count(1)
count_2 = list(y_pred_unlabeled).count(2)

print("Total tweets classified as Hate Speech (Label 0) = {}".format(count_0))
print("Total tweets classified as Offensive Language (Label 1) = {}".format(count_1))
print("Total tweets classified as Neither (Label 2) = {}".format(count_2))

print("\nPercentage correct classification = {}".format((count_2/len(y_pred_unlabeled)) * 100))

x_labels = np.array(['Hate Speech', 'Offensive Language', 'Neither'], dtype='|S13')
y_values = np.array([count_0, count_1, count_2])

# creating the bar plot 
plt.bar(x_labels, y_values) 
  
plt.xlabel("Tweet Type") 
plt.ylabel("Frequency") 
plt.title("Frequency Distribution of various types of tweets using Random Forest") 
plt.show() 