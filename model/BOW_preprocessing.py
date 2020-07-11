import pandas as pd
import nltk
import numpy as np
import os
from nltk.corpus import stopwords, wordnet
from collections import Counter

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

    print(df.head(20))

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

    # Save the Dataframe to a CSV File
    try:
        df.to_csv("BOW_data_preprocessed.csv")
        print("Dataframe saved to CSV file successfully")
    except Exception as e:
        print("Could not save the Dataframe to CSV file successfully")
        print(e)

    # Save the Dataframe to a Pickle file
    try:
        df.to_pickle("BOW_data_preprocessed.pkl")
        print("Dataframe saved to Pickle file successfully")
    except Exception as e:
        print("Could not save the Dataframe to Pickle file successfully")
        print(e)


# Get the sample data path
cwd = os.path.abspath(os.getcwd()) # Get the current working directory
data_path = os.path.join(cwd, "labeled_data.csv") # Join the paths
print(data_path)

# Call the preprocess_data function to preprocess the data
preprocess_data(data_path)