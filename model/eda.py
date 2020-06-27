import pandas as pd
#cleaning and eda of datafile
dataframe = pd.read_csv('labeled_data.csv', header=None)
#memory usage
print('Training Set Shape = {}'.format(dataframe.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(dataframe.memory_usage().sum() / 1024**2))
print(dataframe.head(1))
#number of each label
print('There are this many of each class')
print(dataframe[5].value_counts())


#average size and distribution of comments
results = set()
dataframe[6].str.lower().str.split().apply(results.update)
print("There are ",len(results), "unique words")
print("Due to the offensive nature of this dataset, we will not be sharing the most common words here")

#average length of each row
dataframe['totalwords'] = [len(x.split()) for x in dataframe[6].tolist()]
print("The average size of each tweet is",dataframe['totalwords'].mean(axis = 0))

