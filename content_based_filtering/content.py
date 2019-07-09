import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import collections


#If there are any problematic rows that have misplaced ids then this function
# will take care of it
def get_bad_rows(df):
    N = len(df.index)
    droplist = []
    for i in range(N):
        try:
            int(df[i:i+1]['id'])
        except:
            print(i, df[i:i+1]['id'])
            droplist.append(i)

    return droplist


#Eliminates repeated titles
def drop_repeats(df):
    N = len(df.index)
    droplist = []
    d = {}
    for i in range(N):
        title = df.iloc[i]['title']
        if title in d:
            droplist.append(i)
        else:
            d[title] = 1
    return droplist

#Provides recommendations
def get_recs(title,om,ID):
    #print(om[0])
    score_array = linear_kernel(om[ID],om)
    score_array = list(enumerate(score_array[0]))
    score_array = sorted(score_array, key = lambda x: x[1], reverse=True)
    num_best = 10
    best_ids = [score_array[i][0] for i in range(1,num_best+1)]
    return best_ids



"""
Most useful information for content-based filtering is going to be the
plot description and then the crew/cast. In particular, director and
writers are definitely most important, but the main actors will also
be important.
"""

movie_data = pd.read_csv('movies_metadata.csv')
credit_data = pd.read_csv('credits.csv')
links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
movie_data = movie_data[movie_data['id'].isin(links_small)]
#print(movie_data.shape)

#Inner join the two data sets on movie id
data = movie_data.merge(credit_data,on='id')
#print(data.shape)
#data = data.drop(drop_repeats(data))
data = data.reset_index(drop=True)


movie_ids = pd.Series(data.index,index=data['title']).drop_duplicates()

data['overview'] = data['overview'].fillna('')
tfidf_transform = TfidfVectorizer(stop_words='english')
overview_matrix = tfidf_transform.fit_transform(data['overview'])



title = "The Avengers"

#print(data['title'].iloc[movie_ids['The Avengers']])
if isinstance(movie_ids[title],collections.Iterable):
    for j,i in enumerate(movie_ids[title]):
        print("Option {}".format(j+1))
        test = get_recs(title,overview_matrix,i)
        print(data['title'].iloc[test])
else:
    test = get_recs(title, overview_matrix,movie_ids[title])
    print(data['title'].iloc[test])

#Cannot use similarity matrix because the matrix is too big, need to call dot product as necessary

#print(overview_matrix)
#print(overview_matrix.shape)
#plot_similarity_matrix = linear_kernel(overview_matrix,overview_matrix)

#The similarity between two
