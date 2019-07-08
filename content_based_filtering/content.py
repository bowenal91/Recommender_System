import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

def get_bad_rows(df):
    N = len(df.index)
    droplist = []
    for i in range(N):
        try:
            int(df[i:i+1]['id'])
        except:
            droplist.append(i)
        return droplist




"""
Most useful information for content-based filtering is going to be the
plot description and then the crew/cast. In particular, director and
writers are definitely most important, but the main actors will also
be important.
"""

movie_data = pd.read_csv('movies_metadata.csv')
credit_data = pd.read_csv('credits.csv')
movie_data.drop(get_bad_rows(movie_data))

#Inner join the two data sets on movie id

data = pd.merge(movie_data,credit_data,on='id',how='inner')

#The plot similarity matrix will be used for comparing plots between two movies
data['overview'] = data['overview'].fillna('')
tfidf_transform = TfidfVectorizer(stop_words='english')
overview_matrix = tfidf_transform.fit_transform(data['overview'])
#plot_similarity_matrix = linear_kernel(overview_matrix,overview_matrix)
#The similarity between two
#print(plot_similarity_matrix)
