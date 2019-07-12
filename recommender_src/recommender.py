import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import collections
import sys
import pickle



#Provides recommendations
def get_recs(title,om,pm,um,ID):
    score_array = linear_kernel(pm[ID],pm)
    #score_array = score_array/max(score_array)
    score_array = um[ID]+1.0*score_array+1.0*linear_kernel(om[ID],om)
    score_array = list(enumerate(score_array[0]))
    score_array = sorted(score_array, key = lambda x: x[1], reverse=True)
    num_best = 10
    for i in range(num_best):
        print(score_array[i])
    best_ids = [score_array[i][0] for i in range(1,num_best+1)]

    #Generate scores to go along with it
    ratings = [um[ID][i] for i in best_ids]
    return [best_ids,ratings]

def get_genre_plots():
    return

def get_movie_plots():
    return

def get_score_plots():
    return

"""
Most useful information for content-based filtering is going to be the
plot description and then the crew/cast. In particular, director and
writers are definitely most important, but the main actors will also
be important.
"""
p = open("dataframe.pickle","rb")
data = pickle.load(p)
p.close()
p = open("movie_ids.pickle","rb")
movie_ids = pickle.load(p)
p.close()
p = open("overview_matrix.pickle","rb")
overview_matrix = pickle.load(p)
p.close()
p = open("cast_matrix.pickle","rb")
cast_matrix = pickle.load(p)
p.close()
p = open("user_matrix.pickle","rb")
user_matrix = pickle.load(p)
p.close()


title = input("Please enter a movie\n")


"""
Run through the recommendation Algorithm
"""



ID = movie_ids[title]
#data['ratings'] = pd.Series(user_matrix[ID],index=data.index)
#print(data['matrix_input'].iloc[ID], data['cast'].iloc[ID], data['writer'].iloc[ID], data['director'].iloc[ID],)

#print(data['title'].iloc[movie_ids['The Avengers']])
if isinstance(movie_ids[title],collections.Iterable):
    for j,i in enumerate(movie_ids[title]):
        print("Option {}".format(j+1))
        data['ratings'] = pd.Series(user_matrix[i],index=data.index)
        test,ratings = get_recs(title,overview_matrix,cast_matrix,user_matrix,i)
        print(data[['title','ratings']].iloc[test])
else:
    data['ratings'] = pd.Series(user_matrix[movie_ids[title]],index = data.index)
    test,ratings = get_recs(title, overview_matrix,cast_matrix,user_matrix,movie_ids[title])
    print(data[['title','ratings']].iloc[test])
    #print(output['ratings'])

