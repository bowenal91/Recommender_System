import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import collections
import sys
import pickle

def init_dataframe():
    movie_data = pd.read_csv('movies_metadata.csv')
    credit_data = pd.read_csv('credits.csv')
    keyword_data = pd.read_csv('keywords.csv')
    links_small = pd.read_csv('links_small.csv')
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    movie_data = movie_data[movie_data['id'].isin(links_small)]

    #Inner join the two data sets on movie id
    data = movie_data.merge(credit_data,on='id')
    data = data.merge(keyword_data,on='id')
    data['id'] = data['id'].astype(int)
    data = data.reset_index(drop=True)
    return data

def eval_objects(df):
    df['cast'] = df['cast'].apply(literal_eval)
    df['crew'] = df['crew'].apply(literal_eval)
    df['keywords'] = df['keywords'].apply(literal_eval)
    df['keywords'] = df['keywords'].apply(lambda words: [word['name'] for word in words] if isinstance(words,list) else [])
    df['genres'] = df['genres'].fillna('[]').apply(literal_eval)
    df['genres'] = df['genres'].apply(lambda words: [word['name'] for word in words] if isinstance(words,list) else [])
    return df

def identify_director(crew):
    for person in crew:
        if person['job'] == 'Director':
            name = person['name']
            name = str.lower(name.replace(" ",""))
            return [name]*3
    return []

def identify_writer(crew):
    for person in crew:
        if person['job'] == 'Writer':
            name = person['name']
            name = str.lower(name.replace(" ",""))
            return [name]
    return []

def identify_cast(cast):
    test = [str.lower(person['name'].replace(" ","")) for person in cast] if isinstance(cast,list) else []
    test = test[:3] if len(test) >=3 else test
    return test

def update_jobs(df):
    df['director'] = df['crew'].apply(identify_director)
    df['writer'] = df['crew'].apply(identify_writer)
    df['cast'] = df['cast'].apply(identify_cast)
    return df

def generate_matrix_input(df):
    df['matrix_input'] = df['director']+df['writer']+df['cast']+df['genres']
    df['matrix_input'] = df['matrix_input'].fillna('')
    df['matrix_input'] = df['matrix_input'].apply(lambda x: ' '.join(x))
    return df

def get_overview_matrix(column):
    column = column.fillna('')
    #print(column)
    tfidf_transform = TfidfVectorizer(stop_words='english')
    return tfidf_transform.fit_transform(column)

def get_people_matrix(column):
    column = column.fillna('')
    #print(column)
    counter = CountVectorizer(stop_words='english')
    return counter.fit_transform(column)

"""
Ratings data is far to big to fit into a dataframe
Instead I will read it line by line and analyze as necessary
"""
def get_user_matrix(df):
    ID_ids = pd.Series(df.index,index=data['id']).drop_duplicates()
    N = df["id"].size
    user_matrix = np.zeros((N,N))
    count_matrix = np.zeros((N,N))
    with open("ratings.csv","r") as f:
        firstline = True;
        current_user = 0
        user_data = []
        for line in f:
            if firstline:
                firstline = False
                continue

            dat = line.strip().split(",")
            user = int(dat[0])
            mid = dat[1]
            if user == current_user:
                if int(mid) in ID_ids:
                    if not isinstance(ID_ids[int(mid)],collections.Iterable):
                        user_data.append(dat)
            else:
                #print(user, current_user, user_data)

                n = len(user_data)

                for i in range(n):
                    #print(int(user_data[i][1]))
                    id1 = ID_ids[int(user_data[i][1])]
                    r1 = float(user_data[i][2])
                    if r1 >= 4.0:
                        for j in range(i,n):
                            id2 = ID_ids[int(user_data[j][1])]
                            r2 = float(user_data[j][2])
                            try:
                                user_matrix[id1][id2] += 2.0*r2-5.0
                                count_matrix[id1][id2] += 1.0
                            except:
                                print("ID1: "+str(id1))
                                print("ID2: "+str(id2))
                                print("User Matrix Dimension: "+str(user_matrix.shape))
                                print("MovieID: " + str(user_data[i][1]))
                                print("MovieID2: " + str(user_data[j][1]))

                current_user = user
                if current_user%100==0:
                    print(current_user)
                user_data = []
                if int(mid) in ID_ids:
                    user_data.append(dat)

    for i in range(N):
        for j in range(N):
            if count_matrix[i][j] > 0.0:
                user_matrix[i][j] /= count_matrix[i][j]

    pickle_test = open("count_matrix.pickle","wb")
    pickle.dump(count_matrix,pickle_test)
    pickle_test.close()
    return user_matrix





#Provides recommendations
def get_recs(title,om,pm,ID):
    score_array = linear_kernel(pm[ID],pm)
    #score_array = score_array/max(score_array)
    score_array = 1.0*score_array+20.0*linear_kernel(om[ID],om)
    score_array = list(enumerate(score_array[0]))
    score_array = sorted(score_array, key = lambda x: x[1], reverse=True)
    num_best = 10
    for i in range(num_best):
        print(score_array[i])
    best_ids = [score_array[i][0] for i in range(1,num_best+1)]
    return best_ids



"""
Most useful information for content-based filtering is going to be the
plot description and then the crew/cast. In particular, director and
writers are definitely most important, but the main actors will also
be important.
"""

data = init_dataframe()
data = eval_objects(data)
data = update_jobs(data)
data = generate_matrix_input(data)


data = data.reset_index(drop=True)
print(data.shape)
movie_ids = pd.Series(data.index,index=data['title']).drop_duplicates()
overview_matrix = get_overview_matrix(data['overview'])
cast_matrix = get_people_matrix(data['matrix_input'])

p1 = open("dataframe.pickle","wb")
p2 = open("movie_ids.pickle","wb")
p3 = open("overview_matrix.pickle","wb")
p4 = open("cast_matrix.pickle","wb")
pickle.dump(data,p1)
pickle.dump(movie_ids,p2)
pickle.dump(overview_matrix,p3)
pickle.dump(cast_matrix,p4)
p1.close()
p2.close()
p3.close()
p4.close()
sys.exit()

p5 = open("user_matrix.pickle","wb")
user_matrix = get_user_matrix(data)
pickle.dump(user_matrix,p5)
p5.close()

