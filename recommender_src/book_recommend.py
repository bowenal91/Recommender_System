import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import collections
import sys
import pickle
import matplotlib.pyplot as plt

path = "../book_data/"

#Perform simple data manipulation on dataframe to get it in correct position
def init_dataframe():
    data = pd.read_csv(path+'books.csv')
    book_tags = pd.read_csv(path+'tag_array.csv')
    data = data.merge(book_tags,on='goodreads_book_id')
    data['id'] = data['goodreads_book_id'].astype(int)
    data['tag_id'] = data['tag_id'].apply(literal_eval)
    data['count'] = data['count'].apply(literal_eval)
    data = data[data['goodreads_book_id'].isin(book_tags['goodreads_book_id'])]
    data['author'] = data['authors'].apply(identify_author)
    data['average_rating2'] = data['average_rating'].astype(float).apply(lambda x: 2.0*x-5.0)
    data['tag_id2'] = data['tag_id'].apply(list_to_string)
    data = data.reset_index(drop=True)
    return data


#Identify the first author. This helps for situations where illustrators are also listed
def identify_author(authors):
    #Just use the first author
    test = authors.strip().split(',')
    test = [i.strip() for i in test]
    name = test[0]
    name = str.lower(name.replace(" ",""))
    return name

#Converts list of numbers to format that can be used in tf-idf transform
def list_to_string(l):
    out = ""
    for i in l:
        out += str(i)+" "
    out = out.strip()
    return out

#gets the tag feature matrix
def get_matrix(column):
    column = column.fillna('')
    tfidf_transform = TfidfVectorizer(stop_words='english')
    return tfidf_transform.fit_transform(column)

#Gets the author matrix to check if matching
def get_count_matrix(column):
    column = column.fillna('')
    counter = CountVectorizer(stop_words='english')
    return counter.fit_transform(column)

"""
Ratings data is far to big to fit into a dataframe
Instead I will read it line by line and analyze as necessary
"""


#Provides recommendations
def get_recs(title,df,tm,am,ratings,ID):
    ratings=ratings.fillna('0.0')
    score_array = 10.0*linear_kernel(am[ID],am)+10.0*linear_kernel(tm[ID],tm)
    score_array = list(enumerate(score_array[0]))
    score_array = sorted(score_array, key = lambda x: x[1], reverse=True)
    num_best = 25
    best_ids = [score_array[i][0] for i in range(1,num_best+1)]
    bigger_ids = [score_array[i][0] for i in range(1,100+1)]
    new_array = [(i,df['average_rating2'].iloc[i]) for i in best_ids]
    new_array = sorted(new_array,key=lambda x: x[1],reverse=True)
    best_ids = [new_array[i][0] for i in range(1,11)]
    return [best_ids,bigger_ids]

#Plots most likely authors
def plot_tags(taglist):
    d = {}
    N = taglist.size
    for i in range(N):
        element = taglist.iloc[i]
        if element in d:
            d[element] += 1.0
        else:
            d[element] = 1.0

    sort_data = sorted(d.items(),key=lambda x: x[1], reverse = True)
    maxVal = min(len(sort_data),5)
    best = sort_data[0:maxVal]
    x = []
    y = []
    for i in range(maxVal):
        x.append(best[i][0])
        y.append(best[i][1])

    plt.bar(x,y,edgecolor='k',linewidth=2)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(x,x,fontsize=8)
    plt.ylabel("Count",fontsize=14)
    plt.savefig("Authors.pdf")
    plt.close()
    return

def round_5(num):
    A = round(num/5.0)
    return str(A*5.0)


#Plots a histogram of publication dates for recommended novels to nearest half-decade
def plot_years(taglist):
    d = {}
    N = taglist.size
    for i in range(N):
        element = round_5(taglist.iloc[i])
        if element in d:
            d[element] += 1.0
        else:
            d[element] = 1.0

    sort_data = sorted(d.items(),key=lambda x: x[0], reverse = False)
    x = []
    y = []
    for i in range(len(sort_data)):
        x.append(sort_data[i][0])
        y.append(sort_data[i][1])

    plt.bar(x,y,edgecolor='k',linewidth=2)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(x,x,fontsize=8,rotation='vertical')
    plt.ylabel("Count",fontsize=14)
    plt.savefig("Years.pdf")
    plt.close()
    return


"""
Most useful information for content-based filtering is going to be the
author and the tags associated with each book
"""


data = init_dataframe()
book_ids = pd.Series(data.index,index=data['original_title']).drop_duplicates()
tag_matrix = get_matrix(data['tag_id2'])
author_matrix = get_count_matrix(data['author'])

while True:
    title = input("Please Enter a New Book Name\n")
    try:
        ID = book_ids[title]
        break
    except:
        print("Book Name not Found\n")

if isinstance(book_ids[title],collections.Iterable):
    for j,i in enumerate(book_ids[title]):
        print("Option {}".format(j+1))
        [test,test2] = get_recs(title,tag_matrix,author_matrix,data['average_rating2'],i)
        print(data['original_title'].iloc[test])
else:
    test,test2 = get_recs(title, data,tag_matrix,author_matrix,data['average_rating2'],book_ids[title])
    print(data[['original_title','average_rating']].iloc[test])
    plot_tags(data['author'].iloc[test2])
    plot_years(data['original_publication_year'].iloc[test2])
