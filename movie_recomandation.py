# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 21:21:44 2022

@author: Mamun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



data_credites = pd.read_csv('tmdb_5000_credits.csv')
data_movies = pd.read_csv('tmdb_5000_movies.csv')

data_credites.columns = ['id', 'title', 'cast', 'crew']
data_movies = data_movies.merge(data_credites, on='id')


mean_vote = data_movies['vote_average'].mean()

minimum_votes = data_movies['vote_count'].quantile(0.9)



qualify_movies = data_movies.copy().loc[data_movies['vote_count'] >= minimum_votes]
qualify_movies.shape

def weighted_rating(x, minimum_votes=minimum_votes, mean_vote = mean_vote):
    number_of_votes = x['vote_count']
    vote_avg = x['vote_average']
    
    return (number_of_votes/(number_of_votes + minimum_votes)*vote_avg) + (minimum_votes/(minimum_votes+number_of_votes)*mean_vote)

qualify_movies['score'] = qualify_movies.apply(weighted_rating, axis=1)
qualify_movies = qualify_movies.sort_values('score', ascending=False)
#print(qualify_movies[['title_y', 'vote_count', 'vote_average', 'score']].head(10))

pop_value = qualify_movies.sort_values('popularity', ascending=False)

plt.barh(pop_value['title_y'].head(10), pop_value['popularity'].head(10), align='center', color='skyblue')
plt.xlabel('popularity')
plt.title('Popular Movies')

tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
data_movies['overview'] = data_movies['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(data_movies['overview'])

#Output the shape of tfidf_matrix
#print(tfidf_matrix.shape)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#remove duplicate data
indices = pd.Series(data_movies.index, index=data_movies['title_y']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data_movies['title_y'].iloc[movie_indices]

print(get_recommendations('The Dark Knight Rises'))

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    data_movies[feature] = data_movies[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

data_movies['director'] = data_movies['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    data_movies[feature] = data_movies[feature].apply(get_list)
    
#print(data_movies[['title_y', 'cast', 'director', 'keywords', 'genres']].head(3))
    
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    data_movies[feature] = data_movies[feature].apply(clean_data)
    
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
data_movies['soup'] = data_movies.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data_movies['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

data_movies = data_movies.reset_index()
indices = pd.Series(data_movies.index, index=data_movies['title_y'])

print(get_recommendations('The Godfather', cosine_sim2))



