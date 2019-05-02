import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

pd.set_option('display.float_format', lambda x: '%.4f' % x)
POPULARITY_THRESHOLD = 1000

song_data = pd.read_csv('./song_data.csv')

song_data.fillna(0)
song_data = song_data.drop(columns = ['song_id', 'title', 'artist'])


song_plays = (song_data.
     groupby(by = ['song'])['listen_count'].
     sum().
     reset_index().
     rename(columns = {'listen_count': 'total_song_plays'})
     [['song', 'total_song_plays']])

data_with_song_plays = song_data.merge(song_plays, left_on = 'song', right_on = 'song', how = 'left')

# Uncomment to view data 
# print song_plays.head()
# print song_plays['total_song_plays'].describe()
# print song_plays['total_song_plays'].quantile(np.arange(.9, 1, .01))

popularity_threshold = POPULARITY_THRESHOLD
popular_song_data = data_with_song_plays.query('total_song_plays >= @popularity_threshold')

popular_song_data = popular_song_data.drop_duplicates(['user_id', 'song'])

song_with_users_as_columns_data = popular_song_data.pivot(index = 'song', columns = 'user_id', values = 'listen_count').fillna(0)
song_with_users_as_columns_data_binary = song_with_users_as_columns_data.apply(np.sign)
song_with_users_as_columns_data_binary_sparse = csr_matrix(song_with_users_as_columns_data_binary.values)


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

save_sparse_csr('./sparse_song_matrix_binary.npz', song_with_users_as_columns_data_binary_sparse)
# 
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(song_with_users_as_columns_data_binary_sparse)
pickle.dump(model_knn, open('cf_song_model.sav', 'wb'))


def get_song_recommendations(song_name, song_data, knn_model, num_neighbors):
    query_index = None
    ratios = []
    for i in song_data.index:
        # in song_data, each item is song name AND artist name, divided by a dash. So compare with the full string AND the first after split by dash
        title = i.split(' - ')[0]
        similarity = max(fuzz.ratio(i.lower(), song_name.lower()), fuzz.ratio(title.lower(), song_name.lower()))
        if similarity >= 80:
            index = song_data.index.tolist().index(i)
            ratios.append((i, similarity, index))
    print 'Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratios])

    try:
        query_index = max(ratios, key = lambda x: x[1])[2]
    except:
        print 'Your song didn\'t match any songs in the data. Try again'
        return None

    distances, indices = knn_model.kneighbors(song_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors = num_neighbors + 1)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print 'Recommendations for {0}:\n'.format(song_data.index[query_index])
        else:
            print '{0}: {1}, with distance of {2}:'.format(i, song_data.index[indices.flatten()[i]], distances.flatten()[i])

    return None

# Example call
get_song_recommendations("Learn To Fly - Foo Fighters", song_with_users_as_columns_data_binary, model_knn, 5)
