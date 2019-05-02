import pandas as pd
import numpy as np
import string as str
import pickle
from fuzzywuzzy import fuzz
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

pd.set_option('display.float_format', lambda x: '%.4f' % x)
POPULARITY_THRESHOLD = 40000

user_data = pd.read_pickle('./collaborative_filtering_user_data')
user_profiles = pd.read_pickle('./collaborative_filtering_user_profiles')


# remove data that does not have artist's name
if user_data['artist-name'].isnull().sum() > 0:
    user_data = user_data.dropna(axis = 0, subset = ['artist-name'])

# sum the plays for an artist across all users
artist_plays = (user_data.
     groupby(by = ['artist-name'])['plays'].
     sum().
     reset_index().
     rename(columns = {'plays': 'total_artist_plays'})
     [['artist-name', 'total_artist_plays']])

user_data_with_artist_plays = user_data.merge(artist_plays, left_on = 'artist-name', right_on = 'artist-name', how = 'left')


popularity_threshold = POPULARITY_THRESHOLD
user_data_popular_artists = user_data_with_artist_plays.query('total_artist_plays >= @popularity_threshold')

combined_data_with_country = user_data_popular_artists.merge(user_profiles, left_on = 'users', right_on = 'users', how = 'left')
america_data = combined_data_with_country.query('country == \'United States\'')

# remove duplicate entries
america_data = america_data.drop_duplicates(['users', 'artist-name'])
wide_artist_data = america_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

wide_artist_data_sparse = csr_matrix(wide_artist_data.values)
save_sparse_csr('./lastfm_sparse_artist_matrix.npz', wide_artist_data_sparse)


# fit the model
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(wide_artist_data_sparse)
pickle.dump(model_knn, open('cf_artist_model.sav', 'wb'))


def get_artist_recommendations(artist_name, artist_data, knn_model, num_neighbors):
    query_index = None
    ratios = []
    for i in artist_data.index:
        similarity = fuzz.ratio(i.lower(), artist_name.lower())
        if similarity >= 80:
            index = artist_data.index.tolist().index(i)
            ratios.append((i, similarity, index))
    print 'Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratios])

    try:
        query_index = max(ratios, key = lambda x: x[1])[2]
    except:
        print 'Your artist didn\'t match any artists in the data. Try again'
        return None

    distances, indices = knn_model.kneighbors(artist_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors = num_neighbors + 1)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print 'Recommendations for {0}:\n'.format(artist_data.index[query_index])
        else:
            print '{0}: {1}, with distance of {2}:'.format(i, artist_data.index[indices.flatten()[i]], distances.flatten()[i])

    return None

# model_knn = pickle.load(open('collaborative_filtering_knn_model.sav', 'rb'))

# Example of how to call
get_artist_recommendations('red hot chili peppers', wide_artist_data, model_knn, 10)
