import keras
import tensorflow as tf
import numpy as np
import h5py
import librosa
import os.path
import operator
import pickle
import pandas as pd
from keras.models import load_model
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz


SONG_POPULARITY_THRESHOLD = 1000
ARTIST_POPULARITY_THRESHOLD = 40000
NUM_ARTIST_RECOMMENDATIONS = 5
NUM_CF_SONG_RECOMMENDATIONS = 5
NUM_CONTENT_SONG_RECOMMENDATIONS = 5

dict_genres = {'Pop':0, 'Classical':1, 'Hiphop':2, 'Rock':3}
supported_audio_formats = ['mp3', 'au', 'wav', 'ogg', 'flac']

# Set up collaborative filtering component
# Prepare collaborative filtering song data
print "Preparing collaborative filtering component..."
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
song_popularity_threshold = SONG_POPULARITY_THRESHOLD 
popular_song_data = data_with_song_plays.query('total_song_plays >= @song_popularity_threshold')
popular_song_data = popular_song_data.drop_duplicates(['user_id', 'song'])

song_with_users_as_columns_data = popular_song_data.pivot(index = 'song', columns = 'user_id', values = 'listen_count').fillna(0)
song_with_users_as_columns_data_binary = song_with_users_as_columns_data.apply(np.sign)

# Prepare collaborative filtering artist data
user_data = pd.read_pickle('./collaborative_filtering_user_data')
user_profiles = pd.read_pickle('./collaborative_filtering_user_profiles')

if user_data['artist-name'].isnull().sum() > 0:
    user_data = user_data.dropna(axis = 0, subset = ['artist-name'])
artist_plays = (user_data.
     groupby(by = ['artist-name'])['plays'].
     sum().
     reset_index().
     rename(columns = {'plays': 'total_artist_plays'})
     [['artist-name', 'total_artist_plays']])
user_data_with_artist_plays = user_data.merge(artist_plays, left_on = 'artist-name', right_on = 'artist-name', how = 'left')
artist_popularity_threshold = ARTIST_POPULARITY_THRESHOLD
user_data_popular_artists = user_data_with_artist_plays.query('total_artist_plays >= @artist_popularity_threshold')
combined_data_with_country = user_data_popular_artists.merge(user_profiles, left_on = 'users', right_on = 'users', how = 'left')
america_data = combined_data_with_country.query('country == \'United States\'')
america_data = america_data.drop_duplicates(['users', 'artist-name'])
wide_artist_data = america_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)


# Load kNN models
cf_artist_model = pickle.load(open('cf_artist_model.sav', 'rb'))
cf_song_model = pickle.load(open('cf_song_model.sav', 'rb'))

# define functions to call to get recommendations
def full_engine_get_artist_recommendations(artist_name, artist_data, knn_model, num_neighbors):
    query_index = None
    ratios = []
    artist_recommendations = []
    for i in artist_data.index:
        similarity = fuzz.ratio(i.lower(), artist_name.lower())
        if similarity >= 80:
            index = artist_data.index.tolist().index(i)
            ratios.append((i, similarity, index))

    try:
        query_index = max(ratios, key = lambda x: x[1])[2]
    except:
        return artist_recommendations

    distances, indices = knn_model.kneighbors(artist_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors = num_neighbors + 1)
    for i in range(1, len(distances.flatten())):
        artist_recommendations.append(artist_data.index[indices.flatten()[i]])
    return artist_recommendations

def full_engine_get_song_recommendations(song_name, song_data, knn_model, num_neighbors):
	query_index = None
	ratios = []
	song_recommendations = []
	for i in song_data.index:
	# in song_data, each item is song name AND artist name, divided by a dash. So compare with the full string AND the first after split by dash
		title = i.split(' - ')[0].strip()
		similarity = max(fuzz.ratio(i.lower(), song_name.lower()), fuzz.ratio(title.lower(), song_name.lower()))
		if similarity >= 80:
			index = song_data.index.tolist().index(i)
			ratios.append((i, similarity, index))

	try:
		query_index = max(ratios, key = lambda x: x[1])[2]
	except:
		return song_recommendations

	distances, indices = knn_model.kneighbors(song_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors = num_neighbors + 1)
	for i in range(1, len(distances.flatten())):
		song_recommendations.append(song_data.index[indices.flatten()[i]])
	return song_recommendations

# Set up content-based filtering component
print "Preparing content-based filtering component..."
dict_genres = {'Pop':0, 'Classical':1, 'Hiphop':2, 'Rock':3}
# load model
content_based_model = load_model('models/crnn/4genreweights.best.h5')
# load song library with predictions for each song
song_library = pickle.load(open('song_library_predictions.lib', 'rb'))

def predict_genre_cnn(song_filepath, model):
	features = np.empty((0, 640, 128))
	labels = []
	y, sr = librosa.load(song_filepath)
	mel_spec = librosa.feature.melspectrogram(y, sr=sr,n_fft=2048,hop_length=1024)
	log_mel_spec = librosa.core.power_to_db(mel_spec, ref = np.max)
	log_mel_spec = log_mel_spec[:, :640]
	log_mel_spec = log_mel_spec.transpose()
	if log_mel_spec.shape[0] != features.shape[1] or log_mel_spec.shape[1] != features.shape[2]:
		log_mel_spec = np.resize(log_mel_spec,(640,128))
	features = np.append(features, [log_mel_spec], axis=0)
	y_pred = model.predict(features)
	return y_pred

def full_engine_content_recommendations(query_song_data, num_recommendations):
	song_distances = {}
	for song, predictions in song_library.iteritems():
		dist = np.linalg.norm(query_song_data - song_library[song])
		# the distance for the queried song to itself will be 0, so don't add
		if dist != 0:
			song_distances[song] = dist
	sorted_songs = sorted(song_distances.items(), key=operator.itemgetter(1))
	sorted_songs = sorted_songs[:num_recommendations]
	song_titles = [x[0] for x in sorted_songs]
	return song_titles



if __name__== "__main__":
	print "Provide an audio sample to generate song recommendations."
	print "File name of audio sample needs to follow the format: [Song name] - [Artist name]"
	# while True:
	while True:
		song_filepath = raw_input("Input filepath to song: \n")
		type(song_filepath)
		if os.path.isfile(song_filepath):
			# try:
			split_by_period = song_filepath.rsplit('.',1)
			extension = split_by_period[1]
			full_name = split_by_period[0]
			# In form Test - Song.mp3, full_name is Test - Song
			# Split by the dash to get song name and artist name separately
			split_without_dash = full_name.rsplit('-', 1)
			song_name = split_without_dash[0].strip()
			artist_name = split_without_dash[1].strip()
			if extension in supported_audio_formats:
				song_recommendations = full_engine_get_song_recommendations(full_name, song_with_users_as_columns_data, cf_song_model, NUM_CF_SONG_RECOMMENDATIONS)
				artist_recommendations = full_engine_get_artist_recommendations(artist_name, wide_artist_data, cf_artist_model, NUM_ARTIST_RECOMMENDATIONS)
				query_song_data = predict_genre_cnn(song_filepath, content_based_model)
				content_recommendations = full_engine_content_recommendations(query_song_data, NUM_CONTENT_SONG_RECOMMENDATIONS)
				# remove duplicate song names
				full_song_recommendations = song_recommendations + content_recommendations
				full_song_recommendations = list(set(full_song_recommendations))

				print "The recommended songs for " + full_name + " are:"
				for i in range(0, len(full_song_recommendations)):
					print str(i+1) + ". " + full_song_recommendations[i]
				print "The recommended artists for " + artist_name + " are:"
				for i in range(0, len(artist_recommendations)):
					print str(i+1) + ". " + artist_recommendations[i]
			else:
				print "Invalid audio format. Supported audio formats are: MP3, WAV, AU, OGG, FLAC\n"
			# except:
			# 	print "Invalid filename\n"
		else:
			print "File does not exist\n"
