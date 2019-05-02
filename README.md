# CPSC 490 Music Recommendation Engine
Jason Chu

Advisor: Scott Petersen

Spring 2019

This is a music recommendation engine that uses collaborative filtering and content-based filtering to suggest similar songs and artists.

How to use this engine:
Here is the Google Drive folder that contains all the data that you will need for any of the future steps listed here.

If you DON'T want to start from scratch:
Getting collaborative filtering ready:
1) Download the data from Drive: song_data.csv, collaborative_filtering_user_data, and collaborative_filtering_user_profiles from Drive folder. The first has the Last.fm user data, and the last two are python pickles stored after reading a part of the raw csv. Make sure they are in the same directory as all the other files.
2) Download the kNN models from Drive: cf_artist_model.sav and cf_song_model.sav.

Getting content-based filtering ready:
1) Download 4genreweights.best.h5 from Drive, make sure it is in the same directory as all the other files.

If you DO want to start from scratch:
Build the collaborative filtering component:
1) Download song_data.csv from the Drive folder, make sure it is in the same directory as all the other files.
2) Run collaborative_filtering_songs.py to generate the song kNN model.
3) Download collaborative_filtering_user_data and collaborative_filtering_artist_data from Drive, make sure they are in the same directory as all the other files.
4) Run collaborative_filtering_artists.py to generate the artist kNN model.

Build the content-based filtering component
1) Download 4GenreFMA.data, 4GenreFMA.onehotlabels, 4GenreGTZAN.data, 4GenreGTZAN.onehotlabels, 4GenreGTZAN.labels from Drive folder: these are the audio features and one-hot-encoded genre classifications of the dataset. 4GenreFMA is used for training (4000 genre balanced songs) and 4GenreGTZAN is used for testing (400 genre balanced songs).
2) Run cnn.py to generate and save the model

Either way you choose, after all that, run song_recommender.py and follow its instructions. Most importantly, you have to pass in an audio sample that is named in a certain format. It might take sometime for song_recommender to prompt you for an audio sample as it needs to load a huge spreadsheet in order to run.

Description of files:
song_recommender.py: the main thing; run this file to be able to actually get song recommendations from both CF and content-based components

cnn.py: contains CNN code

collaborative_filtering_artists.py: builds the kNN model to find similar artists

collaborative_filtering_songs.py: builds the kNN model to find similar songs

extract_features.py: extracts features from the audio samples

process_fma.py: goes through FMA data to rename and genre tag otherwise unknown songs

visual_spectrogram.py: plots a spectrogram of the features collected from a song using Librosa
