import glob
import os
import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras.utils import to_categorical


genreDict = {
    'pop'        :   0,
    'classical'  :   1,
    'hiphop'     :   2,
    'rock'       :   3,
}

np.set_printoptions(threshold=sys.maxsize)
TOTAL_NUM_SONGS = 4
NUM_GENRES = 4
NUM_SONGS_GENRE = TOTAL_NUM_SONGS/NUM_GENRES

def extract_features(basedir, extension):
    features = np.empty((0, 640, 128))
    labels=[]
    hiphop_count = 0
    classical_count = 0
    pop_count = 0
    rock_count = 0
    genre_count_dict = {
        'hiphop' : 0,
        'classical' : 0,
        'pop' : 0,
        'rock' : 0,
        'total' : 0,
    }
    # iterate over all files in all subdirectories of the base directory
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+extension))
        # apply function to all files
        for f in files :
            # exit early once we reach 1000 of each genre
            if genre_count_dict['total'] == TOTAL_NUM_SONGS:
                return (features, to_categorical(np.array(labels)), labels) 
            # NOTE: this line of code depends on directory of data set
            genre = f.split('/')[1]
            if (genre == 'hiphop' or genre == 'classical' or genre == 'pop' or genre == 'rock'):
                genre_count = genre_count_dict[genre]
                if genre_count < NUM_SONGS_GENRE:
                    y, sr = librosa.load(f)
                    # Power spectrogram
                    mel_spec = librosa.feature.melspectrogram(y, sr=sr,n_fft=2048,hop_length=1024)
                    # Convert to log
                    log_mel_spec = librosa.core.power_to_db(mel_spec, ref = np.max)
                    log_mel_spec = log_mel_spec[:, :640]
                    log_mel_spec = log_mel_spec.transpose()

                    if log_mel_spec.shape[0] != features.shape[1] or log_mel_spec.shape[1] != features.shape[2]:
                        continue
                    # store into feature arrays
                    features = np.append(features, [log_mel_spec], axis=0)

                    # get label
                    label = genreDict.get(genre)
                    labels.append(label)

                    genre_count_dict[genre] = genre_count + 1
                    genre_count_dict['total'] = genre_count_dict['total'] + 1
    print features
    print features.shape
    print len(labels)
    print labels

    return (features, to_categorical(np.array(labels)), labels)

if __name__ == "__main__":
    trainingPath = 'combined_songs'
    train_data, one_hot_train_labels, train_labels  = extract_features(trainingPath, '.mp3')

    # store preprocessed data in serialised format so we can save computation time and power
    with open('4GenreFMA.data', 'w') as f:
        pickle.dump(train_data, f)

    with open('4GenreFMA.onehotlabels', 'w') as f:
        pickle.dump(one_hot_train_labels, f)

    with open('4GenreFMA.labels', 'w') as f:
        pickle.dump(train_labels, f)
    print "done"
    exit()
