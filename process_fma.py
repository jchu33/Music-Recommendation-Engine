import sys
import os
import pandas as pd
import shutil
import math

genres = ['classical', 'hip-hop', 'pop', 'rock']

pd.set_option('display.float_format', lambda x: '%.4f' % x)

track_data = pd.read_csv('./fma_metadata/tracks.csv')

song_library_filename = 'song_library.txt'
song_metadata_filename = 'song_metadata.txt'

# basedir = base directory that contains all of the other directories of fma data
def extract_songs_for_genres(basedir, dest_base_path, extension):
	row_count = track_data.shape[0]
	col_count = track_data.shape[1]
	row_of_labels = track_data.iloc[0]
	track_id_index = None
	genre_index = None
	track_title_index = None
	artist_name_index = None
	for i in xrange(0, col_count):
		label = track_data.iloc[0, i]
		# fma data set has this weird quirk where the track_id column doesn't have a label and is just NaN
		if isinstance(label, float) and track_id_index == None:
			track_id_index = i
		elif label == 'genre_top':
			genre_index = i
		elif label == 'title' and genre_index != None:
			# this must be the track title and not the album title since the genre has been filled in
			track_title_index = i
		elif label == 'name':
			artist_name_index = i
		if genre_index != None and track_title_index != None and artist_name_index != None:
			break

	# skip 0th row cause that is just the labels
	hiphop_count = 0
	classical_count = 0
	rock_count = 0
	pop_count = 0
	for row_num in xrange(1, row_count):
		genre = track_data.iloc[row_num, genre_index]
		# not NaN genre
		if isinstance(genre, basestring):
			genre = genre.lower()
			if (genre == 'hip-hop' or genre == 'classical' or genre == 'rock' or genre == 'pop'):
				if genre == 'hip-hop':
					genre = 'hiphop'

				track_id = track_data.iloc[row_num, track_id_index]
				filename = convert_track_id_to_filename(track_id)
				track_title = str(track_data.iloc[row_num, track_title_index]).strip()
				artist_name = str(track_data.iloc[row_num, artist_name_index]).strip()
				prefix_directory = filename[:3]
				src_full_path = basedir + '/' + prefix_directory + '/' + filename + extension

				song_library_file = open(song_library_filename, "a")
				song_library_file.write(track_title + ' - ' + artist_name + '\n')

				song_metadata_file = open(song_metadata_filename, "a")
				song_metadata_file.write(filename + '\t' + track_title + '\t' + artist_name + '\t' + genre + '\n')

				if is_valid_for_filename(track_title) and is_valid_for_filename(artist_name):
					dest_full_path = dest_base_path + '/' + genre + '/' + track_title + ' - ' + artist_name + '.mp3'
					shutil.copyfile(src_full_path, dest_full_path)

				else:
					dest_full_path = dest_base_path + '/' + genre + '/' + filename + '.mp3'
					shutil.copyfile(src_full_path, dest_full_path)

				if genre == 'hiphop':
					hiphop_count += 1
				elif genre == 'classical':
					classical_count += 1
				elif genre == 'rock':
					rock_count += 1
				elif genre == 'pop':
					pop_count += 1

	print "Hiphop: " + str(hiphop_count)
	print "Classical: " + str(classical_count)
	print "Rock: " + str(rock_count)
	print "Pop: " + str(pop_count)


def convert_track_id_to_filename(track_id):
	max_length_of_filename = 6
	filename = str(track_id)
	prefix_of_zeros = "0" * (6 - len(filename))
	return prefix_of_zeros + filename

def is_valid_for_filename(word):
	# from: https://docs.microsoft.com/en-us/windows/desktop/FileIO/naming-a-file
	# invalid_chars = ['<', '>', ':', '\"', '/', '\\', '|', '?', '*']
	invalid_chars = [':', '/', '\0']
	for char in invalid_chars:
		if char in word:
			return False
	return True

if __name__ == "__main__":
    training_path = 'fma_large'
    dest_base_path = 'combined_songs'
    extract_songs_for_genres(training_path, dest_base_path, '.mp3')
    exit()

