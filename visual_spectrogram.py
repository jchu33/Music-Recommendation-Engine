import numpy as np
import librosa
import librosa.display as display
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

# change FILENAME to file you want to load
FILENAME = "No_Scrubs.mp3"

y, sr = librosa.load(FILENAME)

hop_length = 1024
n_fft = 2048

mel_spec = librosa.feature.melspectrogram(y, sr=sr,hop_length=hop_length,n_fft=n_fft)
log_spec = librosa.core.power_to_db(mel_spec, ref = np.max)

log_spec = log_spec[:, :640]

plt.figure(figsize=(10, 5))
librosa.display.specshow(log_spec, sr=sr, y_axis='mel', x_axis='time')
plt.title('Spectrogram for a Classical Song')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
plt.show()
