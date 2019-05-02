import numpy as np
import os
from os.path import isfile
import keras
import matplotlib.pyplot as plt
import librosa
from keras.models import Sequential, Model, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization, Lambda
from keras.layers import Input, Dense, LSTM, Dropout, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend, regularizers
from keras.utils import np_utils
from keras.layers.advanced_activations import ELU
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


dict_genres = {'Pop':0, 'Classical':1, 'Hiphop':2, 'Rock':3}

reverse_map = {v: k for k, v in dict_genres.items()}

X_train = joblib.load('4GenreFMA.data')
Y_train = joblib.load('4GenreFMA.onehotlabels')
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)

totalSizeData = X_train.shape[0]
validationDataSize = totalSizeData/10
startingValidationIndex = totalSizeData - validationDataSize

permutation = np.random.permutation(X_train.shape[0])
X_train = X_train[permutation]
Y_train = Y_train[permutation]

X_valid = X_train[startingValidationIndex:totalSizeData]
Y_valid = Y_train[startingValidationIndex:totalSizeData]

X_train = X_train[0:startingValidationIndex]
Y_train = Y_train[0:startingValidationIndex]


NUM_CLASSES = 4
NUM_LAYERS = 3
KERNEL_SIZE = 5
NUM_FILTERS = 56
BATCH_SIZE = 32
NUM_LSTM = 96
NUM_EPOCHS = 100
NUM_HIDDEN = 64
REGULARIZATION = 0.001

def build_crnn_model(model_input):
    print('Building model...')
    layer = model_input
    
    # 3 conv layers
    for i in range(NUM_LAYERS):
        layer = Conv1D(
                filters = NUM_FILTERS,
                kernel_size = KERNEL_SIZE,
                kernel_regularizer = regularizers.l2(REGULARIZATION),
                name = 'convolution_' + str(i + 1)
            )(layer)
        layer = BatchNormalization(momentum=0.9)(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)
        layer = Dropout(0.4)(layer)
    
    # LSTM
    layer = LSTM(NUM_LSTM, return_sequences=False)(layer)
    layer = Dropout(0.4)(layer)
    
    # dense layer
    layer = Dense(NUM_HIDDEN, kernel_regularizer=regularizers.l2(REGULARIZATION), name='dense1')(layer)
    layer = Dropout(0.4)(layer)
    
    # softmax as final layer
    layer = Dense(NUM_CLASSES)(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    model_output = layer
    model = Model(model_input, model_output)

    opt = Adam(lr=0.001)
    model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
    )
    
    print(model.summary())
    return model

def train_model(x_train, y_train, x_val, y_val):
    
    n_features = x_train.shape[2]
    input_shape = (None, n_features)
    model_input = Input(input_shape, name='input')
    
    model = build_crnn_model(model_input)
    
    checkpoint_callback = ModelCheckpoint('./models/crnn/4genreweights.best.h5', monitor='val_acc', verbose=1,
                                          save_best_only=True, mode='max')
    
    reducelr_callback = ReduceLROnPlateau(
                monitor='val_acc', factor=0.5, patience=10, min_delta=0.01, verbose=1)

    print('Training...')
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                        validation_data=(x_val, y_val), verbose=1, callbacks=[checkpoint_callback, reducelr_callback])
    return model, history


def show_summary_stats(history):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

model, history  = train_model(X_train, Y_train, X_valid, Y_valid)
show_summary_stats(history)

X_test = joblib.load('4GenreGTZAN.data')
Y_notonehot = joblib.load('4GenreGTZAN.labels')
Y_test = joblib.load('4GenreGTZAN.onehotlabels')
X_test = np.asarray(X_test)
Y_notonehot = np.asarray(Y_notonehot)
Y_test = np.asarray(Y_test)

permutation = np.random.permutation(X_test.shape[0])
X_test = X_test[permutation]
Y_notonehot = Y_notonehot[permutation]
Y_test = Y_test[permutation]

Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=1)
labels = [0,1,2,3]
target_names = dict_genres.keys()

print(Y_notonehot.shape, Y_pred.shape)
print(classification_report(Y_notonehot, Y_pred, target_names=target_names))

model.evaluate(X_test, Y_test)
print(accuracy_score(Y_notonehot, Y_pred))


