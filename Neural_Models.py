import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, LSTM, Activation
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalMaxPooling1D, Input, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.merge import add
from xgboost import XGBClassifier
from keras import backend as K
from keras.losses import categorical_crossentropy
from sklearn.naive_bayes import GaussianNB

#MultiLayer Dense Model
def new_Dense(num_classes_in, input_shape):
    num_classes = num_classes_in - 1
    model = Sequential()
    model.add(Dense(num_classes*3, activation='relu', input_shape=input_shape))
    model.add(Dense(int(num_classes*2), activation='relu'))
    model.add(Dense(int(num_classes*1.5), activation='relu'))
    model.add(Dense(int(num_classes*1.25), activation='relu'))

    model.add(Flatten())
    model.add(Dense(num_classes_in, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

#LSTM
def new_original_LSTM(num_classes, input_shape):
    model = Sequential()
    model.add(LSTM(115, input_shape=input_shape))
    #model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
