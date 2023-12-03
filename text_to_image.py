import tensorflow as tf
from keras.layers import Dense, LSTM, TimeDistributed
from keras.layers import RepeatVector, Conv2D, Reshape, Conv2DTranspose

import required


def build_text2image_model(use_deconv=True, filters=512):
    text2image = tf.keras.Sequential()
    text2image.add(LSTM(256, input_shape=(None, len(required.unique_characters))))
    text2image.add(RepeatVector(required.max_answer_length))

    if use_deconv:
        text2image.add(TimeDistributed(Dense(7 * 7 * 128, activation="softmax")))
        text2image.add(TimeDistributed(Reshape((7, 7, 128))))
        text2image.add(TimeDistributed(required.BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
        text2image.add(TimeDistributed(required.BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
        text2image.add(TimeDistributed(required.BatchNormalization()))
        text2image.add(TimeDistributed(Conv2D(1, (5, 5), padding="same", activation="sigmoid")))

    text2image.compile(loss='binary_crossentropy', optimizer='adam')  # mse loss increase the accuracy
    text2image.summary()

    return text2image
