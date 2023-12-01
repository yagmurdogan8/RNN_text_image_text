import required
import tensorflow as tf
from keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell
from keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose


def build_text2image_model(use_deconv=True, filters=512):
    text2image = tf.keras.Sequential()
    # text2image.add(Embedding(len(unique_characters), 128, input_length=5))
    text2image.add(LSTM(256, input_shape=(None, len(required.unique_characters))))
    # text2image.add(Dense(256))
    text2image.add(RepeatVector(required.max_answer_length))
    # text2image.add(LSTM(256, return_sequences=True))
    # text2image.add(TimeDistributed(Dense(len(unique_characters), activation="softmax")))

    if (use_deconv):
        text2image.add(TimeDistributed(Dense(7 * 7 * 128, activation="softmax")))
        text2image.add(TimeDistributed(Reshape((7, 7, 128))))
        text2image.add(TimeDistributed(required.BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
        # text2image.add(TimeDistributed(Conv2DTranspose(1, (3, 3), strides=(7, 7), padding="same", activation="relu")))
        text2image.add(TimeDistributed(required.BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
        text2image.add(TimeDistributed(required.BatchNormalization()))
        text2image.add(TimeDistributed(Conv2D(1, (5, 5), padding="same", activation="sigmoid")))

    text2image.compile(loss='binary_crossentropy', optimizer='adam')  # mse loss increase the accuracy
    text2image.summary()

    return text2image
