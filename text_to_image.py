import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM, TimeDistributed
from keras.layers import RepeatVector, Conv2D, Reshape, Conv2DTranspose
from keras.src.layers import BatchNormalization
from sklearn.model_selection import train_test_split
import required


def build_text2image_model(use_deconv=True, filters=512):
    text2image = keras.Sequential()
    text2image.add(LSTM(256, input_shape=(None, len(required.unique_characters))))
    text2image.add(RepeatVector(required.max_answer_length))

    if use_deconv:
        text2image.add(TimeDistributed(Dense(7 * 7 * 128, activation="softmax")))
        text2image.add(TimeDistributed(Reshape((7, 7, 128))))
        text2image.add(TimeDistributed(BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
        text2image.add(TimeDistributed(BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
        text2image.add(TimeDistributed(BatchNormalization()))
        text2image.add(TimeDistributed(Conv2D(1, (5, 5), padding="same", activation="sigmoid")))

    text2image.compile(loss='binary_crossentropy', optimizer='adam')
    text2image.summary()

    return text2image


X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot = train_test_split(required.X_text_onehot,
                                                                                required.y_img, test_size=0.1)
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# print(tf.config.list_physical_devices('GPU'))
# print("GPU kullanılabilir mi:", tf.test.is_gpu_available())

text2image_model = build_text2image_model()

#Train the model
text2image_model.fit(X_train_onehot, y_train_onehot, epochs=50, batch_size=128, validation_split=0.1)

#Evaluate the model
score = text2image_model.evaluate(X_test_onehot, y_test_onehot, batch_size=128)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def build_text2image_model(use_deconv=True, filters=512):
    text2image = tf.keras.Sequential()
    text2image.add(LSTM(256, input_shape=(None, len(required.unique_characters))))
    text2image.add(RepeatVector(required.max_answer_length))

    if use_deconv:
        text2image.add(TimeDistributed(Dense(7 * 7 * 128, activation="softmax")))
        text2image.add(TimeDistributed(Reshape((7, 7, 128))))
        text2image.add(TimeDistributed(BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
        text2image.add(TimeDistributed(BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
        text2image.add(TimeDistributed(BatchNormalization()))
        text2image.add(TimeDistributed(Conv2D(13, (5, 5), padding="same", activation="softmax")))

    text2image.compile(loss='binary_crossentropy', optimizer='adam')  # mse loss increase the accuracy
    text2image.summary()

    return text2image


X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot = train_test_split(required.X_text_onehot,
                                                                                required.y_text_onehot,
                                                                                test_size=0.1, random_state=42)

text2text = build_text2image_model()
print("ytest", y_test_onehot.shape, "ytrain", y_train_onehot.shape)

# Train the model
text2text.fit(X_train_onehot, y_train_onehot, epochs=50, batch_size=128, validation_split=0.1)

# Evaluate the model
score = text2text.evaluate(X_test_onehot, y_test_onehot, batch_size=128)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

