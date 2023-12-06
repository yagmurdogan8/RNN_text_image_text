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
                                                                                required.y_img, test_size=0.2)
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# print(tf.config.list_physical_devices('GPU'))
# print("GPU kullanılabilir mi:", tf.test.is_gpu_available())

text2image_model = build_text2image_model()

# Train the model
text2image_model.fit(X_train_onehot, y_train_onehot, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
score = text2image_model.evaluate(X_test_onehot, y_test_onehot, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Reshape
#
#
# # Assuming you have a function operand_generator() that generates '+', '-' operands
# def operand_generator():
#     operands = ['+', '-']
#     return np.random.choice(operands)
#
#
# # Define the vocabulary
# vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '+', '-']
#
# # Map characters to integers and vice versa
# char_to_int = {char: i for i, char in enumerate(vocab)}
# int_to_char = {i: char for i, char in enumerate(vocab)}
#
# # Define the maximum length of the input and output sequences
# max_len = 5  # Length of input sequence ('89-56')
# output_len = 3  # Length of output sequence ('145')
#
#
# # Function to encode a text sequence into a one-hot encoded array
# def encode_sequence(sequence, vocab_size, max_len):
#     encoded = np.zeros((max_len, vocab_size), dtype=np.float32)
#     for i, char in enumerate(sequence):
#         encoded[i, char_to_int[char]] = 1.0
#     return encoded
#
#
# # Function to generate a dataset of text-to-image pairs with one-hot encoding for images
# def generate_text_to_image_dataset(num_samples):
#     X_text = []
#     y_images = []
#
#     for _ in range(num_samples):
#         num1 = np.random.randint(10, 100)
#         num2 = np.random.randint(10, 100)
#         operand = operand_generator()
#         query = f"{num1}{operand}{num2}"
#         result = str(eval(query))
#
#         # Convert the text query to one-hot encoded array
#         X_text.append(encode_sequence(query, len(vocab), max_len))
#
#         # One-hot encode the images and reshape
#         image_sequence = encode_sequence(result, len(vocab), output_len)
#         y_images.append(image_sequence)
#
#     return np.array(X_text), np.array(y_images)
#
#
# # Generate a dataset of 1000 text-to-image pairs
# num_samples = 1000
# X_text, y_images = generate_text_to_image_dataset(num_samples)
#
# # Build the text-to-image RNN model
# model = Sequential()
# model.add(LSTM(100, input_shape=(max_len, len(vocab))))
# model.add(RepeatVector(output_len))
# model.add(TimeDistributed(Dense(len(vocab), activation='softmax')))
# model.add(Reshape((output_len, len(vocab))))
#
# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_text, y_images, epochs=10, batch_size=32, validation_split=0.2)
