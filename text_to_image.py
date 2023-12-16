import keras
import numpy as np
from keras import Sequential
from matplotlib import pyplot as plt
from keras.layers import Dense, LSTM, TimeDistributed
from keras.layers import RepeatVector, Conv2D, Reshape, Conv2DTranspose
from keras.src.layers import BatchNormalization
from sklearn.model_selection import train_test_split
import required


#
#
# def build_text2image_model(use_deconv=True, filters=512):
#     text2image = keras.Sequential()
#     text2image.add(LSTM(256, input_shape=(None, len(required.unique_characters))))
#     text2image.add(RepeatVector(required.max_answer_length))
#
#     if use_deconv:
#         text2image.add(TimeDistributed(Dense(7 * 7 * 128, activation="softmax")))
#         text2image.add(TimeDistributed(Reshape((7, 7, 128))))
#         text2image.add(TimeDistributed(BatchNormalization()))
#         text2image.add(
#             TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
#         text2image.add(TimeDistributed(BatchNormalization()))
#         text2image.add(
#             TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
#         text2image.add(TimeDistributed(BatchNormalization()))
#         text2image.add(TimeDistributed(Conv2D(1, (5, 5), padding="same", activation="sigmoid")))
#
#     text2image.compile(loss='binary_crossentropy', optimizer='adam')
#     text2image.summary()
#
#     return text2image
#
#
# X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot = train_test_split(required.X_text_onehot,
#                                                                                 required.y_img, test_size=0.2)
# # from tensorflow.python.client import device_lib
# # print(device_lib.list_local_devices())
# # print(tf.config.list_physical_devices('GPU'))
# # print("GPU kullanılabilir mi:", tf.test.is_gpu_available())
#
# text2image_model = build_text2image_model()
#
# # Train the model
# text2image_model.fit(X_train_onehot, y_train_onehot, epochs=10, batch_size=32, validation_split=0.2)
#
# # Evaluate the model
# score = text2image_model.evaluate(X_test_onehot, y_test_onehot, batch_size=32)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#

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

# def build_text2image_model(use_deconv=True, filters=256):
#     text2image = keras.Sequential()
#     # text2image.add(Embedding(len(unique_characters), 128, input_length=5))
#     text2image.add(LSTM(256, input_shape=(None, len(required.unique_characters))))
#     # text2image.add(Dense(256))
#     text2image.add(RepeatVector(required.max_answer_length))
#     text2image.add(LSTM(256, return_sequences=True))
#     text2image.add(TimeDistributed(Dense(len(required.unique_characters), activation='softmax')))
#     # text2image.add(TimeDistributed(tf.keras.layers.Lambda(lambda x: tf.one_hot(tf.argmax(x, axis=1),
#     # depth=tf.shape(x)[-1])))) text2image.add(TimeDistributed(Dense(len(unique_characters), activation="softmax")))
#
#     if use_deconv:
#         text2image.add(TimeDistributed(Dense(4 * 4 * 64, activation="softmax")))
#         text2image.add(TimeDistributed(Reshape((4, 4, 64))))
#         text2image.add(TimeDistributed(BatchNormalization()))
#         # 4x4 to 8x8
#         text2image.add(
#             TimeDistributed(Conv2DTranspose(filters, (5, 5), strides=(2, 2), padding="same", activation="relu")))
#         text2image.add(TimeDistributed(BatchNormalization()))
#         # 8x8 to 16x16
#         text2image.add(
#             TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
#         text2image.add(TimeDistributed(BatchNormalization()))
#         # 16x16 to 32x32
#         text2image.add(
#             TimeDistributed(Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same", activation="relu")))
#         text2image.add(TimeDistributed(BatchNormalization()))
#         text2image.add(TimeDistributed(Conv2D(1, (5, 5), padding="same", activation="sigmoid")))
#         text2image.add(TimeDistributed(keras.layers.Resizing(28, 28)))
#
#     text2image.compile(loss='binary_crossentropy', optimizer='adam')  # mse loss increase the accuracy
#     text2image.summary()
#
#     return text2image
#
#
# # We will use this function to display the output of our models throughout this notebook
# def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):
#     if scale:
#         images = (images + 1) / 2.0
#     for index in range(n):
#         plt.subplot(n, n, 1 + index)
#         plt.axis('off')
#         plt.imshow(images[index])
#     fig = plt.gcf()
#     fig.suptitle(name + '  ' + str(epoch), fontsize=14)
#     if save:
#         filename = 'results/generated_plot_e%03d_f.png'
#         plt.savefig(filename)
#         plt.close()
#         plt.show()
#
#
# def graph_accuracy_text2image(splits, epochs):
#     for s in splits:
#         X_train, X_test, y_train, y_test = train_test_split(required.X_text_onehot,
#                                                             required.y_img, train_size=s)
#         model = build_text2image_model()
#         # log_dir = "logs/A2/text2text/split_" + str(s)
#         indices = np.random.randint(0, len(X_train), 5)
#         samples = required.X_text_onehot[indices]
#         print(indices)
#         for epochs in range(epochs):
#             model.fit(x=X_train, y=y_train, epochs=1, batch_size=64)
#             reconstructed = model.predict(samples)
#             for i in range(len(reconstructed)):
#                 grid_plot(reconstructed[i], 1, name='Reconstructed - ' + str(i)
#                                                     + ' - ' + required.decode_labels(samples[i]), n=3, save=False)
#
#
# # All splits to test the accuracy for
# splits_try = [0.7]
# splits = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]
# # Number of epochs
# epochs = 200
# # graph_accuracy_text2image(splits_try, epochs)

def build_text2image_model(use_deconv=True, filters=256):
    text2image = Sequential()
    text2image.add(LSTM(256, input_shape=(None, len(required.unique_characters))))
    text2image.add(RepeatVector(required.max_answer_length))
    text2image.add(LSTM(256, return_sequences=True))
    text2image.add(TimeDistributed(Dense(len(required.unique_characters), activation='softmax')))

    if use_deconv:
        text2image.add(TimeDistributed(Dense(4 * 4 * 64, activation="softmax")))
        text2image.add(TimeDistributed(Reshape((4, 4, 64))))
        text2image.add(TimeDistributed(BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (5, 5), strides=(2, 2), padding="same", activation="relu")))
        text2image.add(TimeDistributed(BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same", activation="relu")))
        text2image.add(TimeDistributed(BatchNormalization()))
        text2image.add(
            TimeDistributed(Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same", activation="relu")))
        text2image.add(TimeDistributed(BatchNormalization()))
        text2image.add(TimeDistributed(Conv2D(1, (5, 5), padding="same", activation="sigmoid")))
        text2image.add(TimeDistributed(keras.layers.Resizing(28, 28)))

    text2image.compile(loss='mean_squared_error', optimizer='adam')
    text2image.summary()

    return text2image


# Function to plot images
def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):
    if scale:
        images = (images + 1) / 2.0
    for index in range(n):
        plt.subplot(n, n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  ' + str(epoch), fontsize=14)
    if save:
        filename = 'results/generated_plot_e%03d_f.png' % epoch
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


# Function to train and visualize the text-to-image model
def graph_accuracy_text2image(splits, epochs):
    for s in splits:
        X_train, X_test, y_train, y_test = train_test_split(required.X_text_onehot, required.y_img, train_size=s)
        model = build_text2image_model()
        indices = np.random.randint(0, len(X_train), 5)
        samples = required.X_text_onehot[indices]
        print(indices)
        for epoch in range(epochs):
            model.fit(x=X_train, y=y_train, epochs=1, batch_size=64)
            reconstructed = model.predict(samples)
            for i in range(len(reconstructed)):
                grid_plot(reconstructed[i], epoch=epoch, name='Reconstructed - ' + str(i), n=3, save=False)



# Example usage
splits_try = [0.7]
splits = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]
epochs = 200
graph_accuracy_text2image(splits_try, epochs)