import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM, TimeDistributed
from keras.layers import RepeatVector, Conv2D, Reshape, Conv2DTranspose
from keras.src.layers import BatchNormalization, Embedding
from sklearn.model_selection import train_test_split
import required


def build_text2image_model(num_samples=1000):
    texts = []
    images = []

    for _ in range(num_samples):
        num1, num2 = np.random.randint(10, 100, size=2)
        operation = np.random.choice(['+', '-'])

        if operation == '+':
            result = num1 + num2
            texts.append(f"{num1}+{num2}")
        else:
            result = num1 - num2
            texts.append(f"{num1}-{num2}")

        result_str = str(result).zfill(3)
        image = np.array(list(result_str)).astype(np.uint8)  # Image representation
        images.append(image)

    return texts, np.array(images)


texts, images = build_text2image_model(num_samples=1000)

# Tokenize the texts
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(texts)
total_chars = len(tokenizer.word_index) + 1

# Create input sequences and padded sequences
input_sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = max(len(seq) for seq in input_sequences)
padded_sequences = keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length,
                                                              padding='post')

# Convert images to one-hot encoding
images_onehot = keras.utils.to_categorical(images, num_classes=10)

# Build the text-to-image model
embedding_dim = 50  # Adjust this based on your task
model = keras.Sequential([
    Embedding(input_dim=total_chars, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(100),  # Adjust the number of units based on your task
    RepeatVector(3),  # Repeat the vector to match the output shape (3 digits)
    TimeDistributed(Dense(10, activation='softmax'))  # Output layer with 10 units for each digit
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(padded_sequences, images_onehot, epochs=50, batch_size=32)

# Now, given a new text input, you can use the trained model to generate an image:
new_text = ["15+23"]
new_sequence = tokenizer.texts_to_sequences(new_text)
new_padded_sequence = keras.preprocessing.sequence.pad_sequences(new_sequence, maxlen=max_sequence_length,
                                                                 padding='post')
predicted_image_onehot = model.predict(new_padded_sequence)

# Convert one-hot encoding back to image representation
predicted_image = np.argmax(predicted_image_onehot, axis=-1)

# Print the predicted image
print("Predicted Image:", predicted_image)

#
# def build_text2image_model(use_deconv=True, filters=512):
#     text2image = tf.keras.Sequential()
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
#         text2image.add(TimeDistributed(Conv2D(13, (5, 5), padding="same", activation="softmax")))
#
#     text2image.compile(loss='binary_crossentropy', optimizer='adam')  # mse loss increase the accuracy
#     text2image.summary()
#
#     return text2image
#
#
# X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot = train_test_split(required.X_text_onehot,
#                                                                                 required.y_text_onehot,
#                                                                                 test_size=0.1, random_state=42)
#
# text2text = build_text2image_model()
# # print("ytest", y_test_onehot.shape, "ytrain", y_train_onehot.shape)
#
# # Train the model
# text2text.fit(X_train_onehot, y_train_onehot, epochs=50, batch_size=128, validation_split=0.1)
#
# # Evaluate the model
# score = text2text.evaluate(X_test_onehot, y_test_onehot, batch_size=128)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# # Confusion matrix
# y_pred = text2text.predict(X_test_onehot)
# y_pred = np.argmax(y_pred, axis=2)
# y_test = np.argmax(y_test_onehot, axis=2)
#
# # Flatten the arrays
# y_pred = y_pred.flatten()
# y_test = y_test.flatten()
