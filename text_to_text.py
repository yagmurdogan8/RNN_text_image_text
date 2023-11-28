import numpy as np
from matplotlib import pyplot as plt
import required
import tensorflow as tf
from keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell
from keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def build_text2text_model():
    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256. In this case the size of our input
    # vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the
    # query will be fed to the network one by one, as shown in the image above (except with 5 elements). Hint: In
    # other applications, where your input sequences have a variable length (e.g. sentences), you would use
    # input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(required.unique_characters))))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times
    # as that's the maximum length of the output (e.g. '  1-99' = '-98') when using 2-digit integers in queries. In
    # other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(required.max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of
    # (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects the first
    # dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence,
    # decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(required.unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text


# Split dataset
X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot = train_test_split(required.X_text_onehot,
                                                                                required.y_text_onehot,
                                                                                test_size=0.1, random_state=42)

text2text = build_text2text_model()

# Train the model
text2text.fit(X_train_onehot, y_train_onehot, epochs=50, batch_size=128, validation_split=0.1)

# Evaluate the model
score = text2text.evaluate(X_test_onehot, y_test_onehot, batch_size=128)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Confusion matrix
y_pred = text2text.predict(X_test_onehot)
y_pred = np.argmax(y_pred, axis=2)
y_test = np.argmax(y_test_onehot, axis=2)

# Flatten the arrays
y_pred = y_pred.flatten()
y_test = y_test.flatten()

cm = confusion_matrix(y_test, y_pred) / len(y_test)

# Plot confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar(format='%d%%')
# Add labels in each cell

plt.show()