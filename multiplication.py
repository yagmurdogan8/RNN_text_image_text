# Illustrate the generated query/answer pairs

unique_characters = '0123456789* '  # All unique characters that are used in the queries (13 in total: digits
# 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99  # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))  # Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1  # Maximum length of the query string (consists of two integers and an
# operand [e.g. '22+10'])
max_answer_length = 5  # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
X_text, X_img, y_text, y_img = create_data(highest_integer, operands=['*'])
print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)


# Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1, 2, i + 1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('=' * 50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()


for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])
