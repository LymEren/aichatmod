import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Sample dataset (positive and negative texts)
texts = [
    "This movie was really great!",
    "A magnificent performance was displayed.",
    "This book was very boring.",
    "I didn't like it at all, a waste of time.",
    "I highly recommend this restaurant.",
    "The quality of this product is very poor.",
    "I had a wonderful experience.",
    "It didn't meet my expectations, I was disappointed."
]

labels = [1, 1, 0, 0, 1, 0, 1, 0]  # Positive (1) or negative (0) labels

# Tokenizing text data
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

# Converting text data to sequences and equalizing lengths
text_sequences = tokenizer.texts_to_sequences(texts)
text_sequences = pad_sequences(text_sequences, maxlen=10).tolist()

# Model creation
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Model training
model.fit(text_sequences, labels, epochs=10, batch_size=2)

# Making a prediction for an example text
example_text = ["The quality of this product is very poor."]
example_text_sequences = tokenizer.texts_to_sequences(example_text)
example_text_sequences = pad_sequences(example_text_sequences, maxlen=10)

prediction = model.predict(example_text_sequences)
if prediction[0] > 0.5:
    print("Positive text.")
else:
    print("Negative text.")

