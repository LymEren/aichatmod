from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Data Preparation
texts = [
    "This article contains basic information about deep learning.",
    "Deep learning models are widely used in the field of artificial intelligence.",
    "Artificial neural networks are a powerful tool that can be used for text analysis."
]

keywords = [
    "deep learning",
    "artificial intelligence",
    "artificial neural networks",
    "text analysis"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_matrix(texts, mode='binary')

# Model Creation
model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(len(keywords), activation='sigmoid'))

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model (requires training data and labels)
# model.fit(X, y, epochs=10, batch_size=32)

# Making Predictions for New Texts
new_texts = [
    "Deep learning is a powerful technique for artificial intelligence.",
    "Artificial neural networks are useful for text classification."
]

new_X = tokenizer.texts_to_matrix(new_texts, mode='binary')
predictions = model.predict(new_X)

for i, text in enumerate(new_texts):
    print(f"Text: {text}")
    print(f"Predicted Keywords: {', '.join([keywords[j] for j in range(len(keywords)) if predictions[i][j] > 0.5])}")
