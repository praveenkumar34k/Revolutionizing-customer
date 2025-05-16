import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Download required data
nltk.download('punkt')

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Preprocess data
sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(sentences)
model = LogisticRegression()
model.fit(X, labels)

# Predict intent
def get_response(user_input):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]

    for intent in data['intents']:
        if intent['tag'] == prediction:
            return random.choice(intent['responses'])

    return "Sorry, I didn't understand that."

# Test (optional)
if __name__ == '__main__':
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        print("Bot:", get_response(user_input))
