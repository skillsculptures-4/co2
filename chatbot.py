import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents file
with open('intents.json') as file:
    data = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Prepare patterns and responses
patterns = []
responses = {}
tags = []
tag_labels = {}

for intent in data['intents']:
    tags.append(intent['tag'])
    tag_labels[intent['tag']] = len(tags) - 1
    for pattern in intent['patterns']:
        patterns.append(pattern)
    responses[intent['tag']] = intent['responses']

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(tokenizer=lambda text: [lemmatizer.lemmatize(token.lower()) for token in word_tokenize(text)])
X = vectorizer.fit_transform(patterns)

# Train a classifier
y = np.array([tag_labels[intent['tag']] for intent in data['intents'] for _ in intent['patterns']])
clf = MultinomialNB()
clf.fit(X, y)

def classify_intent(text):
    tokens = [lemmatizer.lemmatize(token.lower()) for token in word_tokenize(text)]
    X_test = vectorizer.transform([" ".join(tokens)])
    prediction = clf.predict(X_test)
    tag = tags[prediction[0]]
    return tag

def get_responses(tag):
    if tag in responses:
        return responses[tag]
    return ["Sorry, I didn't understand that."]

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    print(f"Received message: {user_message}")  # Debug print
    tag = classify_intent(user_message)
    print(f"Classified tag: {tag}")  # Debug print
    responses = get_responses(tag)
    print(f"Responses: {responses}")  # Debug print
    return jsonify({'responses': responses})

if __name__ == "__main__":
    app.run(debug=True)
