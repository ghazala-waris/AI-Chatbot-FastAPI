import json
import random
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

# Load intents
with open("intents.json", "r") as f:
    data = json.load(f)

# Prepare data
patterns = []
labels = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        labels.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

# Vectorize patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = labels

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("responses.pkl", "wb") as f:
    pickle.dump(responses, f)

print("Model trained successfully!")
