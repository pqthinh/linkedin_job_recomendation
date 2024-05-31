from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load the model and vectorizer
with open('../model/job_recommendation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('../model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)


# Define the home route
@app.route('/')
def home():
    return "Job Recommendation System API"


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    description = data['description']

    # Transform the description using the vectorizer
    description_vector = vectorizer.transform([description])

    # Predict using the loaded model
    prediction = model.predict(description_vector)

    return jsonify({'job_type': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
