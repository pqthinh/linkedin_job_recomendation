import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load job listings data
df = pd.read_csv('../crawler/linkedin_job_listings_2.csv')

# Preprocessing and feature extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['Title'])

# Example target variable: job type (e.g., Data Scientist, Software Engineer, etc.)
# For simplicity, assume the 'Title' column contains the job type
df['Job Type'] = df['Title'].apply(lambda x: 'Data Scientist' if 'Data Scientist' in x else 'Other')
y = df['Job Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Data Scientist')
recall = recall_score(y_test, y_pred, pos_label='Data Scientist')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Save the model and vectorizer for future use
import pickle
with open('job_recommendation_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
