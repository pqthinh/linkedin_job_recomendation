from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the job data
jobs_df = pd.read_csv('jobs_data.csv')
jobs_df['combined'] = jobs_df['title'] + ' ' + jobs_df['company']

# Vectorize the text data
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(jobs_df['combined'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    title = data['skills']
    idx = jobs_df[jobs_df['title'].str.contains(title, case=False, na=False)].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar jobs
    
    job_indices = [i[0] for i in sim_scores]
    recommendations = jobs_df.iloc[job_indices].to_dict('records')
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
