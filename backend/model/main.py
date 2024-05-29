import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the job data
jobs_df = pd.read_csv('jobs_data.csv')

# Combine title and company for vectorization
jobs_df['combined'] = jobs_df['title'] + ' ' + jobs_df['company']

# Vectorize the text data
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(jobs_df['combined'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = jobs_df[jobs_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar jobs
    
    job_indices = [i[0] for i in sim_scores]
    return jobs_df.iloc[job_indices]

# Example usage
recommended_jobs = get_recommendations('Software Engineer')
print(recommended_jobs)
