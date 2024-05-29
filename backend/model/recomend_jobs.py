import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def recommend_jobs(user_skills, jobs):
    df = pd.DataFrame(jobs)
    tfidf = TfidfVectorizer(stop_words='english')
    df['description'] = df['description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    
    user_skills_vector = tfidf.transform([user_skills])
    cosine_similarities = linear_kernel(user_skills_vector, tfidf_matrix).flatten()
    
    related_docs_indices = cosine_similarities.argsort()[:-5:-1]
    recommendations = df.iloc[related_docs_indices]
    return recommendations

# Example usage
jobs = [
    {'title': 'Python Developer', 'company': 'Company A', 'location': 'New York', 'description': 'Python, Django, Flask'},
    # Add more job data here
]
user_skills = 'Python, Django'
recommended_jobs = recommend_jobs(user_skills, jobs)
print(recommended_jobs)