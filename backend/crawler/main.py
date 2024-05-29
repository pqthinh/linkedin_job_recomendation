import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_jobs_data():
    jobs = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for page in range(1, 3):  # Limit the number of pages to crawl for demonstration
        url = f'https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=software%20engineer&location=Worldwide&start={page*25}'
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for job_card in soup.find_all('li'):
            title = job_card.find('span', class_='screen-reader-text').text.strip()
            company = job_card.find('a', class_='hidden-nested-link').text.strip()
            location = job_card.find('span', class_='job-result-card__location').text.strip()
            job_link = job_card.find('a', class_='result-card__full-card-link')['href']
            
            jobs.append({
                'title': title,
                'company': company,
                'location': location,
                'link': job_link
            })
            
    return pd.DataFrame(jobs)

jobs_df = fetch_jobs_data()
jobs_df.to_csv('jobs_data.csv', index=False)
print(jobs_df.head())
