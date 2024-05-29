import requests
from bs4 import BeautifulSoup

def fetch_job_listings():
    url = "https://www.linkedin.com/jobs/search/?keywords=python%20developer"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    jobs = []

    for job_card in soup.find_all('div', class_='result-card'):
        title = job_card.find('h3', class_='result-card__title').text.strip()
        company = job_card.find('h4', class_='result-card__subtitle').text.strip()
        location = job_card.find('span', class_='job-result-card__location').text.strip()
        jobs.append({
            'title': title,
            'company': company,
            'location': location,
            # Add other necessary fields here
        })

    return jobs

if __name__ == "__main__":
    job_listings = fetch_job_listings()
    for job in job_listings:
        print(job)