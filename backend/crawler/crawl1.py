import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random


# Function to fetch job listings from LinkedIn
def fetch_job_listings(search_query, num_pages=1):
    base_url = 'https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search'
    job_listings = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    for page in range(num_pages):
        params = {
            'keywords': search_query,
            'start': page * 25
        }
        response = requests.get(base_url, headers=headers, params=params)
        print(response.status_code)
        if response.status_code != 200:
            print(f"Failed to retrieve page {page + 1}: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        jobs = soup.find_all('li')
        for job in jobs:
            try:
                title = job.find('h3', class_='base-search-card__title').text.strip()
                company = job.find('a', class_='hidden-nested-link').text.strip()
                location = job.find('span', class_='job-search-card__location').text.strip()
                # description = job.find('p', class_='job-result-card__snippet').text.strip()
                date_posted = job.find('time')['datetime']

                job_listings.append({
                    'Title': title,
                    'Company': company,
                    'Location': location,
                    # 'Description': description,
                    'Date Posted': date_posted
                })
            except AttributeError as e:
                print(f"Error parsing job data: {e}")
                continue

        time.sleep(random.uniform(1, 3))  # to avoid being blocked
    # print("job_listings", job_listings)
    return job_listings


# Example usage
if __name__ == '__main__':
    search_query = 'Data Scientist'
    num_pages = 40000  # number of pages to scrape to reach ~1 million job listings
    all_job_listings = []
    print(num_pages // 100)
    if num_pages > 100:
        for i in range(num_pages // 100):
            job_listings = fetch_job_listings(search_query, num_pages=100)
            all_job_listings.extend(job_listings)
            df = pd.DataFrame(all_job_listings)
            df.to_csv(f'linkedin_job_listings_{i}.csv', index=False)
            print(f"Batch {i} saved to linkedin_job_listings_{i}.csv")
    else:
        job_listings = fetch_job_listings(search_query, num_pages=num_pages)
        all_job_listings.extend(job_listings)
        df = pd.DataFrame(all_job_listings)
        df.to_csv(f'linkedin_job_listings.csv', index=False)
        print(f"Batch {0} saved to linkedin_job_listings.csv")

    print("All job listings saved")
