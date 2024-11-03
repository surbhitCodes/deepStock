import requests
from bs4 import BeautifulSoup, Comment
from pymongo import MongoClient
from datetime import datetime

from tqdm import tqdm

# MongoDB connection
client = MongoClient('localhost', 27017)
db = client['deepStockDB']
news_collection = db['news_collection']


# Function to insert news articles into MongoDB
def insert_news(news_data):
    inserted_count = 0
    duplicate_count = 0

    for news in news_data:
        try:
            # Check if the news article already exists (based on headline)
            if not news_collection.find_one({'news_heading': news['news_heading']}):
                # Insert the article if it's unique
                result = news_collection.insert_one(news)
                inserted_count += 1
            else:
                duplicate_count += 1

        except Exception as e:
            print(f"Error inserting document: {news['news_heading']} - Error: {e}")

    print(f"\nSummary: {inserted_count} new articles inserted, {duplicate_count} duplicates found.")


# Scraping function
def scrape_news():
    urls = ['https://www.moneycontrol.com/news/business/stocks/',
            'https://economictimes.indiatimes.com/markets/stocks/news']
    news_data = []
    for url in urls:
        print(f'Scraping {url}')
        if url.startswith('https://economictimes.indiatimes.com'):
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Assuming articles are in <div> tags with the class 'eachStory'
            articles = soup.find_all('div', class_='eachStory')
            for article in tqdm(articles, desc=f"Processing articles from {url}"):
                # Extract news heading from <a> tag inside the <div>
                title_tag = article.find('a')
                news_heading = title_tag.get_text(strip=True) if title_tag else 'N/A'

                # Extract the link to the article
                news_link = title_tag['href'] if title_tag else 'N/A'

                # Extract publication date
                date_tag = article.find('time')
                date = date_tag.get_text(strip=True) if date_tag else datetime.today().strftime('%Y-%m-%d')

                # Append the news data to the list
                news_data.append({
                    'date': date,
                    'news_heading': news_heading,
                    'link': news_link
                })
        else:
            page_cnt = 1
            while True:
                print(f'Scraping page {page_cnt}')
                paginated_url = f'{url}page-{page_cnt}/'
                page_cnt += 1

                response = requests.get(paginated_url)
                soup = BeautifulSoup(response.content, 'html.parser')

                # Terminate if "No Content is Available" is found
                no_content_tag = soup.find('h2')
                if no_content_tag and "no content is available" in no_content_tag.get_text(strip=True).lower():
                    break

                articles = soup.find_all('li', class_='clearfix')
                if not articles:
                    break

                for article in tqdm(articles, desc=f"Processing articles from {url}"):
                    # Extract news heading
                    title_tag = article.find('h2')
                    news_heading = title_tag.get_text(strip=True) if title_tag else 'N/A'

                    # Extract the link to the article
                    link_tag = article.find('a')
                    news_link = link_tag['href'] if link_tag else 'N/A'

                    # Extract publication date from commented HTML
                    comments = article.find_all(string=lambda text: isinstance(text, Comment))
                    date = 'N/A'
                    for comment in comments:
                        if 'AM' in comment or 'PM' in comment:  # Assuming the date contains 'AM' or 'PM'
                            comment_soup = BeautifulSoup(comment, 'html.parser')
                            date = comment_soup.get_text(strip=True) if comment_soup else 'N/A'
                            break

                    news_data.append({
                        'date': date,
                        'news_heading': news_heading,
                        'link': news_link
                    })

    # Insert news into MongoDB
    insert_news(news_data)


if __name__ == '__main__':
    scrape_news()
