import random
import requests
from bs4 import BeautifulSoup


def scrape_page(soup):
    quo = []
    quotes = soup.find_all('p')
    for quote in quotes[3:-5]:
        quo.append(quote.get_text())

    return quo


URL = f"https://ponly.com/best-insults/"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")
quotes = scrape_page(soup)
random_quote = random.choice(quotes)
print(random_quote)

data = {

}
