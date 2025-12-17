import csv
import requests
from bs4 import BeautifulSoup

class Scraper:
    def __init__(self):
        self.results = []

    def fetch(self, url):
        print(f"HTTP GET request to URL: {url}", end="")
        res = requests.get(url)
        print(f" | Status code: {res.status_code}")
        return res

    def parse(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        a_tags = soup.select('h2.heading-6 a')

        for a in a_tags:
            title = a.get_text(strip=True)
            href = a.get('href', '')
            link = "https://www.magicbricks.com" + href if href.startswith('/') else href

            print(title, "->", link)

            self.results.append({
                "title": title,
                "link": link
            })

    def to_csv(self, filename="nobroker_results.csv"):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "link"])
            writer.writeheader()
            writer.writerows(self.results)

        print(f"\nSaved {len(self.results)} listings to {filename}")

    def run(self):
        def build_city_url():
            return f"https://www.magicbricks.com/property-for-sale-in-mumbai"
        url = build_city_url()
        res = self.fetch(url)
        self.parse(res.text)
        self.to_csv()


if __name__ == "__main__":
    scraper = Scraper()
    scraper.run()
