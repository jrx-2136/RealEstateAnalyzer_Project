import csv
import requests
from bs4 import BeautifulSoup

class Scraper:
    def __init__(self):
        self.results = []

    def fetch(self, url):
        print(f"HTTP GET request to URL: {url}", end="")
        res = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        print(f" | Status code: {res.status_code}")
        return res

    def parse(self, html):
        soup = BeautifulSoup(html, 'html.parser')

        # MagicBricks listing cards
        cards = soup.select("div.mb-srp__card")

        print(f"Found {len(cards)} listings")

        for card in cards:
            # Title
            title_el = card.select_one("a.mb-srp__card--title")
            title = title_el.get_text(strip=True) if title_el else None

            # Property link
            link = title_el.get("href") if title_el else None
            if link and not link.startswith("http"):
                link = "https://www.magicbricks.com" + link

            # Extract price
            price_el = card.select_one("div.mb-srp__card__price")
            price = price_el.get_text(strip=True) if price_el else None

            # Extract area
            area_el = card.select_one("div.mb-srp__card__summary--value")
            area = area_el.get_text(strip=True) if area_el else None

            # Extract locality
            loc_el = card.select_one("div.mb-srp__card__society")
            locality = loc_el.get_text(strip=True) if loc_el else None

            print(title, "->", link)

            self.results.append({
                "title": title,
                "location": locality,
                "price": price,
                "area_sqft": area,
                "link": link
            })

    def to_csv(self, filename="magicbricks_requests_results.csv"):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["title", "location", "price", "area_sqft", "link"]
            )
            writer.writeheader()
            writer.writerows(self.results)

        print(f"\nSaved {len(self.results)} listings to {filename}")

    def run(self):
        # Mumbai fixed URL (you can later extend city-wise)
        url = "https://www.magicbricks.com/property-for-sale-in-mumbai"

        res = self.fetch(url)
        self.parse(res.text)
        self.to_csv()


if __name__ == "__main__":
    scraper = Scraper()
    scraper.run()