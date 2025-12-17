import csv
import time
from playwright.sync_api import sync_playwright

CITIES = {
    "Mumbai": "mumbai",
    "Bangalore": "bangalore",
    "New-Delhi": "new-delhi",
    "Ernakulam":"ernakulam"
}

def build_city_url(city_slug):
    return f"https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=2,3&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&cityName={city_slug}"

def auto_scroll(page, scrolls=6):
    for _ in range(scrolls):
        page.mouse.wheel(0, 4000)
        time.sleep(2)

def safe_text(el):
    return el.inner_text().strip() if el else None

def scrape_listing(card, city_name):
    title_el = card.query_selector("a.mb-srp__card--title")
    price_el = card.query_selector("div.mb-srp__card__price")
    area_el = card.query_selector("div.mb-srp__card__summary--value")
    loc_el = card.query_selector("div.mb-srp__card__society")

    link = title_el.get_attribute("href") if title_el else None
    if link and not link.startswith("http"):
        link = "https://www.magicbricks.com" + link

    bedrooms = None
    title_text = safe_text(title_el)
    if title_text and "BHK" in title_text:
        try:
            bedrooms = int(title_text.split("BHK")[0].strip())
        except:
            pass

    return {
    "title": title_text,
    "location": safe_text(loc_el),
    "city": city_name,
    "price": safe_text(price_el),
    "area_sqft": safe_text(area_el),
    "bedrooms": bedrooms,
    "bathrooms": None,
    "link": link
    }

def run():
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            slow_mo=30,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        page = context.new_page()

        for city_name, city_slug in CITIES.items():
            print(f"\n===== Scraping {city_name} =====")

            url = build_city_url(city_slug)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
            except:
                print("Retrying page load...")
                time.sleep(5)
                page.goto(url, wait_until="domcontentloaded", timeout=60000)

            time.sleep(5)
            auto_scroll(page)

            cards = page.query_selector_all("div.mb-srp__card")
            print(f"Found {len(cards)} listings in {city_name}")

            for card in cards[:20]:  # limit per city
                try:
                    data = scrape_listing(card, city_name)
                    results.append(data)
                except Exception as e:
                    print("Failed listing:", e)

            time.sleep(4)

        browser.close()

    with open("data/outputs/magicbricks_india_properties.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["title", "location", "city","price", "area_sqft","bedrooms", "bathrooms", "link"]
        )   
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved: data/outputs/magicbricks_india_properties.csv")

if __name__ == "__main__":
    run()
