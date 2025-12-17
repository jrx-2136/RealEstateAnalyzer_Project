import csv
import time
import re
from pathlib import Path
from data_cleaner import clean_location_csv
from playwright.sync_api import sync_playwright

CITIES = {
    "Mumbai": "mumbai",
    "Bangalore": "bangalore",
    "New-Delhi": "new-delhi",
    "Ernakulam": "ernakulam",
    "Chennai": "chennai",
    "Hyderabad": "hyderabad",
    "Pune": "pune",
    "Kochi": "kochi",
    "Ahmedabad": "ahmedabad",
    "Gurgaon": "gurgaon",
    "Surat": "surat",
    "Noida": "noida",
    "Vadodara": "vadodara",
    "Jaipur": "jaipur",
    "Lucknow": "lucknow",
    "Kanpur": "kanpur",
    "Nagpur": "nagpur",
    "Indore": "indore",
    "Thane": "thane",
    "Bhopal": "bhopal"
}

# -------------------- HELPERS --------------------

def build_city_url(city_slug):
    return (
        "https://www.magicbricks.com/property-for-sale/"
        "residential-real-estate?"
        "bedroom=2,3&"
        "proptype=Multistorey-Apartment,Builder-Floor-Apartment,"
        "Penthouse,Studio-Apartment,Residential-House,Villa&"
        f"cityName={city_slug}"
    )

def auto_scroll(page, scrolls=6):
    for _ in range(scrolls):
        page.mouse.wheel(0, 4000)
        time.sleep(2)

def safe_text(el):
    return el.inner_text().strip() if el else None

# -------------------- PARSERS --------------------

def parse_price(price_text):
    """
    Returns (total_price_inr, price_per_sqft)
    """
    if not price_text:
        return None, None

    total_price = None
    price_psf = None

    cr_match = re.search(r'([\d.]+)\s*Cr', price_text)
    lac_match = re.search(r'([\d.]+)\s*Lac', price_text)

    if cr_match:
        total_price = float(cr_match.group(1)) * 1e7
    elif lac_match:
        total_price = float(lac_match.group(1)) * 1e5

    psf_match = re.search(r'₹\s*([\d,]+)\s*per\s*sqft', price_text)
    if psf_match:
        price_psf = int(psf_match.group(1).replace(",", ""))

    return total_price, price_psf

def parse_area(area_text):
    """
    Converts area to sqft
    """
    if not area_text:
        return None

    match = re.search(r'([\d.]+)\s*(sqft|sqyrd|sqm)', area_text.lower())
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "sqft":
        return value
    elif unit == "sqyrd":
        return value * 9
    elif unit == "sqm":
        return value * 10.7639

    return None

# -------------------- SCRAPER --------------------

def scrape_listing(card, city_name):
    title_el = card.query_selector("a.mb-srp__card--title")
    price_el = card.query_selector("div.mb-srp__card__price")
    area_el = card.query_selector("div.mb-srp__card__summary--value")
    loc_el = card.query_selector("div.mb-srp__card__society")

    title_text = safe_text(title_el)
    price_raw = safe_text(price_el)
    area_raw = safe_text(area_el)

    link = title_el.get_attribute("href") if title_el else None
    if link and not link.startswith("http"):
        link = "https://www.magicbricks.com" + link

    bedrooms = None
    if title_text and "BHK" in title_text:
        try:
            bedrooms = int(title_text.split("BHK")[0].strip())
        except:
            pass

    price_total, price_psf = parse_price(price_raw)
    area_sqft = parse_area(area_raw)

    return {
        "title": title_text,
        "location": safe_text(loc_el),
        "city": city_name,
        "price_total_inr": price_total,
        "price_per_sqft": price_psf,
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": None,
        "link": link
    }

def delete_csv_file(file_path: str):
    """
    Deletes a CSV file if it exists.
    """
    file_path = Path(file_path)

    if file_path.exists():
        file_path.unlink()
        print(f"[Cleanup] Deleted file → {file_path}")
    else:
        print(f"[Cleanup] File not found, skipping delete → {file_path}")

# -------------------- MAIN --------------------

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
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            time.sleep(5)

            auto_scroll(page)

            cards = page.query_selector_all("div.mb-srp__card")
            print(f"Found {len(cards)} listings in {city_name}")

            for card in cards[:15]:
                try:
                    results.append(scrape_listing(card, city_name))
                except Exception as e:
                    print("Failed listing:", e)

            time.sleep(5)

        browser.close()

    with open(
        "data/outputs/magicbricks_india_properties.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "location",
                "city",
                "price_total_inr",
                "price_per_sqft",
                "area_sqft",
                "bedrooms",
                "bathrooms",
                "link"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved: data/outputs/magicbricks_india_properties.csv")

    clean_location_csv(
        input_csv="data/outputs/magicbricks_india_properties.csv",
        output_csv="data/outputs/magicbricks_india_properties_cleaned.csv",
        drop_empty_location=True
    )
    delete_csv_file("data/outputs/magicbricks_india_properties.csv")


if __name__ == "__main__":
    run()
