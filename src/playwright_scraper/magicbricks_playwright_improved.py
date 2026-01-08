import csv
import time
import re
import logging
import sys
from pathlib import Path
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from data_cleaner import clean_location_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.logs'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CITIES = [
    "mumbai", "thane", "navi-mumbai", "pune", "new-delhi", "noida", "gurgaon", 
    "jaipur", "udaipur", "ahmedabad", "surat", "vadodara", "indore", "bhopal", 
    "lucknow", "kanpur", "patna", "gaya", "kolkata", "howrah", "bhubaneswar", 
    "cuttack", "ranchi", "jamshedpur", "raipur", "bilaspur", "nagpur", "nashik", 
    "aurangabad", "bengaluru", "mysore", "chennai", "coimbatore", "hyderabad", 
    "warangal", "kochi", "trivandrum", "thrissur", "kottayam"
]

# CSS Selectors with fallbacks
SELECTORS = {
    'card': ['div.mb-srp__card', 'div[data-property-card]'],
    'title': ['a.mb-srp__card--title', 'a.property-title'],
    'price': ['div.mb-srp__card__price', 'span.property-price'],
    'area': ['div.mb-srp__card__summary--value', 'span.property-area'],
    'location': ['div.mb-srp__card__society', 'div.property-location']
}

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
PAGE_LOAD_TIMEOUT = 60000  # ms
PAGINATION_THRESHOLD = 20  # Stop if listings aren't increasing


def build_city_url(city_slug, bedroom_filter="2,3"):
    """
    Build MagicBricks URL with configurable filters.
    
    Args:
        city_slug: City name
        bedroom_filter: Comma-separated bedroom counts (default: "2,3")
    """
    return (
        "https://www.magicbricks.com/property-for-sale/"
        "residential-real-estate?"
        f"bedroom={bedroom_filter}&"
        "proptype=Multistorey-Apartment,Builder-Floor-Apartment,"
        "Penthouse,Studio-Apartment,Residential-House,Villa&"
        f"cityName={city_slug}"
    )


def wait_for_element(page, selectors, timeout=10000):
    """
    Try multiple selector strategies to find and wait for an element.
    
    Args:
        page: Playwright page object
        selectors: List of CSS selectors to try
        timeout: Timeout in milliseconds
    
    Returns:
        Element or None
    """
    for selector in selectors:
        try:
            page.wait_for_selector(selector, timeout=timeout)
            return page.query_selector(selector)
        except PlaywrightTimeoutError:
            continue
    return None


def smart_scroll(page, max_scrolls=20):
    """
    Intelligently scroll to load all dynamic content.
    
    Args:
        page: Playwright page object
        max_scrolls: Maximum scroll attempts
    
    Returns:
        Number of actual scrolls performed
    """
    previous_height = page.evaluate("document.body.scrollHeight")
    scrolls = 0
    
    for i in range(max_scrolls):
        page.mouse.wheel(0, 4000)
        time.sleep(1.5)
        
        new_height = page.evaluate("document.body.scrollHeight")
        
        # If height didn't change, we've loaded all content
        if new_height == previous_height:
            logger.info(f"Reached end of content after {scrolls} scrolls")
            return scrolls
        
        previous_height = new_height
        scrolls += 1
    
    logger.warning(f"Max scrolls ({max_scrolls}) reached, may not have all content")
    return scrolls


def safe_text(el):
    """Safely extract and clean text from element."""
    try:
        return el.inner_text().strip() if el else None
    except Exception as e:
        logger.debug(f"Error extracting text: {e}")
        return None


def parse_price(price_text):
    """
    Parse price text and return (total_price_inr, price_per_sqft).
    Handles: "₹ 2.5 Cr", "45 Lac", "₹ 2,500 per sqft"
    """
    if not price_text:
        return None, None

    total_price = None
    price_psf = None

    # Handle Crores
    cr_match = re.search(r'([\d.]+)\s*Cr', price_text, re.IGNORECASE)
    if cr_match:
        try:
            total_price = float(cr_match.group(1)) * 1e7
        except ValueError:
            logger.warning(f"Failed to parse Cr price: {price_text}")

    # Handle Lacs (if Cr not found)
    if not total_price:
        lac_match = re.search(r'([\d.]+)\s*Lac', price_text, re.IGNORECASE)
        if lac_match:
            try:
                total_price = float(lac_match.group(1)) * 1e5
            except ValueError:
                logger.warning(f"Failed to parse Lac price: {price_text}")

    # Handle per sqft price
    psf_match = re.search(r'₹\s*([\d,]+)\s*(?:per\s*)?sqft', price_text, re.IGNORECASE)
    if psf_match:
        try:
            price_psf = int(psf_match.group(1).replace(",", ""))
        except ValueError:
            logger.warning(f"Failed to parse per sqft price: {price_text}")

    return total_price, price_psf


def parse_area(area_text):
    """
    Convert area to sqft. Handles: "1500 sqft", "450 sqyrd", "200 sqm"
    """
    if not area_text:
        return None

    match = re.search(r'([\d.]+)\s*(sqft|sqyrd|sqm)', area_text.lower())
    if not match:
        return None

    try:
        value = float(match.group(1))
        unit = match.group(2).lower()

        if unit == "sqft":
            return value
        elif unit == "sqyrd":
            return value * 9
        elif unit == "sqm":
            return value * 10.7639
    except ValueError:
        logger.warning(f"Failed to parse area: {area_text}")

    return None


def parse_bedrooms(title_text):
    """
    Extract number of bedrooms from title.
    Handles: "2 BHK", "3BHK", "Studio", "1 RK"
    """
    if not title_text:
        return None

    # Special cases
    if "studio" in title_text.lower():
        return 0
    
    if "1 rk" in title_text.lower() or "1rk" in title_text.lower():
        return 1

    # Standard BHK parsing
    try:
        match = re.search(r'(\d+)\s*(?:bhk|bhk)', title_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    except (ValueError, AttributeError):
        pass

    return None


def scrape_listing(card, city_name):
    """
    Extract data from a single property card.
    
    Args:
        card: Playwright element handle for the card
        city_name: City name
    
    Returns:
        Dictionary with property data or None if extraction fails
    """
    try:
        # Extract elements with fallback selectors
        title_el = None
        for selector in SELECTORS['title']:
            title_el = card.query_selector(selector)
            if title_el:
                break
        
        price_el = None
        for selector in SELECTORS['price']:
            price_el = card.query_selector(selector)
            if price_el:
                break
        
        area_el = None
        for selector in SELECTORS['area']:
            area_el = card.query_selector(selector)
            if area_el:
                break
        
        loc_el = None
        for selector in SELECTORS['location']:
            loc_el = card.query_selector(selector)
            if loc_el:
                break

        # Extract text
        title_text = safe_text(title_el)
        price_raw = safe_text(price_el)
        area_raw = safe_text(area_el)
        location_text = safe_text(loc_el)

        # Extract link
        link = None
        if title_el:
            try:
                link = title_el.get_attribute("href")
                if link and not link.startswith("http"):
                    link = "https://www.magicbricks.com" + link
            except Exception as e:
                logger.debug(f"Failed to extract link: {e}")

        # Parse bedrooms
        bedrooms = parse_bedrooms(title_text)

        # Parse price
        price_total, price_psf = parse_price(price_raw)

        # Parse area
        area_sqft = parse_area(area_raw)

        return {
            "title": title_text,
            "location": location_text,
            "city": city_name,
            "price_total_inr": price_total,
            "price_per_sqft": price_psf,
            "area_sqft": area_sqft,
            "bedrooms": bedrooms,
            "bathrooms": None,  # Not available in current HTML
            "link": link
        }

    except Exception as e:
        logger.error(f"Error scraping listing: {e}")
        return None


def scrape_city(page, city, max_listings=None):
    """
    Scrape all listings for a single city.
    
    Args:
        page: Playwright page object
        city: City name
        max_listings: Maximum listings to scrape (None = all)
    
    Returns:
        List of property dictionaries
    """
    listings = []
    attempt = 0
    
    while attempt < MAX_RETRIES:
        try:
            url = build_city_url(city)
            logger.info(f"Loading {city} (attempt {attempt + 1}/{MAX_RETRIES}): {url}")
            
            page.goto(url, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
            
            # Wait for cards to appear
            page.wait_for_selector("div.mb-srp__card", timeout=15000)
            time.sleep(2)
            
            # Smart scroll to load all content
            smart_scroll(page, max_scrolls=20)
            
            # Get all cards
            cards = page.query_selector_all("div.mb-srp__card")
            logger.info(f"Found {len(cards)} total listings in {city}")
            
            if max_listings:
                cards = cards[:max_listings]
            
            # Scrape each card
            successful = 0
            for idx, card in enumerate(cards):
                listing = scrape_listing(card, city)
                if listing and listing.get('price_total_inr') or listing.get('area_sqft'):
                    listings.append(listing)
                    successful += 1
                else:
                    logger.debug(f"Skipped incomplete listing {idx + 1}/{len(cards)}")
            
            logger.info(f"Successfully scraped {successful}/{len(cards)} listings from {city}")
            return listings
            
        except PlaywrightTimeoutError:
            attempt += 1
            logger.warning(f"Timeout for {city}, retrying... ({attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            attempt += 1
            logger.error(f"Error scraping {city}: {e}, retrying... ({attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    
    logger.error(f"Failed to scrape {city} after {MAX_RETRIES} attempts")
    return listings


def save_to_csv(results, output_path):
    """Save results to CSV file."""
    if not results:
        logger.warning("No results to save")
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "title", "location", "city", "price_total_inr", 
        "price_per_sqft", "area_sqft", "bedrooms", "bathrooms", "link"
    ]
    
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Saved {len(results)} listings to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")


def delete_csv_file(file_path):
    """Delete a CSV file if it exists."""
    file_path = Path(file_path)
    if file_path.exists():
        file_path.unlink()
        logger.info(f"Deleted file: {file_path}")


def run(headless=True, max_listings_per_city=None):
    """
    Main scraper function.
    
    Args:
        headless: Run browser in headless mode (faster)
        max_listings_per_city: Limit listings per city (None = all)
    """
    start_time = datetime.now()
    logger.info("=" * 50)
    logger.info("Starting MagicBricks Scraper (Improved Version)")
    logger.info(f"Headless mode: {headless}")
    logger.info(f"Max listings per city: {max_listings_per_city or 'All'}")
    logger.info("=" * 50)
    
    all_results = []
    city_stats = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720}
        )

        page = context.new_page()

        for idx, city in enumerate(CITIES, 1):
            logger.info(f"\n[{idx}/{len(CITIES)}] Processing {city}")
            
            results = scrape_city(page, city, max_listings=max_listings_per_city)
            all_results.extend(results)
            city_stats[city] = len(results)
            
            time.sleep(3)  # Rate limiting between cities

        browser.close()

    # Save results
    output_path = "data/outputs/magicbricks_india_properties.csv"
    save_to_csv(all_results, output_path)

    # Clean data
    logger.info("Cleaning location data...")
    clean_location_csv(
        input_csv=output_path,
        output_csv="data/outputs/magicbricks_india_properties_cleaned.csv",
        drop_empty_location=True
    )
    delete_csv_file(output_path)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 50)
    logger.info("SCRAPING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total listings scraped: {len(all_results)}")
    logger.info(f"Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Average per city: {len(all_results)/len(CITIES):.1f}")
    logger.info("\nListings per city:")
    for city, count in sorted(city_stats.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {city}: {count}")
    logger.info("=" * 50)


if __name__ == "__main__":
    # Run in headless mode for faster scraping
    # Set max_listings_per_city=100 to limit per city for testing
    run(headless=True, max_listings_per_city=None)
