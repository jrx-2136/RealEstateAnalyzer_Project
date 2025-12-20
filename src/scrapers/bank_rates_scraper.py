import requests
import json
import re
from pathlib import Path


URL = "https://www.bankbazaar.com/home-loan-interest-rate.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# here specific banks are chosen to compare. 
BANKS = ["State Bank of India","SBI","HDFC Bank","ICICI Bank","Axis Bank","Bank of Baroda","Punjab National Bank","Canara Bank",
        "Union Bank of India","Bank of India","Central Bank of India","IDBI Bank","Kotak Mahindra Bank","Federal Bank",
        "YES Bank","LIC Housing Finance"]


def scrape_bank_rates():
    response = requests.get(URL, headers=HEADERS, timeout=15)
    response.raise_for_status()

    text = response.text

    bank_rates = {}

    for bank in BANKS:
        # Look for patterns like: "HDFC Bank ... 7.9%"
        pattern = rf"{bank}.*?(\d+(\.\d+)?)\s*%"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            rate = float(match.group(1))
            bank_rates[bank] = rate

    return bank_rates

def save_bank_rates(bank_rates, path="src/scrapers/bank_rates.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(bank_rates, f, indent=2)

    print(f"[✓] Saved {len(bank_rates)} banks → {path}")


if __name__ == "__main__":
    rates = scrape_bank_rates()
    save_bank_rates(rates)
