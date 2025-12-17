import re

def parse_price(text):
    if not text or text.strip() == "":
        return None

    # Take only first line (remove square foot price line)
    text = text.split()[0]  

    text = text.replace("₹", "").replace(",", "").replace("Onwards", "").strip()

    # Remove everything except digits and dots
    num = re.sub(r"[^0-9.]", "", text)

    # If nothing numeric remains → invalid price
    if num == "":
        return None

    # Handle Cr
    if "Cr" in text:
        return float(num) * 1e7

    # Handle Lac or Lakh
    if "Lac" in text or "Lakh" in text:
        return float(num) * 1e5

    return float(num)


import re

def parse_area(text):
    if not text:
        return None

    # Find FIRST number in the string
    match = re.search(r"[0-9]+\.?[0-9]*", text)
    if not match:
        return None

    return float(match.group(0))
