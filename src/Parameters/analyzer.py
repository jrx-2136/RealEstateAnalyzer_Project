import pandas as pd
from .parsers import parse_price, parse_area
from .buy_vs_rent import buying_case, renting_case, compare_results


def estimate_rent(area):
    return area * 20   # simple model


def run_analysis():

    df = pd.read_csv("data/outputs/magicbricks_india_properties_cleaned.csv")
    results = []

    for _, row in df.iterrows():
        price = float(row["price_total_inr"]) if row["price_total_inr"] else None
        area = float(row["area_sqft"]) if row["area_sqft"] else None


        if not price or not area:
            print("Skipping:", row["title"], "| price:", price, "| area:", area)
            continue

        down_payment = 0.20 * price

        buy = buying_case(
            price,
            down_payment,
            loan_rate=8.5,
            tax_rate=20,
            appreciation_rate=5
        )

        rent = renting_case(
            estimate_rent(area),
            escalation=5,
            down_payment=down_payment,
            invest_rate=10,
            monthly_saving=15000
        )

        decision = compare_results(buy, rent)

        results.append({
            "title": row["title"],
            "location": row["location"],
            "price": price,
            "area": area,
            "wealth_buying": buy["wealth_buying"],
            "wealth_renting": rent["wealth_renting"],
            "decision": decision
        })

    out = pd.DataFrame(results)
    out.to_csv("data/outputs/analyzed_properties.csv", index=False)

    print("Saved → data/outputs/analyzed_properties.csv")
