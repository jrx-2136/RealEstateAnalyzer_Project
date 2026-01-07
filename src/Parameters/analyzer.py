# src/analyzer.py

import pandas as pd
from src.Parameters.buy_vs_rent import buying_case, renting_case, compare_results


def estimate_rent(area_sqft):
    return area_sqft * 20  # simple heuristic


def run_analysis():

    df = pd.read_csv("data/outputs/magicbricks_india_final.csv")
    results = []

    for _, row in df.iterrows():
        try:
            price = float(row["price_total_inr"])
            area = float(row["area_sqft"])
        except:
            continue

        down_payment = 0.20 * price

        buy = buying_case(
            property_price=price,
            down_payment=down_payment,
            loan_rate=8.5,
            tax_rate=20,
            appreciation_rate=5
        )

        rent = renting_case(
            initial_rent=estimate_rent(area),
            escalation=5,
            down_payment=down_payment,
            invest_rate=10,
            monthly_saving=15000
        )

        decision = compare_results(buy, rent)

        results.append({
            "location": row["location"],
            "city": row["city"],
            "price": price,
            "area_sqft": area,
            "bhk": row["BHK"],
            "price_per_sqft": row["price_per_sqft"],
            "wealth_buying": buy["wealth_buying"],
            "wealth_renting": rent["wealth_renting"],
            "decision": decision
        })

    out = pd.DataFrame(results)
    out.to_csv("data/outputs/analyzed_properties.csv", index=False)

    print("Saved → data/outputs/analyzed_properties.csv")
