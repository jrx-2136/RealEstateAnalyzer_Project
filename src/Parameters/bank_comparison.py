import json
from pathlib import Path
import pandas as pd


def load_bank_rates(path="src/scrapers/bank_rates.json"):
    with open(Path(path), "r") as f:
        return json.load(f)


def calculate_emi(principal, annual_rate, tenure_years):
    r = annual_rate / 100 / 12
    n = tenure_years * 12
    return (principal * r * (1 + r) ** n) / ((1 + r) ** n - 1)


def run_bank_comparison(property_price, down_payment, tenure_years=20):
    loan_amount = property_price - down_payment
    bank_rates = load_bank_rates()

    results = []

    for bank, rate in bank_rates.items():
        emi = calculate_emi(loan_amount, rate, tenure_years)
        total_paid = emi * tenure_years * 12
        interest_paid = total_paid - loan_amount

        results.append({
            "bank": bank,
            "interest_rate (%)": rate,
            "loan_amount": round(loan_amount, 2),
            "monthly_emi": round(emi, 2),
            "total_interest": round(interest_paid, 2),
            "total_payment": round(total_paid, 2)
        })

    df = pd.DataFrame(results)
    df.to_csv("data/outputs/bank_loan_comparison.csv", index=False)
    print("Saved → data/outputs/bank_loan_comparison.csv")

    return df
