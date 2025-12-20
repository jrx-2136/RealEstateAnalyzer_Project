import json
from pathlib import Path
import pandas as pd


# -----------------------------
# Load bank rates
# -----------------------------
def load_bank_rates(path="src/scrapers/bank_rates.json"):
    with open(Path(path), "r") as f:
        return json.load(f)


# -----------------------------
# EMI calculation
# -----------------------------
def calculate_emi(principal, annual_rate, tenure_years):
    r = annual_rate / 100 / 12
    n = tenure_years * 12
    emi = (principal * r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return emi


# -----------------------------
# EMI split (first month)
# -----------------------------
def emi_breakup(principal, annual_rate, emi):
    monthly_rate = annual_rate / 100 / 12
    interest_component = principal * monthly_rate
    principal_component = emi - interest_component
    return interest_component, principal_component


# -----------------------------
# Main function
# -----------------------------
def run_home_loan_analysis(property_price):

    bank_rates = load_bank_rates()

    avg_rate = sum(bank_rates.values()) / len(bank_rates)
    tenure_years = 20

    emi = calculate_emi(property_price, avg_rate, tenure_years)
    interest, principal = emi_breakup(property_price, avg_rate, emi)

    result = {
        "property_price": property_price,
        "banks_considered": list(bank_rates.keys()),
        "average_interest_rate (%)": round(avg_rate, 2),
        "tenure_years": tenure_years,
        "monthly_emi": round(emi, 2),
        "interest_component": round(interest, 2),
        "principal_component": round(principal, 2)
    }

    df = pd.DataFrame([result])
    df.to_csv("data/outputs/home_loan_emi_analysis.csv", index=False)

    print("Saved → data/outputs/home_loan_emi_analysis.csv")
    return result
