# src/rag/property_explanations.py

import pandas as pd

CSV_PATH = "data/outputs/analyzed_properties.csv"


def build_property_explanation(row: dict) -> str:
    return f"""
Property Overview
Location: {row.get('location', 'Unknown')}
City: {row.get('city', 'Unknown')}
Area: {row.get('area_sqft')} sqft
Price: ₹{row.get('price_total_inr')}

Financial Summary
Wealth if Buying: ₹{row.get('wealth_buying')}
Wealth if Renting: ₹{row.get('wealth_renting')}

Final Decision
{row.get('final_decision')}

Rationale
This decision is based on backend financial simulations comparing long-term wealth outcomes
between buying and renting under fixed assumptions.
""".strip()


def load_property_explanations():
    df = pd.read_csv(CSV_PATH)
    explanations = []

    for _, row in df.iterrows():
        explanations.append(build_property_explanation(row.to_dict()))

    return explanations
