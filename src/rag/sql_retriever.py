# src/rag/sql_retriever.py

import pandas as pd

CSV_PATH = "data/outputs/analyzed_properties.csv"
df = pd.read_csv(CSV_PATH)


def filter_properties(city=None, decision=None, limit=5):
    result = df.copy()

    if city:
        result = result[result["city"].str.lower() == city.lower()]

    if decision:
        result = result[result["final_decision"] == decision]

    return result.head(limit).to_dict(orient="records")
