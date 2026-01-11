# src/rag/property_explanations.py

import pandas as pd

CSV_PATH = "data/outputs/analyzed_properties.csv"


def build_property_explanation(row: dict) -> str:
    """
    Build a searchable property explanation text for vector embeddings.
    Includes keywords that make semantic search more effective.
    """
    price = row.get('price', 0)
    price_cr = price / 10000000 if price else 0
    price_per_sqft = row.get('price_per_sqft', 0)
    area = row.get('area_sqft', 0)
    wealth_buying = row.get('wealth_buying', 0)
    wealth_renting = row.get('wealth_renting', 0)
    decision = row.get('decision', 'Unknown')
    city = str(row.get('city', 'Unknown')).lower()
    location = str(row.get('location', 'Unknown'))
    bhk = row.get('bhk', 'Unknown')
    
    # Determine buy vs rent recommendation
    is_buy = 'buy' in str(decision).lower()
    recommendation = "BUY" if is_buy else "RENT"
    
    # Create searchable text with multiple keyword variations
    return f"""
Property in {location}, {city.title()}
City: {city} | Location: {location}
Type: {bhk} BHK apartment flat residential property
Area: {area} sqft square feet
Price: ₹{price:,.0f} (₹{price_cr:.2f} Crore)
Price per sqft: ₹{price_per_sqft:,.0f} per square foot
Cost per sqft in {city}: ₹{price_per_sqft:,.0f}
Average price per sqft {location}: ₹{price_per_sqft:,.0f}

Financial Analysis for {location}:
20-year wealth if buying: ₹{wealth_buying:,.0f}
20-year wealth if renting: ₹{wealth_renting:,.0f}
Wealth difference: ₹{abs(wealth_renting - wealth_buying):,.0f}

Investment Recommendation: {recommendation}
Decision: {decision}
Should you buy or rent in {location}? {recommendation} is recommended.
Buy vs rent analysis for {city}: {recommendation} is better for this property.

Keywords: {city} real estate, {location} property, {bhk}bhk in {city}, properties in {city}, {city} property prices, affordable housing {city}, investment property {city}
""".strip()


def build_city_summary(df: pd.DataFrame, city: str) -> str:
    """Build a summary document for a city's aggregate statistics."""
    city_df = df[df['city'].str.lower() == city.lower()]
    if city_df.empty:
        return ""
    
    avg_price = city_df['price'].mean()
    avg_price_per_sqft = city_df['price_per_sqft'].mean()
    total_properties = len(city_df)
    min_price = city_df['price'].min()
    max_price = city_df['price'].max()
    locations = city_df['location'].unique().tolist()[:15]  # Top 15 locations
    
    buy_count = city_df['decision'].str.contains('buy', case=False, na=False).sum()
    rent_count = total_properties - buy_count
    
    return f"""
City Overview: {city.title()}
Total properties in {city}: {total_properties}
Average property price in {city}: ₹{avg_price:,.0f} (₹{avg_price/10000000:.2f} Crore)
Average price per sqft in {city}: ₹{avg_price_per_sqft:,.0f}
Price range in {city}: ₹{min_price:,.0f} to ₹{max_price:,.0f}
Buy recommendations in {city}: {buy_count}
Rent recommendations in {city}: {rent_count}

Locations in {city}: {', '.join(locations)}
Areas in {city} include: {', '.join(locations)}

Keywords: {city} average price, {city} property rates, {city} real estate market, cost of property in {city}, {city} locations, areas in {city}, {city} buy vs rent
""".strip()


def load_property_explanations():
    """
    Load all property explanations plus city summaries for vector store.
    Returns list of text documents optimized for semantic search.
    """
    df = pd.read_csv(CSV_PATH)
    explanations = []

    # Add individual property explanations
    for _, row in df.iterrows():
        explanations.append(build_property_explanation(row.to_dict()))

    # Add city-level summary documents
    cities = df['city'].unique()
    for city in cities:
        summary = build_city_summary(df, city)
        if summary:
            explanations.append(summary)
    
    print(f"📄 Built {len(explanations)} documents ({len(df)} properties + {len(cities)} city summaries)")
    return explanations
