# src/rag/sql_retriever.py

"""
SQL-based retrieval for structured queries on property data.
Provides aggregate statistics and filtered property listings.

PRECISION FIX: Added exact/fuzzy property name matching functions
to handle specific property queries with high accuracy.
"""

import pandas as pd
import os
from difflib import SequenceMatcher

CSV_PATH = "data/outputs/analyzed_properties.csv"

# Load DataFrame once at module level
_df = None

def _get_df():
    """Lazy load the DataFrame"""
    global _df
    if _df is None:
        if os.path.exists(CSV_PATH):
            _df = pd.read_csv(CSV_PATH)
            print(f"📊 SQL Retriever: Loaded {len(_df)} properties")
        else:
            _df = pd.DataFrame()
            print(f"⚠️ SQL Retriever: CSV not found at {CSV_PATH}")
    return _df


def get_all_property_names() -> list:
    """
    Get list of all unique property/location names in the dataset.
    PRECISION FIX: Used for fuzzy matching against specific property queries.
    """
    df = _get_df()
    if df.empty:
        return []
    return df['location'].unique().tolist()


def find_property_by_name(name: str, threshold: float = 0.6) -> tuple:
    """
    Find a property by exact or fuzzy name match.
    
    PRECISION FIX: This is the core function for high-precision property lookup.
    Returns only the best match, not multiple unrelated properties.
    
    Args:
        name: Property name to search for
        threshold: Minimum similarity score (0-1) for fuzzy match
        
    Returns:
        tuple: (matched_property: dict or None, match_type: str, similar_properties: list)
        - match_type: 'exact', 'fuzzy', or 'none'
        - similar_properties: List of close matches if no exact match (for clarification)
    """
    df = _get_df()
    if df.empty:
        return (None, 'none', [])
    
    name_lower = name.lower().strip()
    
    # Step 1: Try exact match (case-insensitive)
    exact_match = df[df['location'].str.lower() == name_lower]
    if not exact_match.empty:
        return (exact_match.iloc[0].to_dict(), 'exact', [])
    
    # Step 2: Try substring match (property name contains search term or vice versa)
    substring_matches = df[
        df['location'].str.lower().str.contains(name_lower, regex=False, na=False) |
        df['location'].apply(lambda x: name_lower in x.lower() if pd.notna(x) else False)
    ]
    if len(substring_matches) == 1:
        return (substring_matches.iloc[0].to_dict(), 'exact', [])
    elif len(substring_matches) > 1:
        # Multiple substring matches - return best one and list others for clarification
        # Sort by how closely the name matches
        matches_with_scores = []
        for _, row in substring_matches.iterrows():
            score = SequenceMatcher(None, name_lower, row['location'].lower()).ratio()
            matches_with_scores.append((row.to_dict(), score))
        matches_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        best = matches_with_scores[0][0]
        similar = [m[0]['location'] for m in matches_with_scores[1:4]]  # Top 3 alternatives
        return (best, 'fuzzy', similar)
    
    # Step 3: Fuzzy match using similarity scoring
    best_score = 0
    best_match = None
    similar_properties = []
    
    for _, row in df.iterrows():
        prop_name = row['location']
        if pd.isna(prop_name):
            continue
            
        prop_lower = prop_name.lower()
        
        # Calculate similarity score
        score = SequenceMatcher(None, name_lower, prop_lower).ratio()
        
        # Also check word-level matching
        name_words = set(name_lower.split())
        prop_words = set(prop_lower.split())
        common_words = name_words & prop_words
        if common_words and len(common_words) >= min(len(name_words), len(prop_words)) * 0.5:
            score = max(score, 0.6 + len(common_words) * 0.1)
        
        if score >= threshold:
            if score > best_score:
                if best_match:
                    similar_properties.append(best_match['location'])
                best_score = score
                best_match = row.to_dict()
            elif score >= threshold - 0.1:
                similar_properties.append(prop_name)
    
    if best_match and best_score >= threshold:
        return (best_match, 'fuzzy', similar_properties[:3])
    
    return (None, 'none', similar_properties[:5])


def format_single_property(prop: dict) -> str:
    """
    Format a single property for display with investment intelligence.
    
    PRECISION FIX: Used for specific property queries - shows only ONE property
    in detail with investment score and risk indicators.
    
    Args:
        prop: Property dictionary
        
    Returns:
        str: Formatted property details
    """
    if not prop:
        return "Property not found."
    
    price = prop.get('price', 0)
    price_cr = price / 10000000 if price else 0
    city = prop.get('city', 'Unknown')
    
    # Get investment score
    try:
        from src.rag.investment_intelligence import calculate_investment_score, get_risk_summary
        from src.rag.sql_retriever import get_city_stats
        
        city_stats = get_city_stats(city)
        score_result = calculate_investment_score(prop, city_stats)
        risk = get_risk_summary(city)
        
        has_intelligence = True
    except:
        has_intelligence = False
        score_result = None
        risk = None
    
    lines = []
    lines.append(f"## {prop.get('location', 'Unknown Property')}")
    lines.append("")
    lines.append(f"**City:** {city.title()}")
    lines.append(f"**Price:** ₹{price:,.0f} (₹{price_cr:.2f} Crore)")
    lines.append(f"**Area:** {prop.get('area_sqft', 'N/A')} sqft")
    lines.append(f"**Configuration:** {prop.get('bhk', 'N/A')} BHK")
    lines.append(f"**Price per sqft:** ₹{prop.get('price_per_sqft', 0):,.0f}")
    lines.append("")
    
    # Investment Score (if available)
    if has_intelligence and score_result:
        lines.append(f"### Investment Score: {score_result['total_score']}/100 (Grade: {score_result['grade']})")
        lines.append(f"*{score_result['grade_explanation']}*")
        lines.append("")
    
    lines.append("### Investment Analysis:")
    lines.append(f"• Wealth if Buying (20Y): ₹{prop.get('wealth_buying', 0):,.0f}")
    lines.append(f"• Wealth if Renting (20Y): ₹{prop.get('wealth_renting', 0):,.0f}")
    lines.append(f"• **Recommendation:** {prop.get('decision', 'N/A')}")
    lines.append("")
    
    # Risk Assessment (if available)
    if has_intelligence and risk:
        lines.append(f"### Risk Profile: {risk['overall_risk_level'].upper()}")
        lines.append(f"• Price Volatility: {risk['price_volatility']['risk_level']}")
        lines.append(f"• Market Liquidity: {risk['liquidity']['risk_level']}")
        lines.append("")
    
    # How it was calculated
    lines.append("### How This Was Calculated:")
    lines.append("• Wealth projections assume 20-year holding period")
    lines.append("• Down payment: 20% | Loan rate: 8.5% | Appreciation: 5%/year")
    lines.append("• Renting scenario invests savings at 10% annual return")
    lines.append("• See INVESTMENT_METRICS.md for full methodology")
    lines.append("")
    lines.append("📊 *Data sourced from property database*")
    
    return "\n".join(lines)


def get_available_cities():
    """Get list of unique cities in the dataset."""
    df = _get_df()
    if df.empty:
        return []
    return df['city'].unique().tolist()


def filter_properties(city=None, bhk=None, min_price=None, max_price=None, decision=None, limit=10):
    """
    Filter properties based on criteria.
    Returns list of property dictionaries.
    """
    df = _get_df()
    if df.empty:
        return []
    
    result = df.copy()

    if city:
        result = result[result["city"].str.lower() == city.lower()]

    if bhk:
        result = result[result["bhk"] == bhk]
    
    if min_price:
        result = result[result["price"] >= min_price]
    
    if max_price:
        result = result[result["price"] <= max_price]

    if decision:
        # Handle decision column - may contain full text
        result = result[result["decision"].str.contains(decision, case=False, na=False)]

    return result.head(limit).to_dict(orient="records")


def get_city_stats(city: str = None) -> dict:
    """
    Get aggregate statistics for a city or all cities.
    
    Returns dict with:
        - total_properties
        - avg_price
        - avg_price_per_sqft
        - price_range (min, max)
        - buy_count, rent_count
        - locations list
    """
    df = _get_df()
    if df.empty:
        return {}
    
    if city:
        city_df = df[df['city'].str.lower() == city.lower()]
        if city_df.empty:
            return {"error": f"No data found for city: {city}"}
    else:
        city_df = df
    
    buy_count = city_df['decision'].str.contains('buy', case=False, na=False).sum()
    
    return {
        "city": city if city else "all cities",
        "total_properties": len(city_df),
        "avg_price": round(city_df['price'].mean(), 2),
        "avg_price_crore": round(city_df['price'].mean() / 10000000, 2),
        "avg_price_per_sqft": round(city_df['price_per_sqft'].mean(), 2),
        "min_price": city_df['price'].min(),
        "max_price": city_df['price'].max(),
        "avg_area_sqft": round(city_df['area_sqft'].mean(), 2),
        "buy_recommendations": int(buy_count),
        "rent_recommendations": int(len(city_df) - buy_count),
        "locations": city_df['location'].unique().tolist()[:20]
    }


def get_locations_in_city(city: str) -> list:
    """Get all unique locations/areas in a city."""
    df = _get_df()
    if df.empty:
        return []
    
    city_df = df[df['city'].str.lower() == city.lower()]
    return city_df['location'].unique().tolist()


def get_top_properties(city: str = None, sort_by: str = "price", ascending: bool = True, limit: int = 5) -> list:
    """
    Get top/bottom properties by price or price_per_sqft.
    
    Args:
        city: Filter by city (optional)
        sort_by: Column to sort by (price, price_per_sqft)
        ascending: True for cheapest first, False for most expensive
        limit: Number of results
    """
    df = _get_df()
    if df.empty:
        return []
    
    result = df.copy()
    if city:
        result = result[result['city'].str.lower() == city.lower()]
    
    result = result.sort_values(by=sort_by, ascending=ascending).head(limit)
    return result.to_dict(orient="records")


def get_comparison_stats(cities: list) -> dict:
    """
    Get comparison statistics for multiple cities.
    
    Args:
        cities: List of city names to compare
        
    Returns:
        Dict with stats for each city
    """
    df = _get_df()
    if df.empty:
        return {}
    
    comparison = {}
    for city in cities:
        stats = get_city_stats(city)
        if "error" not in stats:
            comparison[city] = stats
    
    return comparison


def format_properties_for_context(properties: list) -> str:
    """Format property list as readable context for LLM."""
    if not properties:
        return "No properties found matching the criteria."
    
    lines = []
    for i, prop in enumerate(properties, 1):
        price_cr = prop.get('price', 0) / 10000000
        lines.append(
            f"{i}. {prop.get('location', 'Unknown')}, {prop.get('city', 'Unknown').title()}\n"
            f"   Price: ₹{prop.get('price', 0):,.0f} (₹{price_cr:.2f} Cr)\n"
            f"   Area: {prop.get('area_sqft', 0)} sqft | {prop.get('bhk', 'N/A')} BHK\n"
            f"   Price/sqft: ₹{prop.get('price_per_sqft', 0):,.0f}\n"
            f"   Recommendation: {prop.get('decision', 'N/A')}"
        )
    return "\n\n".join(lines)


def format_city_stats_for_context(stats: dict) -> str:
    """Format city statistics as readable context for LLM."""
    if not stats or "error" in stats:
        return stats.get("error", "No statistics available.")
    
    locations_sample = stats.get('locations', [])[:10]
    
    return f"""
City: {stats.get('city', 'Unknown').title()}
Total Properties: {stats.get('total_properties', 0)}
Average Price: ₹{stats.get('avg_price', 0):,.0f} (₹{stats.get('avg_price_crore', 0):.2f} Crore)
Average Price per Sqft: ₹{stats.get('avg_price_per_sqft', 0):,.0f}
Price Range: ₹{stats.get('min_price', 0):,.0f} to ₹{stats.get('max_price', 0):,.0f}
Average Area: {stats.get('avg_area_sqft', 0):,.0f} sqft
Buy Recommendations: {stats.get('buy_recommendations', 0)}
Rent Recommendations: {stats.get('rent_recommendations', 0)}
Sample Locations: {', '.join(locations_sample)}
""".strip()
