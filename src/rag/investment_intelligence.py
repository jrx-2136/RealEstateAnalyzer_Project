# src/rag/investment_intelligence.py

"""
Investment Intelligence Module

Provides advanced investment analytics, risk indicators, and decision support.
All calculations are rule-based and grounded in the actual property dataset.

NEW FEATURES:
- Risk Indicators (Price Volatility, Liquidity, Rental Stability)
- Composite Investment Score (0-100)
- City-Level Investment Profiles
- Scenario Sensitivity (Conservative/Moderate/Aggressive)
- Market Trend Signals (Overheated/Stable/Cooling)
- Investment Context Explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os

CSV_PATH = "data/outputs/analyzed_properties.csv"

# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================
SCENARIOS = {
    "conservative": {
        "name": "Conservative",
        "description": "Cautious assumptions - lower growth, higher costs",
        "appreciation_rate": 3.0,      # Below historical average
        "rent_growth_rate": 3.0,       # Below inflation
        "investment_return": 8.0,      # Fixed income focus
        "vacancy_rate": 10.0,          # Higher vacancy buffer
        "maintenance_percent": 1.5,    # Higher maintenance costs
        "loan_rate": 9.5,              # Higher interest rate
    },
    "moderate": {
        "name": "Moderate",
        "description": "Balanced assumptions - historical averages",
        "appreciation_rate": 5.0,      # Historical average
        "rent_growth_rate": 5.0,       # In line with inflation
        "investment_return": 10.0,     # Equity index returns
        "vacancy_rate": 5.0,           # Standard vacancy
        "maintenance_percent": 1.0,    # Normal maintenance
        "loan_rate": 8.5,              # Current average
    },
    "aggressive": {
        "name": "Aggressive",
        "description": "Optimistic assumptions - higher growth potential",
        "appreciation_rate": 8.0,      # Growth market assumption
        "rent_growth_rate": 7.0,       # Strong rental demand
        "investment_return": 12.0,     # Active equity management
        "vacancy_rate": 2.0,           # Low vacancy expectation
        "maintenance_percent": 0.5,    # Minimal maintenance
        "loan_rate": 7.5,              # Best rate assumption
    }
}

# Explicit assumptions documentation
ASSUMPTIONS = {
    "holding_period_years": 20,
    "down_payment_percent": 20,
    "loan_tenure_years": 20,
    "rent_to_price_ratio": 0.003,  # ~0.3% monthly or ~3.6% annual yield
    "stamp_duty_percent": 6,       # Average across states
    "registration_percent": 1,
    "brokerage_percent": 1,
}


# ============================================================================
# DATA LOADING
# ============================================================================
_df = None

def _get_df():
    """Lazy load the DataFrame"""
    global _df
    if _df is None:
        if os.path.exists(CSV_PATH):
            _df = pd.read_csv(CSV_PATH)
        else:
            _df = pd.DataFrame()
    return _df


# ============================================================================
# RISK INDICATORS (Rule-Based)
# ============================================================================

def calculate_price_volatility_risk(city: str = None, location: str = None) -> Dict[str, Any]:
    """
    Calculate price volatility risk based on price variance in the area.
    
    HEURISTIC:
    - High: Coefficient of Variation (CV) > 50%
    - Medium: CV 25-50%
    - Low: CV < 25%
    
    Returns:
        Dict with risk_level, cv_percent, explanation
    """
    df = _get_df()
    if df.empty:
        return {"risk_level": "unknown", "cv_percent": 0, "explanation": "No data available"}
    
    # Filter by city/location if provided
    subset = df.copy()
    if city:
        subset = subset[subset['city'].str.lower() == city.lower()]
    if location:
        subset = subset[subset['location'].str.lower().str.contains(location.lower(), na=False)]
    
    if len(subset) < 3:
        return {
            "risk_level": "unknown",
            "cv_percent": 0,
            "explanation": f"Insufficient data (only {len(subset)} properties) to assess volatility"
        }
    
    # Calculate Coefficient of Variation for price per sqft
    price_per_sqft = subset['price_per_sqft']
    mean_price = price_per_sqft.mean()
    std_price = price_per_sqft.std()
    cv = (std_price / mean_price) * 100 if mean_price > 0 else 0
    
    # Determine risk level
    if cv > 50:
        risk_level = "high"
        explanation = f"High price variance (CV: {cv:.1f}%) indicates inconsistent pricing. Prices range significantly, suggesting market inefficiency or diverse property quality."
    elif cv > 25:
        risk_level = "medium"
        explanation = f"Moderate price variance (CV: {cv:.1f}%) suggests some pricing diversity. Compare similar properties carefully."
    else:
        risk_level = "low"
        explanation = f"Low price variance (CV: {cv:.1f}%) indicates consistent market pricing. Fair value is easier to determine."
    
    return {
        "risk_level": risk_level,
        "cv_percent": round(cv, 1),
        "mean_price_per_sqft": round(mean_price, 0),
        "std_price_per_sqft": round(std_price, 0),
        "sample_size": len(subset),
        "explanation": explanation
    }


def calculate_liquidity_risk(city: str = None) -> Dict[str, Any]:
    """
    Calculate liquidity risk based on listing density.
    
    HEURISTIC:
    - High: < 20 listings (thin market, hard to buy/sell)
    - Medium: 20-50 listings
    - Low: > 50 listings (active market)
    
    Returns:
        Dict with risk_level, listing_count, explanation
    """
    df = _get_df()
    if df.empty:
        return {"risk_level": "unknown", "listing_count": 0, "explanation": "No data available"}
    
    subset = df.copy()
    if city:
        subset = subset[subset['city'].str.lower() == city.lower()]
    
    count = len(subset)
    
    if count < 20:
        risk_level = "high"
        explanation = f"Only {count} listings in this market. Low liquidity may make it harder to sell and could affect pricing power."
    elif count < 50:
        risk_level = "medium"
        explanation = f"{count} listings indicate moderate market activity. Reasonable liquidity but monitor for changes."
    else:
        risk_level = "low"
        explanation = f"{count} listings indicate an active market. Good liquidity supports fair pricing and easier transactions."
    
    return {
        "risk_level": risk_level,
        "listing_count": count,
        "explanation": explanation
    }


def calculate_rental_stability_risk(city: str = None) -> Dict[str, Any]:
    """
    Calculate rental stability risk based on rent dispersion.
    
    HEURISTIC:
    - Uses estimated_rent column if available, or derives from rent_to_price_ratio
    - High: Rent CV > 40%
    - Medium: Rent CV 20-40%
    - Low: Rent CV < 20%
    
    Returns:
        Dict with risk_level, explanation
    """
    df = _get_df()
    if df.empty:
        return {"risk_level": "unknown", "explanation": "No data available"}
    
    subset = df.copy()
    if city:
        subset = subset[subset['city'].str.lower() == city.lower()]
    
    if len(subset) < 5:
        return {
            "risk_level": "unknown",
            "explanation": f"Insufficient data ({len(subset)} properties) for rental stability analysis"
        }
    
    # Check for estimated_rent column, otherwise derive
    if 'estimated_rent' in subset.columns:
        rents = subset['estimated_rent'].dropna()
    else:
        # Derive from price using typical rent-to-price ratio
        rents = subset['price'] * ASSUMPTIONS['rent_to_price_ratio']
    
    if len(rents) < 5:
        return {"risk_level": "unknown", "explanation": "Insufficient rental data"}
    
    mean_rent = rents.mean()
    std_rent = rents.std()
    cv = (std_rent / mean_rent) * 100 if mean_rent > 0 else 0
    
    if cv > 40:
        risk_level = "high"
        explanation = f"High rental variance (CV: {cv:.1f}%) indicates unpredictable rental income. Budget conservatively for vacancies and rent fluctuations."
    elif cv > 20:
        risk_level = "medium"
        explanation = f"Moderate rental variance (CV: {cv:.1f}%) suggests some rental income uncertainty. Standard vacancy assumptions apply."
    else:
        risk_level = "low"
        explanation = f"Low rental variance (CV: {cv:.1f}%) indicates stable rental market. Income projections are more reliable."
    
    return {
        "risk_level": risk_level,
        "cv_percent": round(cv, 1),
        "avg_estimated_rent": round(mean_rent, 0),
        "sample_size": len(rents),
        "explanation": explanation
    }


def get_risk_summary(city: str = None, location: str = None) -> Dict[str, Any]:
    """
    Get comprehensive risk summary combining all risk indicators.
    
    Returns:
        Dict with all risk indicators and overall risk assessment
    """
    price_vol = calculate_price_volatility_risk(city, location)
    liquidity = calculate_liquidity_risk(city)
    rental_stability = calculate_rental_stability_risk(city)
    
    # Calculate overall risk score (0-100, higher = more risky)
    risk_scores = {
        "high": 30,
        "medium": 15,
        "low": 5,
        "unknown": 20  # Assume moderate risk when unknown
    }
    
    total_risk = (
        risk_scores.get(price_vol['risk_level'], 20) +
        risk_scores.get(liquidity['risk_level'], 20) +
        risk_scores.get(rental_stability['risk_level'], 20)
    )
    
    # Normalize to 0-100
    overall_risk = min(100, total_risk)
    
    if overall_risk >= 60:
        overall_level = "high"
        overall_explanation = "Multiple risk factors present. Proceed with caution and thorough due diligence."
    elif overall_risk >= 30:
        overall_level = "medium"
        overall_explanation = "Moderate risk profile. Standard investment precautions recommended."
    else:
        overall_level = "low"
        overall_explanation = "Favorable risk profile. Market conditions support investment."
    
    return {
        "price_volatility": price_vol,
        "liquidity": liquidity,
        "rental_stability": rental_stability,
        "overall_risk_score": overall_risk,
        "overall_risk_level": overall_level,
        "overall_explanation": overall_explanation
    }


# ============================================================================
# INVESTMENT SCORE (Composite 0-100)
# ============================================================================

def calculate_investment_score(property_data: Dict, city_stats: Dict = None) -> Dict[str, Any]:
    """
    Calculate composite investment score (0-100) for a property.
    
    METHODOLOGY:
    - ROI Component (30 pts): Based on ROI relative to target
    - Rental Yield Component (20 pts): Annual rental yield
    - Value Component (25 pts): Price/sqft vs city average
    - Buy Recommendation (15 pts): From buy vs rent analysis
    - Risk Penalty (-10 pts max): Based on risk indicators
    
    Args:
        property_data: Dict with property metrics
        city_stats: Optional city-level stats for comparison
    
    Returns:
        Dict with total_score, component_scores, explanation
    """
    scores = {}
    explanations = []
    
    # 1. ROI Component (0-30 points)
    roi = property_data.get('roi_percent', 0)
    if roi >= 150:
        scores['roi'] = 30
        explanations.append(f"Excellent ROI ({roi:.1f}%) significantly above target")
    elif roi >= 100:
        scores['roi'] = 25
        explanations.append(f"Strong ROI ({roi:.1f}%) above average")
    elif roi >= 50:
        scores['roi'] = 20
        explanations.append(f"Moderate ROI ({roi:.1f}%)")
    elif roi >= 0:
        scores['roi'] = 10
        explanations.append(f"Low ROI ({roi:.1f}%), limited growth potential")
    else:
        scores['roi'] = 0
        explanations.append(f"Negative ROI ({roi:.1f}%), value decline expected")
    
    # 2. Rental Yield Component (0-20 points)
    rental_yield = property_data.get('rental_yield', 0)
    if rental_yield >= 5:
        scores['rental_yield'] = 20
        explanations.append(f"Excellent rental yield ({rental_yield:.2f}%)")
    elif rental_yield >= 4:
        scores['rental_yield'] = 16
        explanations.append(f"Good rental yield ({rental_yield:.2f}%)")
    elif rental_yield >= 3:
        scores['rental_yield'] = 12
        explanations.append(f"Average rental yield ({rental_yield:.2f}%)")
    elif rental_yield >= 2:
        scores['rental_yield'] = 8
        explanations.append(f"Below average rental yield ({rental_yield:.2f}%)")
    else:
        scores['rental_yield'] = 4
        explanations.append(f"Low rental yield ({rental_yield:.2f}%)")
    
    # 3. Value Component (0-25 points) - Price/sqft vs city average
    price_per_sqft = property_data.get('price_per_sqft', 0)
    if city_stats and city_stats.get('avg_price_per_sqft'):
        city_avg = city_stats['avg_price_per_sqft']
        value_ratio = price_per_sqft / city_avg if city_avg > 0 else 1
        
        if value_ratio <= 0.7:
            scores['value'] = 25
            explanations.append(f"Significantly undervalued ({(1-value_ratio)*100:.0f}% below city avg)")
        elif value_ratio <= 0.85:
            scores['value'] = 20
            explanations.append(f"Moderately undervalued ({(1-value_ratio)*100:.0f}% below city avg)")
        elif value_ratio <= 1.0:
            scores['value'] = 15
            explanations.append(f"Fair value (at or slightly below city avg)")
        elif value_ratio <= 1.15:
            scores['value'] = 10
            explanations.append(f"Slightly overpriced ({(value_ratio-1)*100:.0f}% above city avg)")
        else:
            scores['value'] = 5
            explanations.append(f"Significantly overpriced ({(value_ratio-1)*100:.0f}% above city avg)")
    else:
        scores['value'] = 12  # Neutral if no comparison
        explanations.append("Value assessment unavailable (no city comparison)")
    
    # 4. Buy Recommendation Component (0-15 points)
    decision = property_data.get('decision', '').lower()
    if 'buy' in decision:
        scores['recommendation'] = 15
        explanations.append("Buy recommendation based on wealth projection")
    else:
        scores['recommendation'] = 5
        explanations.append("Rent recommendation - better alternatives may exist")
    
    # 5. Risk Penalty (0 to -10 points)
    city = property_data.get('city', '')
    if city:
        risk_summary = get_risk_summary(city)
        overall_risk = risk_summary.get('overall_risk_score', 50)
        if overall_risk >= 60:
            scores['risk_penalty'] = -10
            explanations.append(f"High risk penalty (-10) for elevated market risk")
        elif overall_risk >= 30:
            scores['risk_penalty'] = -5
            explanations.append(f"Moderate risk penalty (-5)")
        else:
            scores['risk_penalty'] = 0
            explanations.append("No risk penalty - favorable risk profile")
    else:
        scores['risk_penalty'] = -5  # Default moderate penalty if city unknown
    
    # Calculate total score
    total_score = sum(scores.values())
    total_score = max(0, min(100, total_score))  # Clamp to 0-100
    
    # Generate grade
    if total_score >= 80:
        grade = "A"
        grade_explanation = "Excellent investment opportunity"
    elif total_score >= 65:
        grade = "B"
        grade_explanation = "Good investment potential"
    elif total_score >= 50:
        grade = "C"
        grade_explanation = "Average opportunity - careful evaluation needed"
    elif total_score >= 35:
        grade = "D"
        grade_explanation = "Below average - consider alternatives"
    else:
        grade = "F"
        grade_explanation = "Poor investment metrics - not recommended"
    
    return {
        "total_score": total_score,
        "grade": grade,
        "grade_explanation": grade_explanation,
        "component_scores": scores,
        "explanations": explanations,
        "methodology": "Score = ROI(30) + Yield(20) + Value(25) + Recommendation(15) - Risk(10 max)"
    }


# ============================================================================
# CITY-LEVEL INVESTMENT PROFILES
# ============================================================================

def get_city_investment_profile(city: str) -> Dict[str, Any]:
    """
    Generate comprehensive investment profile for a city.
    
    Includes:
    - Avg price per sqft
    - Avg rental yield
    - Market signal (Overpriced/Fair/Undervalued)
    - Trend direction
    - Risk summary
    - Investment opportunities count
    
    Returns:
        Dict with city investment profile
    """
    df = _get_df()
    if df.empty:
        return {"error": "No data available"}
    
    city_df = df[df['city'].str.lower() == city.lower()]
    if city_df.empty:
        return {"error": f"No data found for city: {city}"}
    
    # Basic metrics
    avg_price = city_df['price'].mean()
    avg_price_per_sqft = city_df['price_per_sqft'].mean()
    total_properties = len(city_df)
    
    # Calculate rental yield if data available
    if 'estimated_rent' in city_df.columns:
        avg_rent = city_df['estimated_rent'].mean()
        avg_rental_yield = (avg_rent * 12 / avg_price) * 100 if avg_price > 0 else 0
    else:
        avg_rent = avg_price * ASSUMPTIONS['rent_to_price_ratio']
        avg_rental_yield = ASSUMPTIONS['rent_to_price_ratio'] * 12 * 100
    
    # Buy vs Rent distribution
    buy_count = city_df['decision'].str.contains('buy', case=False, na=False).sum()
    buy_percentage = (buy_count / total_properties) * 100 if total_properties > 0 else 0
    
    # Calculate market signal
    all_df = _get_df()
    overall_avg_price_sqft = all_df['price_per_sqft'].mean()
    price_ratio = avg_price_per_sqft / overall_avg_price_sqft if overall_avg_price_sqft > 0 else 1
    
    if price_ratio < 0.85:
        market_signal = "Undervalued"
        market_explanation = f"Prices {(1-price_ratio)*100:.0f}% below market average - potential value opportunity"
    elif price_ratio > 1.15:
        market_signal = "Overpriced"
        market_explanation = f"Prices {(price_ratio-1)*100:.0f}% above market average - premium market"
    else:
        market_signal = "Fair"
        market_explanation = "Prices in line with market averages"
    
    # Calculate trend signal (based on price distribution)
    trend_signal = calculate_trend_signal(city)
    
    # Risk summary
    risk = get_risk_summary(city)
    
    # Find top opportunities (high investment score properties)
    opportunities = []
    for _, prop in city_df.head(20).iterrows():
        prop_dict = prop.to_dict()
        score = calculate_investment_score(prop_dict, {"avg_price_per_sqft": avg_price_per_sqft})
        if score['total_score'] >= 60:
            opportunities.append({
                "location": prop_dict.get('location'),
                "score": score['total_score'],
                "grade": score['grade']
            })
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        "city": city.title(),
        "metrics": {
            "total_properties": total_properties,
            "avg_price": round(avg_price, 0),
            "avg_price_crore": round(avg_price / 10000000, 2),
            "avg_price_per_sqft": round(avg_price_per_sqft, 0),
            "avg_rental_yield": round(avg_rental_yield, 2),
            "buy_recommendation_percent": round(buy_percentage, 1),
        },
        "market_signal": market_signal,
        "market_explanation": market_explanation,
        "trend": trend_signal,
        "risk_summary": {
            "overall_level": risk['overall_risk_level'],
            "overall_score": risk['overall_risk_score'],
            "explanation": risk['overall_explanation']
        },
        "top_opportunities": opportunities[:5],
        "data_source": f"Based on {total_properties} properties in {city.title()} from dataset"
    }


def get_all_city_profiles() -> Dict[str, Any]:
    """Get investment profiles for all cities in the dataset."""
    df = _get_df()
    if df.empty:
        return {"error": "No data available"}
    
    cities = df['city'].unique().tolist()
    profiles = {}
    
    for city in cities:
        profile = get_city_investment_profile(city)
        if 'error' not in profile:
            profiles[city.lower()] = profile
    
    # Sort by buy recommendation percentage
    sorted_cities = sorted(
        profiles.items(),
        key=lambda x: x[1]['metrics']['buy_recommendation_percent'],
        reverse=True
    )
    
    return {
        "total_cities": len(profiles),
        "profiles": dict(sorted_cities),
        "methodology": "City profiles based on aggregated property metrics and risk analysis"
    }


# ============================================================================
# TREND SIGNALS (Rule-Based)
# ============================================================================

def calculate_trend_signal(city: str = None) -> Dict[str, Any]:
    """
    Calculate market trend signal based on listing density and price distribution.
    
    HEURISTICS:
    - Overheated: High prices + Low inventory + High price variance
    - Stable: Moderate metrics across the board
    - Cooling: Lower prices + High inventory + Low variance
    
    Returns:
        Dict with signal, confidence, explanation
    """
    df = _get_df()
    if df.empty:
        return {"signal": "unknown", "confidence": "low", "explanation": "No data available"}
    
    all_df = df.copy()
    if city:
        city_df = df[df['city'].str.lower() == city.lower()]
        if city_df.empty:
            return {"signal": "unknown", "confidence": "low", "explanation": f"No data for {city}"}
    else:
        city_df = df
    
    # Calculate metrics for trend analysis
    city_avg_price_sqft = city_df['price_per_sqft'].mean()
    overall_avg_price_sqft = all_df['price_per_sqft'].mean()
    price_ratio = city_avg_price_sqft / overall_avg_price_sqft if overall_avg_price_sqft > 0 else 1
    
    listing_count = len(city_df)
    avg_listings_per_city = len(all_df) / all_df['city'].nunique() if all_df['city'].nunique() > 0 else 50
    listing_ratio = listing_count / avg_listings_per_city if avg_listings_per_city > 0 else 1
    
    price_cv = (city_df['price_per_sqft'].std() / city_df['price_per_sqft'].mean() * 100) if city_df['price_per_sqft'].mean() > 0 else 0
    
    # Determine signal
    score = 0  # Higher = more overheated
    
    if price_ratio > 1.2:
        score += 2
    elif price_ratio > 1.0:
        score += 1
    elif price_ratio < 0.8:
        score -= 2
    elif price_ratio < 1.0:
        score -= 1
    
    if listing_ratio < 0.5:
        score += 1  # Low inventory = heating
    elif listing_ratio > 1.5:
        score -= 1  # High inventory = cooling
    
    if price_cv > 50:
        score += 1  # High variance = volatility/heating
    elif price_cv < 25:
        score -= 1  # Low variance = stability
    
    # Determine signal based on score
    if score >= 2:
        signal = "Overheated"
        explanation = f"Market shows signs of overheating: prices {(price_ratio-1)*100:.0f}% above average, limited inventory"
        confidence = "medium" if score == 2 else "high"
    elif score <= -2:
        signal = "Cooling"
        explanation = f"Market shows cooling signs: prices {(1-price_ratio)*100:.0f}% below average, ample inventory"
        confidence = "medium" if score == -2 else "high"
    else:
        signal = "Stable"
        explanation = "Market conditions are stable with balanced supply and pricing"
        confidence = "medium"
    
    return {
        "signal": signal,
        "confidence": confidence,
        "explanation": explanation,
        "metrics": {
            "price_vs_market": f"{price_ratio*100:.0f}%",
            "listing_density": f"{listing_ratio*100:.0f}% of avg",
            "price_variance": f"{price_cv:.1f}%"
        }
    }


# ============================================================================
# INVESTMENT CONTEXT EXPLANATIONS
# ============================================================================

def generate_metric_explanation(metric_name: str, value: float, context: Dict = None) -> str:
    """
    Generate investor-friendly explanation for a metric value.
    
    Args:
        metric_name: Name of the metric (roi, rental_yield, price_per_sqft, etc.)
        value: The metric value
        context: Optional context dict (city_avg, market_avg, etc.)
    
    Returns:
        str: Explanation of what the metric means for investors
    """
    explanations = {
        "roi": _explain_roi,
        "rental_yield": _explain_rental_yield,
        "price_per_sqft": _explain_price_per_sqft,
        "wealth_buying": _explain_wealth,
        "investment_score": _explain_investment_score,
    }
    
    explainer = explanations.get(metric_name.lower())
    if explainer:
        return explainer(value, context)
    
    return f"{metric_name}: {value}"


def _explain_roi(value: float, context: Dict = None) -> str:
    """Explain ROI value."""
    if value >= 150:
        quality = "Excellent"
        implication = "significantly outperforms typical investments"
    elif value >= 100:
        quality = "Strong"
        implication = "above average returns expected"
    elif value >= 50:
        quality = "Moderate"
        implication = "reasonable growth potential"
    elif value >= 0:
        quality = "Low"
        implication = "limited appreciation expected"
    else:
        quality = "Negative"
        implication = "potential value decline"
    
    base = f"**ROI: {value:.1f}%** ({quality})\n"
    base += f"This means: Over the holding period, the total return (appreciation + rental income) relative to purchase price is {value:.1f}%.\n"
    base += f"Implication: {implication.capitalize()}."
    
    if context and context.get('city_avg_roi'):
        diff = value - context['city_avg_roi']
        comparison = "above" if diff > 0 else "below"
        base += f"\nCompared to city average: {abs(diff):.1f}% {comparison}."
    
    return base


def _explain_rental_yield(value: float, context: Dict = None) -> str:
    """Explain rental yield value."""
    if value >= 5:
        quality = "High"
        implication = "strong passive income potential"
    elif value >= 4:
        quality = "Good"
        implication = "decent rental income"
    elif value >= 3:
        quality = "Average"
        implication = "typical for Indian markets"
    else:
        quality = "Below Average"
        implication = "rental income may not cover costs"
    
    base = f"**Rental Yield: {value:.2f}%** ({quality})\n"
    base += f"This means: Annual rental income is {value:.2f}% of property value.\n"
    base += f"For a ₹1 Cr property, expect ~₹{value*10000:.0f}/month rent.\n"
    base += f"Implication: {implication.capitalize()}."
    
    return base


def _explain_price_per_sqft(value: float, context: Dict = None) -> str:
    """Explain price per sqft value."""
    base = f"**Price per Sqft: ₹{value:,.0f}**\n"
    
    if context and context.get('city_avg'):
        city_avg = context['city_avg']
        diff_pct = ((value - city_avg) / city_avg) * 100 if city_avg > 0 else 0
        
        if diff_pct < -15:
            assessment = "Significantly undervalued"
        elif diff_pct < 0:
            assessment = "Below market average"
        elif diff_pct < 15:
            assessment = "At market rate"
        else:
            assessment = "Premium pricing"
        
        base += f"City average: ₹{city_avg:,.0f}/sqft\n"
        base += f"This property: {abs(diff_pct):.1f}% {'below' if diff_pct < 0 else 'above'} average\n"
        base += f"Assessment: {assessment}"
    
    return base


def _explain_wealth(value: float, context: Dict = None) -> str:
    """Explain wealth projection value."""
    value_cr = value / 10000000
    base = f"**Projected Wealth: ₹{value_cr:.2f} Cr**\n"
    base += f"This is the estimated net worth after {ASSUMPTIONS['holding_period_years']} years.\n"
    
    if context and context.get('wealth_renting'):
        rent_wealth = context['wealth_renting'] / 10000000
        diff = value_cr - rent_wealth
        
        if diff > 0:
            base += f"Buying advantage: ₹{diff:.2f} Cr more wealth vs renting.\n"
            base += f"Implication: Ownership builds more wealth in this scenario."
        else:
            base += f"Renting advantage: ₹{abs(diff):.2f} Cr more wealth vs buying.\n"
            base += f"Implication: Rent and invest difference for better returns."
    
    return base


def _explain_investment_score(value: float, context: Dict = None) -> str:
    """Explain investment score."""
    if value >= 80:
        grade = "A"
        quality = "Excellent"
    elif value >= 65:
        grade = "B"
        quality = "Good"
    elif value >= 50:
        grade = "C"
        quality = "Average"
    elif value >= 35:
        grade = "D"
        quality = "Below Average"
    else:
        grade = "F"
        quality = "Poor"
    
    base = f"**Investment Score: {value}/100 (Grade: {grade})**\n"
    base += f"Overall assessment: {quality} investment opportunity.\n\n"
    base += "Score Components:\n"
    base += "- ROI potential (30 pts)\n"
    base += "- Rental yield (20 pts)\n"
    base += "- Value vs market (25 pts)\n"
    base += "- Buy recommendation (15 pts)\n"
    base += "- Risk adjustment (-10 pts max)"
    
    return base


# ============================================================================
# SCENARIO ANALYSIS
# ============================================================================

def run_scenario_analysis(property_price: float, monthly_rent: float, scenario: str = "moderate") -> Dict[str, Any]:
    """
    Run buy vs rent analysis under different scenarios.
    
    Args:
        property_price: Property purchase price
        monthly_rent: Monthly rent amount
        scenario: 'conservative', 'moderate', or 'aggressive'
    
    Returns:
        Dict with scenario analysis results
    """
    if scenario not in SCENARIOS:
        scenario = "moderate"
    
    params = SCENARIOS[scenario]
    
    # Import analyzer for calculations
    from services.analysis import RealEstateAnalyzer
    analyzer = RealEstateAnalyzer()
    
    # Override default params with scenario params
    analysis_params = analyzer.default_params.copy()
    analysis_params['appreciation_rate'] = params['appreciation_rate']
    analysis_params['rent_escalation'] = params['rent_growth_rate']
    analysis_params['investment_return_rate'] = params['investment_return']
    analysis_params['loan_rate'] = params['loan_rate']
    
    # Run analysis
    result = analyzer.buy_vs_rent_analysis(property_price, monthly_rent, analysis_params)
    
    return {
        "scenario": params['name'],
        "description": params['description'],
        "assumptions": {
            "appreciation_rate": f"{params['appreciation_rate']}%",
            "rent_growth_rate": f"{params['rent_growth_rate']}%",
            "investment_return": f"{params['investment_return']}%",
            "loan_rate": f"{params['loan_rate']}%",
            "vacancy_rate": f"{params['vacancy_rate']}%",
        },
        "results": {
            "wealth_buying": result['buy_wealth'],
            "wealth_renting": result['rent_wealth'],
            "recommendation": result['recommendation'],
            "wealth_difference": result['wealth_difference'],
        }
    }


def run_all_scenarios(property_price: float, monthly_rent: float) -> Dict[str, Any]:
    """
    Run analysis under all three scenarios for comparison.
    
    Returns:
        Dict with all scenario results and summary
    """
    results = {}
    for scenario_key in SCENARIOS.keys():
        results[scenario_key] = run_scenario_analysis(property_price, monthly_rent, scenario_key)
    
    # Determine consensus
    buy_votes = sum(1 for r in results.values() if r['results']['recommendation'] == 'Buy')
    
    if buy_votes == 3:
        consensus = "Strong Buy"
        explanation = "All scenarios recommend buying - robust investment"
    elif buy_votes >= 2:
        consensus = "Buy"
        explanation = "Majority of scenarios recommend buying"
    elif buy_votes == 1:
        consensus = "Rent"
        explanation = "Majority of scenarios recommend renting"
    else:
        consensus = "Strong Rent"
        explanation = "All scenarios recommend renting - consider alternatives"
    
    return {
        "scenarios": results,
        "consensus": consensus,
        "explanation": explanation,
        "methodology": "Analysis run under Conservative, Moderate, and Aggressive assumptions"
    }


# ============================================================================
# FORMATTED OUTPUT FOR AI ASSISTANT
# ============================================================================

def format_investment_context(property_data: Dict, city: str = None) -> str:
    """
    Format comprehensive investment context for AI assistant.
    
    Returns a formatted string with all investment intelligence for a property.
    """
    lines = []
    
    # Get city stats for comparison
    from src.rag.sql_retriever import get_city_stats
    city_stats = get_city_stats(city) if city else {}
    
    # Investment Score
    score = calculate_investment_score(property_data, city_stats)
    lines.append(f"## Investment Score: {score['total_score']}/100 (Grade: {score['grade']})")
    lines.append(f"{score['grade_explanation']}")
    lines.append("")
    
    # Component breakdown
    lines.append("### Score Breakdown:")
    for explanation in score['explanations']:
        lines.append(f"- {explanation}")
    lines.append("")
    
    # Risk Assessment
    if city:
        risk = get_risk_summary(city)
        lines.append(f"### Risk Assessment: {risk['overall_risk_level'].upper()}")
        lines.append(f"- Price Volatility: {risk['price_volatility']['risk_level']}")
        lines.append(f"- Market Liquidity: {risk['liquidity']['risk_level']}")
        lines.append(f"- Rental Stability: {risk['rental_stability']['risk_level']}")
        lines.append(f"{risk['overall_explanation']}")
        lines.append("")
    
    # Market Context
    if city:
        trend = calculate_trend_signal(city)
        lines.append(f"### Market Trend: {trend['signal']}")
        lines.append(trend['explanation'])
        lines.append("")
    
    # Assumptions
    lines.append("### Assumptions Used:")
    lines.append(f"- Holding Period: {ASSUMPTIONS['holding_period_years']} years")
    lines.append(f"- Down Payment: {ASSUMPTIONS['down_payment_percent']}%")
    lines.append("- See INVESTMENT_METRICS.md for full methodology")
    
    return "\n".join(lines)
