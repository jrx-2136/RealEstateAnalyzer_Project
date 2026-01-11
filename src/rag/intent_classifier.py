# src/rag/intent_classifier.py

"""
Intent classifier for RAG queries.
Routes queries to appropriate retrieval strategy.

ENHANCED INTENT CLASSIFICATION:
- SPECIFIC_PROPERTY: Named property queries (exact/fuzzy match)
- AGGREGATE: Statistical queries (averages, counts, totals)
- FILTER: Property listing queries with filters
- COMPARE: Comparison between cities/locations
- RECOMMEND: Buy vs rent recommendations
- ADVISORY: Investment advice questions ("Is this a good investment?")
- CITY_PROFILE: City-level investment analysis
- LOCATION: Location-specific queries
- EDUCATIONAL: General real estate questions
- RISK: Risk assessment queries
- SCENARIO: Scenario-based analysis queries
"""

import re

INTENTS = {
    "SPECIFIC_PROPERTY": "Queries about a specific named property/project",
    "AGGREGATE": "Statistical queries (averages, counts, totals)",
    "FILTER": "Property listing queries with filters",
    "COMPARE": "Comparison between cities/locations",
    "RECOMMEND": "Buy vs rent recommendations",
    "ADVISORY": "Investment quality/advice questions",
    "CITY_PROFILE": "City-level investment profile requests",
    "LOCATION": "Location-specific queries",
    "EDUCATIONAL": "General real estate questions",
    "RISK": "Risk assessment queries",
    "SCENARIO": "Scenario-based analysis (conservative/aggressive)",
}


def detect_specific_property_query(query: str, property_names: list = None) -> tuple:
    """
    Detect if query is asking about a specific named property.
    
    PRECISION FIX: This function identifies queries that should use exact/fuzzy
    matching instead of broad vector search.
    
    Heuristics used:
    1. Quoted property names ("Tulip Infinity")
    2. "tell me about X" / "details of X" / "info on X" patterns
    3. Capitalized multi-word phrases that match property names
    4. High similarity against known property names
    
    Args:
        query: User query string
        property_names: List of known property names from database
        
    Returns:
        tuple: (is_specific_query: bool, extracted_name: str or None)
    """
    q = query.strip()
    q_lower = q.lower()
    
    # Pattern 1: Quoted names - highest confidence
    quoted_match = re.search(r'["\']([^"\']+)["\']', q)
    if quoted_match:
        return (True, quoted_match.group(1).strip())
    
    # Pattern 2: "tell me about X" / "details of X" / "info on X" / "what is X"
    specific_patterns = [
        r'tell me about\s+(.+?)(?:\s+property|\s+project|\s*$)',
        r'details (?:of|about|for)\s+(.+?)(?:\s+property|\s+project|\s*$)',
        r'info(?:rmation)? (?:on|about|for)\s+(.+?)(?:\s+property|\s+project|\s*$)',
        r'what is\s+(.+?)(?:\s+property|\s+project|\?|\s*$)',
        r'show me\s+(.+?)(?:\s+property|\s+project|\s*$)',
        r'about\s+(.+?)(?:\s+property|\s+project|\s*$)',
    ]
    
    for pattern in specific_patterns:
        match = re.search(pattern, q_lower)
        if match:
            extracted = match.group(1).strip()
            # Filter out generic terms
            generic_terms = ['properties', 'property', 'the', 'a', 'an', 'this', 'that', 'some']
            if extracted and extracted not in generic_terms and len(extracted) > 2:
                return (True, extracted)
    
    # Pattern 3: Check for capitalized multi-word phrases (property names)
    # Look for 2+ consecutive capitalized words
    cap_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', q)
    if cap_phrases:
        # Return the longest capitalized phrase as likely property name
        longest = max(cap_phrases, key=len)
        if len(longest) > 5:  # At least "X Y" format
            return (True, longest)
    
    # Pattern 4: If property_names provided, check for fuzzy match
    if property_names:
        best_match = find_best_property_match(q, property_names, threshold=0.7)
        if best_match:
            return (True, best_match)
    
    return (False, None)


def find_best_property_match(query: str, property_names: list, threshold: float = 0.6) -> str:
    """
    Find the best matching property name using fuzzy matching.
    
    PRECISION FIX: Uses simple similarity scoring to find exact/near-exact matches.
    
    Args:
        query: User query string
        property_names: List of property names to match against
        threshold: Minimum similarity score (0-1) to consider a match
        
    Returns:
        str: Best matching property name, or None if no good match
    """
    from difflib import SequenceMatcher
    
    q_lower = query.lower()
    best_score = 0
    best_match = None
    
    for prop_name in property_names:
        prop_lower = prop_name.lower()
        
        # Check if property name is contained in query (substring match)
        if prop_lower in q_lower:
            # Exact substring match - high confidence
            score = 0.95
        else:
            # Fuzzy match using SequenceMatcher
            score = SequenceMatcher(None, q_lower, prop_lower).ratio()
            
            # Also check each word of property name
            prop_words = prop_lower.split()
            query_words = q_lower.split()
            
            # Boost score if all property words appear in query
            words_found = sum(1 for pw in prop_words if any(pw in qw or qw in pw for qw in query_words))
            if words_found == len(prop_words) and len(prop_words) >= 2:
                score = max(score, 0.85)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = prop_name
    
    return best_match


def classify_intent(query: str) -> str:
    """
    Classify user query intent for routing to appropriate retrieval.
    
    ENHANCED INTENT DETECTION:
    - Prioritizes specific intents over general ones
    - Better handling of advisory/investment quality questions
    - Support for risk and scenario queries
    
    Returns:
        str: Intent type (AGGREGATE, FILTER, COMPARE, RECOMMEND, ADVISORY, 
             CITY_PROFILE, LOCATION, EDUCATIONAL, RISK, SCENARIO)
    """
    q = query.lower().strip()
    
    # SCENARIO: Scenario-based analysis queries
    scenario_keywords = [
        "conservative", "aggressive", "moderate scenario",
        "what if", "scenario", "assumption", "if interest rate",
        "if appreciation", "sensitivity"
    ]
    if any(kw in q for kw in scenario_keywords):
        return "SCENARIO"
    
    # RISK: Risk assessment queries
    risk_keywords = [
        "risk", "risky", "volatile", "volatility", "safe",
        "liquidity", "stability", "how stable", "dangerous"
    ]
    if any(kw in q for kw in risk_keywords):
        return "RISK"
    
    # ADVISORY: Investment quality/advice questions - CHECK EARLY
    # These are "Is this a good investment?" type questions
    advisory_keywords = [
        "good investment", "worth investing", "should i invest",
        "investment quality", "is it worth", "worthwhile",
        "smart investment", "wise to buy", "viable investment",
        "investment potential", "investment opportunity",
        "good deal", "bad deal", "fair price", "overpriced",
        "undervalued", "good buy", "worth the price",
        "investment score", "rate this", "evaluate this"
    ]
    if any(kw in q for kw in advisory_keywords):
        return "ADVISORY"
    
    # CITY_PROFILE: City-level investment profile requests
    city_profile_keywords = [
        "city profile", "investment profile", "market overview",
        "how is the market in", "market in", "investment in",
        "overall", "general", "market conditions", "city analysis",
        "investment outlook", "market sentiment"
    ]
    if any(kw in q for kw in city_profile_keywords):
        return "CITY_PROFILE"
    
    # AGGREGATE: Statistical/aggregate queries
    aggregate_keywords = [
        "average", "avg", "mean", "total", "count", "how many",
        "number of", "statistics", "stats", "sum", "minimum", "maximum",
        "min ", "max ", "highest", "lowest", "cheapest", "expensive",
        "most expensive", "least expensive"
    ]
    if any(kw in q for kw in aggregate_keywords):
        return "AGGREGATE"
    
    # RECOMMEND: Buy vs rent / investment advice - CHECK BEFORE COMPARE (has "or")
    recommend_keywords = [
        "should i buy", "should i rent", "buy or rent", "recommend",
        "suggestion", "advice", "invest", "worth buying", "worth it",
        "decision", "what should", "buying or renting",
        "renting or buying", "rent or buy", "better to buy", "better to rent"
    ]
    if any(kw in q for kw in recommend_keywords):
        return "RECOMMEND"
    
    # COMPARE: Comparison queries
    compare_keywords = [
        "compare", "versus", " vs ", "difference between", 
        "better than", "cheaper than", "which is better",
        "compared to", "comparison", "between"
    ]
    # Check for city comparison pattern
    if any(kw in q for kw in compare_keywords):
        return "COMPARE"
    
    # FILTER: Property listing queries with filters (BHK, price range, etc.)
    filter_keywords = [
        "bhk", "bedroom", "1bhk", "2bhk", "3bhk", "4bhk", "5bhk",
        "under", "below", "above", "budget", "price range",
        "between", "less than", "more than", "crore", "lakh",
        "sqft", "square feet", "top properties", "best properties"
    ]
    if any(kw in q for kw in filter_keywords):
        return "FILTER"
    
    # LOCATION: Location-specific queries
    location_keywords = [
        "locations in", "areas in", "places in", "localities in",
        "where in", "properties in", "list of", "show me", "find"
    ]
    if any(kw in q for kw in location_keywords):
        return "LOCATION"
    
    # EDUCATIONAL: General knowledge questions
    educational_keywords = [
        "what is", "how does", "explain", "why", "define", "meaning",
        "how is calculated", "formula", "methodology"
    ]
    if any(kw in q for kw in educational_keywords):
        return "EDUCATIONAL"
    
    # Check for city mentions without other intent - likely CITY_PROFILE
    from src.rag.sql_retriever import get_available_cities
    cities = get_available_cities()
    city_mentioned = any(city.lower() in q for city in cities)
    if city_mentioned and len(q.split()) <= 5:
        return "CITY_PROFILE"
    
    # Default: FILTER for property queries
    return "FILTER"


def extract_scenario_from_query(query: str) -> str:
    """
    Extract scenario type from query.
    
    Returns:
        str: 'conservative', 'moderate', 'aggressive', or None
    """
    q = query.lower()
    
    if 'conservative' in q or 'cautious' in q or 'safe' in q:
        return 'conservative'
    elif 'aggressive' in q or 'optimistic' in q:
        return 'aggressive'
    elif 'moderate' in q or 'balanced' in q:
        return 'moderate'
    
    return None
    if any(kw in q for kw in aggregate_keywords):
        return "AGGREGATE"
    
    # RECOMMEND: Buy vs rent / investment advice - CHECK BEFORE COMPARE (has "or")
    recommend_keywords = [
        "should i buy", "should i rent", "buy or rent", "recommend",
        "suggestion", "advice", "invest", "worth buying", "worth it",
        "good investment", "decision", "what should", "buying or renting",
        "renting or buying", "rent or buy"
    ]
    if any(kw in q for kw in recommend_keywords):
        return "RECOMMEND"
    
    # COMPARE: Comparison queries
    compare_keywords = [
        "compare", "versus", " vs ", "difference between", 
        "better than", "cheaper than", "which is better",
        "compared to"
    ]
    # Exclude "or" alone - too broad, catches "buy or rent"
    if any(kw in q for kw in compare_keywords):
        return "COMPARE"
    
    # FILTER: Property listing queries with filters (BHK, price range, etc.)
    filter_keywords = [
        "bhk", "bedroom", "1bhk", "2bhk", "3bhk", "4bhk", "5bhk",
        "under", "below", "above", "budget", "price range",
        "between", "less than", "more than", "crore"
    ]
    if any(kw in q for kw in filter_keywords):
        return "FILTER"
    
    # LOCATION: Location-specific queries
    location_keywords = [
        "locations in", "areas in", "places in", "localities in",
        "where in", "properties in", "list of", "show me", "find"
    ]
    if any(kw in q for kw in location_keywords):
        return "LOCATION"
    
    # EDUCATIONAL: General knowledge questions
    educational_keywords = [
        "what is", "how does", "explain", "why", "define", "meaning"
    ]
    if any(kw in q for kw in educational_keywords):
        return "EDUCATIONAL"
    
    # FILTER: Default for property queries
    return "FILTER"


def extract_cities_from_query(query: str, available_cities: list) -> list:
    """
    Extract city names mentioned in the query.
    
    Args:
        query: User query string
        available_cities: List of cities in the dataset
        
    Returns:
        List of detected city names (lowercase)
    """
    q = query.lower()
    detected = []
    
    # Normalize city name variations to match dataset
    # Dataset cities: ahmedabad, aurangabad, bhopal, bhubaneswar, bilaspur, chennai, coimbatore, 
    # cuttack, gaya, gurgaon, hyderabad, indore, jaipur, jamshedpur, kanpur, kochi, kolkata, 
    # kottayam, lucknow, mumbai, mysore, nagpur, nashik, navi-mumbai, new-delhi, noida, patna, 
    # pune, raipur, ranchi, surat, thane, thrissur, trivandrum, udaipur, vadodara
    city_aliases = {
        "new-delhi": ["delhi", "new delhi", "new-delhi", "ncr"],
        "mumbai": ["mumbai", "bombay"],
        "chennai": ["chennai", "madras"],
        "kolkata": ["kolkata", "calcutta"],
        "hyderabad": ["hyderabad", "hyd"],
        "pune": ["pune", "poona"],
        "gurgaon": ["gurgaon", "gurugram", "ggn"],
        "noida": ["noida", "greater noida"],
        "navi-mumbai": ["navi mumbai", "navi-mumbai", "new mumbai"],
        "trivandrum": ["trivandrum", "thiruvananthapuram"],
        "kochi": ["kochi", "cochin"],
        "thrissur": ["thrissur", "trichur"],
    }
    
    # Check aliases first
    for canonical, aliases in city_aliases.items():
        for alias in aliases:
            if alias in q:
                # Find the actual city name in available_cities
                for city in available_cities:
                    if city.lower() == canonical or canonical in city.lower():
                        detected.append(city.lower())
                        break
                break
    
    # Then check available cities directly
    for city in available_cities:
        city_lower = city.lower()
        if city_lower in q and city_lower not in detected:
            detected.append(city_lower)
    
    return detected


def extract_bhk_from_query(query: str) -> int:
    """Extract BHK number from query if mentioned."""
    import re
    q = query.lower()
    
    # Match patterns like "2bhk", "2 bhk", "2-bhk", "two bhk"
    patterns = [
        r'(\d)\s*bhk',
        r'(\d)\s*-\s*bhk',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))
    
    # Word numbers
    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for word, num in word_to_num.items():
        if f"{word} bhk" in q or f"{word}bhk" in q:
            return num
    
    return None
