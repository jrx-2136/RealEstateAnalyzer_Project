# src/rag/rag_engine.py

"""
RAG Engine for data-grounded real estate responses.
All responses are strictly derived from CSV data through embeddings and SQL.

RELIABILITY FIXES:
- LLM call throttling (max 5 calls per minute)
- Response caching to prevent duplicate API calls
- Timeout protection with graceful fallbacks
- Guaranteed string returns on all paths
"""

import time
import threading
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================================
# LLM THROTTLING: Prevent API rate limit errors
# ============================================================================
class LLMThrottler:
    """
    Simple in-memory rate limiter for LLM calls.
    Limits to MAX_CALLS within TIME_WINDOW seconds.
    """
    MAX_CALLS = 5           # Maximum LLM calls allowed
    TIME_WINDOW = 60        # Time window in seconds (1 minute)
    
    def __init__(self):
        self.calls = []      # Timestamps of recent calls
        self.lock = threading.Lock()
    
    def can_call(self) -> bool:
        """Check if we can make another LLM call."""
        with self.lock:
            now = time.time()
            # Remove calls outside the time window
            self.calls = [t for t in self.calls if now - t < self.TIME_WINDOW]
            return len(self.calls) < self.MAX_CALLS
    
    def record_call(self):
        """Record a new LLM call."""
        with self.lock:
            self.calls.append(time.time())
    
    def get_wait_time(self) -> int:
        """Get seconds until next call is allowed."""
        with self.lock:
            if not self.calls:
                return 0
            oldest = min(self.calls)
            return max(0, int(self.TIME_WINDOW - (time.time() - oldest)))


# Global throttler instance
_llm_throttler = LLMThrottler()


# ============================================================================
# RESPONSE CACHE: Avoid duplicate LLM calls for same queries
# ============================================================================
# Simple in-memory cache using LRU with max 100 entries
@lru_cache(maxsize=100)
def _cached_llm_response(query_hash: str, context_hash: str, intent: str) -> str:
    """
    Cached wrapper - actual call happens in generate_rag_response.
    This function exists to enable caching by query+context hash.
    """
    # This is a placeholder - actual implementation returns None
    # The cache decorator stores results from generate_rag_response
    return None


# Store for actual cached responses (LRU cache only works with hashable args)
_response_cache = {}
_cache_lock = threading.Lock()

def get_cached_response(query: str, context_preview: str) -> str:
    """Get cached response if available."""
    cache_key = f"{query[:100]}|{context_preview[:100]}"
    with _cache_lock:
        return _response_cache.get(cache_key)

def set_cached_response(query: str, context_preview: str, response: str):
    """Cache a response. Limit cache size to 50 entries."""
    cache_key = f"{query[:100]}|{context_preview[:100]}"
    with _cache_lock:
        if len(_response_cache) >= 50:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(_response_cache))
            del _response_cache[oldest_key]
        _response_cache[cache_key] = response


SYSTEM_PROMPT = """You are a real estate data assistant. You ONLY answer questions using the EXACT data provided below from our property database.

STRICT DATA GROUNDING RULES:
1. ONLY use numbers, prices, locations, and facts that appear in the "Retrieved Data" section below.
2. NEVER invent, estimate, or hallucinate any property data, prices, or statistics.
3. If specific data is not in the retrieved records, say: "I don't have that specific information in the current database."
4. Always cite the source data when providing answers (e.g., "According to the database..." or "The data shows...").
5. For aggregate questions (averages, counts), only use pre-calculated statistics if provided, or state what individual records show.
6. Do NOT perform mathematical calculations beyond what's explicitly shown in the data.
7. If asked about a city/location not in the data, explicitly state it's not in the database.

RESPONSE FORMAT:
- Be concise and direct
- Use bullet points for multiple properties
- Include actual prices and numbers from the data
- State data limitations clearly

You are NOT a general chatbot. You are a data retrieval assistant for this specific property database."""


def generate_rag_response(context_docs: list[str], user_query: str, intent: str = None):
    """
    Generate a response using RAG with retrieved context documents.
    Strictly grounded in the provided data.
    
    RELIABILITY FEATURES:
    - Checks cache first to avoid duplicate LLM calls
    - Throttles LLM calls to prevent rate limiting
    - Timeout protection with graceful fallback
    - Guaranteed string return on all paths
    
    Args:
        context_docs: List of retrieved document contents from vector store/SQL
        user_query: The user's question
        intent: Query intent type (AGGREGATE, FILTER, COMPARE, etc.)
        
    Returns:
        str: Generated response based only on the provided context (NEVER None)
    """
    context = "\n\n---\n\n".join(context_docs) if context_docs else ""
    
    # FIX: Check if context is meaningful - return early with helpful message
    if not context or len(context.strip()) < 20:
        return "I couldn't find relevant property data in the database for your query. The database contains properties from cities like Mumbai, Pune, Delhi, Bangalore, and others. Try asking about:\n- Average prices in a specific city\n- Properties in a location\n- Buy vs rent recommendations"

    # FIX: Check cache first to avoid duplicate LLM calls
    cached = get_cached_response(user_query, context[:200])
    if cached:
        print(f"📦 Cache hit for query: {user_query[:50]}...")
        return cached

    # FIX: Check throttle before making LLM call
    if not _llm_throttler.can_call():
        wait_time = _llm_throttler.get_wait_time()
        print(f"⚠️ LLM throttled. Wait time: {wait_time}s")
        # Return data summary instead of calling LLM
        return f"""⚠️ AI insights are temporarily limited due to API usage constraints. Here's the data I found:

{context[:2000]}

📊 Data-based answers are still available. Try asking specific questions about properties or prices."""

    # Add intent-specific instructions
    intent_guidance = ""
    if intent == "AGGREGATE":
        intent_guidance = "\nThe user is asking for statistics. Only provide numbers that are explicitly stated in the data. Do not calculate new statistics."
    elif intent == "LOCATION":
        intent_guidance = "\nThe user is asking about locations. List only the locations that appear in the retrieved data."
    elif intent == "COMPARE":
        intent_guidance = "\nThe user is comparing options. Compare only using data points explicitly available in the records."
    elif intent == "RECOMMEND":
        intent_guidance = "\nThe user wants a recommendation. Base your recommendation ONLY on the buy/rent decisions already in the data."

    prompt = f"""{SYSTEM_PROMPT}
{intent_guidance}

=== RETRIEVED DATA FROM DATABASE ===
{context}
=== END OF DATA ===

User Question: {user_query}

Instructions:
1. Answer using ONLY the data above
2. If the data doesn't contain the answer, say so clearly
3. Quote specific numbers and locations from the data
4. Do not make up any information

Your data-grounded response:"""

    # FIX: Wrap LLM call with timeout and comprehensive error handling
    max_retries = 2  # Reduced retries to prevent long waits
    for attempt in range(max_retries):
        try:
            # Record the call for throttling
            _llm_throttler.record_call()
            
            # Initialize LLM with timeout
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,  # Zero temperature for factual responses
                timeout=30      # 30 second timeout
            )
            
            response = llm.invoke(prompt).content
            
            # FIX: Validate response before returning
            if response and isinstance(response, str) and len(response.strip()) > 0:
                # Cache successful response
                set_cached_response(user_query, context[:200], response)
                return response
            else:
                # Invalid response from LLM
                return f"I found relevant data but couldn't generate a proper summary. Here's what the database contains:\n\n{context[:1500]}"
                
        except Exception as e:
            error_str = str(e).lower()
            print(f"❌ LLM error (attempt {attempt + 1}): {str(e)[:100]}")
            
            # Handle rate limiting
            if "429" in str(e) or "resource_exhausted" in error_str or "quota" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10s, 20s backoff (reduced)
                    time.sleep(wait_time)
                    continue
                else:
                    # Return data without LLM summary
                    return f"""⚠️ AI service is temporarily rate-limited. Here's the raw data from the database:

{context[:2000]}

Please try again in a few moments for an AI-generated summary."""
            
            # Handle timeout
            elif "timeout" in error_str or "timed out" in error_str:
                return f"The request timed out. Here's the data I found:\n\n{context[:1500]}\n\nPlease try a more specific question."
            
            # Handle other errors
            else:
                return f"I found data but encountered an issue generating the summary. Here's what's in the database:\n\n{context[:1500]}"
    
    # FIX: Final fallback - NEVER return None
    return "I couldn't generate a complete response. Please try rephrasing your question or ask about a specific city like Mumbai, Pune, or Delhi."


def generate_no_data_response(query: str, available_cities: list = None) -> str:
    """
    Generate a helpful response when no relevant data is found.
    """
    cities_str = ", ".join(available_cities[:10]) if available_cities else "various Indian cities"
    
    return f"""I couldn't find specific data matching your query in the current database.

The database contains {len(available_cities) if available_cities else 'properties from'} cities including: {cities_str}.

You can try asking:
• "What is the average price in Mumbai?"
• "Show me properties in Pune"
• "Which locations in Delhi have buy recommendations?"
• "Compare prices between Mumbai and Pune"
• "What are the cheapest properties in Bangalore?"

Please rephrase your question with a specific city or location from the database."""


def generate_filter_response(properties: list, city: str = None, bhk: int = None) -> str:
    """
    Generate a direct response for FILTER intent queries using SQL data only.
    This bypasses the LLM to provide fast, reliable responses for property listings.
    
    Args:
        properties: List of property dictionaries from SQL retrieval
        city: City filter applied (optional)
        bhk: BHK filter applied (optional)
        
    Returns:
        str: Human-readable summary of filtered properties
    """
    # FIX: Always return a string - never return None
    if not properties:
        location_hint = f" in {city.title()}" if city else ""
        bhk_hint = f" with {bhk} BHK" if bhk else ""
        return f"No properties found{location_hint}{bhk_hint} in the current dataset. Try a different city or remove filters."
    
    count = len(properties)
    city_name = city.title() if city else "the database"
    bhk_info = f" ({bhk} BHK)" if bhk else ""
    
    # Build response header
    response_lines = [f"Found {count} properties in {city_name}{bhk_info}:\n"]
    
    # Format each property
    for i, prop in enumerate(properties[:8], 1):  # Limit to 8 for readability
        price = prop.get('price', 0)
        price_cr = price / 10000000 if price else 0
        location = prop.get('location', 'Unknown Location')
        area = prop.get('area_sqft', 'N/A')
        prop_bhk = prop.get('bhk', 'N/A')
        price_per_sqft = prop.get('price_per_sqft', 0)
        decision = prop.get('decision', 'N/A')
        
        response_lines.append(
            f"{i}. **{location}**\n"
            f"   • Price: ₹{price:,.0f} (₹{price_cr:.2f} Cr)\n"
            f"   • Area: {area} sqft | {prop_bhk} BHK\n"
            f"   • Price/sqft: ₹{price_per_sqft:,.0f}\n"
            f"   • Recommendation: {decision}\n"
        )
    
    if count > 8:
        response_lines.append(f"\n...and {count - 8} more properties.")
    
    response_lines.append(f"\n📊 Data sourced from property database for {city_name}.")
    
    return "\n".join(response_lines)


def generate_location_response(locations: list, city: str = None) -> str:
    """
    Generate a direct response for LOCATION intent queries using SQL data only.
    This bypasses the LLM to provide fast, reliable responses for location listings.
    
    FIX: LOCATION queries now return immediately without LLM call.
    
    Args:
        locations: List of location/area names from SQL retrieval
        city: City being queried (optional)
        
    Returns:
        str: Human-readable list of locations (NEVER None)
    """
    # FIX: Always return a string - never return None
    if not locations:
        city_hint = f" in {city.title()}" if city else ""
        return f"No locations found{city_hint} in the current dataset. Try a different city."
    
    city_name = city.title() if city else "the database"
    count = len(locations)
    
    # Build response
    response_lines = [f"📍 Found {count} locations/areas in {city_name}:\n"]
    
    # Show up to 25 locations in columns for readability
    for i, loc in enumerate(locations[:25], 1):
        response_lines.append(f"  {i}. {loc}")
    
    if count > 25:
        response_lines.append(f"\n  ...and {count - 25} more locations.")
    
    response_lines.append(f"\n📊 Data sourced from property database for {city_name}.")
    response_lines.append(f"\n💡 Tip: Ask about properties in a specific location, e.g., 'properties in {locations[0] if locations else 'Andheri'}'")
    
    return "\n".join(response_lines)


def generate_aggregate_response(stats: dict, city: str = None) -> str:
    """
    Generate a direct response for AGGREGATE intent queries using SQL stats only.
    This bypasses the LLM to provide fast, reliable statistical responses.
    
    FIX: AGGREGATE queries can now return immediately without LLM for simple stats.
    
    Args:
        stats: Dictionary of statistics from SQL retrieval
        city: City being queried (optional)
        
    Returns:
        str: Human-readable statistics summary (NEVER None)
    """
    # FIX: Always return a string - never return None
    if not stats or "error" in stats:
        return stats.get("error", "No statistics available for this query.")
    
    city_name = stats.get('city', city or 'all cities').title()
    
    response_lines = [f"📊 Property Statistics for {city_name}:\n"]
    
    response_lines.append(f"  • Total Properties: {stats.get('total_properties', 'N/A'):,}")
    response_lines.append(f"  • Average Price: ₹{stats.get('avg_price', 0):,.0f} (₹{stats.get('avg_price_crore', 0):.2f} Cr)")
    response_lines.append(f"  • Avg Price/sqft: ₹{stats.get('avg_price_per_sqft', 0):,.0f}")
    response_lines.append(f"  • Price Range: ₹{stats.get('min_price', 0):,.0f} - ₹{stats.get('max_price', 0):,.0f}")
    response_lines.append(f"  • Average Area: {stats.get('avg_area_sqft', 0):,.0f} sqft")
    response_lines.append(f"  • Buy Recommendations: {stats.get('buy_recommendations', 0)}")
    response_lines.append(f"  • Rent Recommendations: {stats.get('rent_recommendations', 0)}")
    
    locations = stats.get('locations', [])
    if locations:
        sample_locs = ", ".join(locations[:5])
        response_lines.append(f"\n📍 Sample Locations: {sample_locs}")
    
    response_lines.append(f"\n📊 Data sourced from property database.")
    
    return "\n".join(response_lines)


# ============================================================================
# NEW ENHANCED RESPONSE GENERATORS
# ============================================================================

def generate_advisory_response(property_data: dict, city_stats: dict = None) -> str:
    """
    Generate investment advisory response for a property.
    Answers "Is this a good investment?" type questions with data-grounded analysis.
    
    Args:
        property_data: Property dictionary with metrics
        city_stats: City-level stats for comparison
        
    Returns:
        str: Data-grounded investment assessment (NEVER None)
    """
    try:
        from src.rag.investment_intelligence import (
            calculate_investment_score,
            get_risk_summary,
            generate_metric_explanation
        )
        
        # Calculate investment score
        score_result = calculate_investment_score(property_data, city_stats)
        
        # Get risk summary
        city = property_data.get('city', '')
        risk = get_risk_summary(city) if city else None
        
        # Build response
        lines = []
        location = property_data.get('location', 'This property')
        price_cr = property_data.get('price', 0) / 10000000
        
        lines.append(f"## Investment Analysis: {location}")
        lines.append(f"**Price:** ₹{price_cr:.2f} Cr | **City:** {city.title()}")
        lines.append("")
        
        # Investment Score
        lines.append(f"### Investment Score: {score_result['total_score']}/100 (Grade: {score_result['grade']})")
        lines.append(f"**Assessment:** {score_result['grade_explanation']}")
        lines.append("")
        
        # Key factors
        lines.append("### Key Factors:")
        for explanation in score_result['explanations'][:5]:
            lines.append(f"  • {explanation}")
        lines.append("")
        
        # Risk assessment
        if risk:
            lines.append(f"### Risk Profile: {risk['overall_risk_level'].upper()}")
            lines.append(f"  • Price Volatility: {risk['price_volatility']['risk_level']}")
            lines.append(f"  • Market Liquidity: {risk['liquidity']['risk_level']}")
            lines.append(f"  • Rental Stability: {risk['rental_stability']['risk_level']}")
            lines.append("")
        
        # Bottom line recommendation
        decision = property_data.get('decision', 'N/A')
        wealth_buying = property_data.get('wealth_buying', 0) / 10000000
        wealth_renting = property_data.get('wealth_renting', 0) / 10000000
        
        lines.append("### Bottom Line:")
        if 'buy' in decision.lower():
            lines.append(f"✅ **Recommendation: BUY**")
            lines.append(f"Buying builds ₹{wealth_buying:.2f} Cr wealth vs ₹{wealth_renting:.2f} Cr if renting (20-year projection).")
        else:
            lines.append(f"⚠️ **Recommendation: RENT**")
            lines.append(f"Renting + investing builds ₹{wealth_renting:.2f} Cr vs ₹{wealth_buying:.2f} Cr if buying.")
        
        lines.append("")
        lines.append("📊 *Analysis based on dataset of 895 properties. See methodology in INVESTMENT_METRICS.md*")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Unable to generate investment analysis. Error: {str(e)[:100]}"


def generate_city_profile_response(city: str) -> str:
    """
    Generate comprehensive city investment profile.
    
    Args:
        city: City name
        
    Returns:
        str: City investment profile (NEVER None)
    """
    try:
        from src.rag.investment_intelligence import get_city_investment_profile
        
        profile = get_city_investment_profile(city)
        
        if 'error' in profile:
            return f"I don't have data for {city}. Try cities like Mumbai, Pune, Delhi, Bangalore, or Hyderabad."
        
        metrics = profile['metrics']
        lines = []
        
        lines.append(f"## {profile['city']} Investment Profile")
        lines.append("")
        
        # Market Signal
        signal = profile['market_signal']
        signal_emoji = "🟢" if signal == "Undervalued" else "🟡" if signal == "Fair" else "🔴"
        lines.append(f"### Market Signal: {signal_emoji} {signal}")
        lines.append(f"{profile['market_explanation']}")
        lines.append("")
        
        # Key Metrics
        lines.append("### Key Metrics:")
        lines.append(f"  • Total Properties: {metrics['total_properties']}")
        lines.append(f"  • Avg Price: ₹{metrics['avg_price_crore']:.2f} Cr")
        lines.append(f"  • Avg Price/sqft: ₹{metrics['avg_price_per_sqft']:,.0f}")
        lines.append(f"  • Avg Rental Yield: {metrics['avg_rental_yield']:.2f}%")
        lines.append(f"  • Buy Recommendations: {metrics['buy_recommendation_percent']:.1f}%")
        lines.append("")
        
        # Market Trend
        trend = profile['trend']
        trend_emoji = "🔥" if trend['signal'] == "Overheated" else "❄️" if trend['signal'] == "Cooling" else "⚖️"
        lines.append(f"### Market Trend: {trend_emoji} {trend['signal']}")
        lines.append(f"{trend['explanation']}")
        lines.append("")
        
        # Risk Summary
        risk = profile['risk_summary']
        risk_emoji = "🔴" if risk['overall_level'] == "high" else "🟡" if risk['overall_level'] == "medium" else "🟢"
        lines.append(f"### Risk Level: {risk_emoji} {risk['overall_level'].upper()}")
        lines.append(f"{risk['explanation']}")
        lines.append("")
        
        # Top Opportunities
        if profile['top_opportunities']:
            lines.append("### Top Investment Opportunities:")
            for opp in profile['top_opportunities'][:3]:
                lines.append(f"  • {opp['location']} - Score: {opp['score']}/100 (Grade: {opp['grade']})")
            lines.append("")
        
        lines.append(f"📊 *{profile['data_source']}*")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Unable to generate city profile. Error: {str(e)[:100]}"


def generate_risk_response(city: str = None, location: str = None) -> str:
    """
    Generate risk assessment response.
    
    Args:
        city: City name (optional)
        location: Location/property name (optional)
        
    Returns:
        str: Risk assessment (NEVER None)
    """
    try:
        from src.rag.investment_intelligence import get_risk_summary
        
        risk = get_risk_summary(city, location)
        
        location_str = f"{location}, " if location else ""
        city_str = city.title() if city else "Overall Market"
        
        lines = []
        lines.append(f"## Risk Assessment: {location_str}{city_str}")
        lines.append("")
        
        # Overall
        level = risk['overall_risk_level']
        emoji = "🔴" if level == "high" else "🟡" if level == "medium" else "🟢"
        lines.append(f"### Overall Risk: {emoji} {level.upper()} (Score: {risk['overall_risk_score']}/100)")
        lines.append(f"{risk['overall_explanation']}")
        lines.append("")
        
        # Price Volatility
        pv = risk['price_volatility']
        lines.append(f"### 1. Price Volatility Risk: {pv['risk_level'].upper()}")
        lines.append(f"   {pv['explanation']}")
        if 'cv_percent' in pv:
            lines.append(f"   Coefficient of Variation: {pv['cv_percent']}%")
        lines.append("")
        
        # Liquidity
        lq = risk['liquidity']
        lines.append(f"### 2. Liquidity Risk: {lq['risk_level'].upper()}")
        lines.append(f"   {lq['explanation']}")
        lines.append("")
        
        # Rental Stability
        rs = risk['rental_stability']
        lines.append(f"### 3. Rental Stability Risk: {rs['risk_level'].upper()}")
        lines.append(f"   {rs['explanation']}")
        lines.append("")
        
        lines.append("📊 *Risk indicators are rule-based heuristics derived from dataset patterns.*")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Unable to generate risk assessment. Error: {str(e)[:100]}"


def generate_scenario_response(property_price: float, monthly_rent: float, scenario: str = None) -> str:
    """
    Generate scenario analysis response.
    
    Args:
        property_price: Property price
        monthly_rent: Monthly rent
        scenario: Optional specific scenario (conservative/moderate/aggressive)
        
    Returns:
        str: Scenario analysis (NEVER None)
    """
    try:
        from src.rag.investment_intelligence import run_all_scenarios, SCENARIOS
        
        results = run_all_scenarios(property_price, monthly_rent)
        
        price_cr = property_price / 10000000
        
        lines = []
        lines.append(f"## Scenario Analysis")
        lines.append(f"**Property Price:** ₹{price_cr:.2f} Cr | **Monthly Rent:** ₹{monthly_rent:,.0f}")
        lines.append("")
        
        # Consensus
        lines.append(f"### Consensus: {results['consensus']}")
        lines.append(f"{results['explanation']}")
        lines.append("")
        
        # Each scenario
        for scenario_key, data in results['scenarios'].items():
            wealth_buy = data['results']['wealth_buying'] / 10000000
            wealth_rent = data['results']['wealth_renting'] / 10000000
            rec = data['results']['recommendation']
            emoji = "✅" if rec == "Buy" else "⚠️"
            
            lines.append(f"### {data['scenario']} Scenario")
            lines.append(f"   *{data['description']}*")
            lines.append(f"   • Wealth if Buying: ₹{wealth_buy:.2f} Cr")
            lines.append(f"   • Wealth if Renting: ₹{wealth_rent:.2f} Cr")
            lines.append(f"   • Recommendation: {emoji} {rec}")
            lines.append("")
        
        # Assumptions
        lines.append("### Key Assumptions Vary By Scenario:")
        moderate = SCENARIOS['moderate']
        lines.append(f"   • Appreciation: 3% (cons) / {moderate['appreciation_rate']}% (mod) / 8% (agg)")
        lines.append(f"   • Investment Return: 8% / {moderate['investment_return']}% / 12%")
        lines.append(f"   • Loan Rate: 9.5% / {moderate['loan_rate']}% / 7.5%")
        lines.append("")
        
        lines.append("📊 *See INVESTMENT_METRICS.md for full methodology.*")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Unable to generate scenario analysis. Error: {str(e)[:100]}"

