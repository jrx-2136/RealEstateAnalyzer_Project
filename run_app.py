"""
Real Estate Investment Analyzer - Flask Application
Clean, production-ready Flask app with service layer architecture
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import service layer
from services.analysis import RealEstateAnalyzer, load_properties_data

# Import RAG components
try:
    from src.rag.rag_engine import (
        generate_rag_response, generate_no_data_response,
        generate_filter_response, generate_location_response, generate_aggregate_response
    )
    from src.rag.vector_store import build_or_load_vector_store, similarity_search_with_score
    from src.rag.property_explanations import load_property_explanations
    from src.rag.intent_classifier import (
        classify_intent, extract_cities_from_query, extract_bhk_from_query,
        detect_specific_property_query, extract_scenario_from_query  # PRECISION FIX + SCENARIOS
    )
    from src.rag.sql_retriever import (
        filter_properties, get_city_stats, get_locations_in_city,
        get_available_cities, get_top_properties, get_comparison_stats,
        format_properties_for_context, format_city_stats_for_context,
        get_all_property_names, find_property_by_name, format_single_property  # PRECISION FIX
    )
    RAG_AVAILABLE = True
    print("[OK] RAG engine loaded successfully")
except Exception as e:
    RAG_AVAILABLE = False
    generate_rag_response = None
    generate_no_data_response = None
    generate_filter_response = None
    generate_location_response = None
    generate_aggregate_response = None
    build_or_load_vector_store = None
    similarity_search_with_score = None
    load_property_explanations = None
    classify_intent = None
    extract_cities_from_query = None
    extract_bhk_from_query = None
    detect_specific_property_query = None
    extract_scenario_from_query = None
    filter_properties = None
    get_city_stats = None
    get_locations_in_city = None
    get_available_cities = None
    get_top_properties = None
    get_comparison_stats = None
    format_properties_for_context = None
    format_city_stats_for_context = None
    get_all_property_names = None
    find_property_by_name = None
    format_single_property = None
    print(f"[WARNING] RAG engine not available: {e}")
    print("          Chat requires RAG setup (run: python run_rag.py)")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production

# Initialize analyzer service
analyzer = RealEstateAnalyzer()

# Initialize RAG vector database
vector_db = None
if RAG_AVAILABLE:
    try:
        import os
        vector_dir = "data/vectorstore"
        # Load property explanations to build/load embeddings
        explanations = load_property_explanations()
        print(f"[INFO] Loaded {len(explanations)} property explanations")
        
        # Build or load the vector store with explanations
        vector_db = build_or_load_vector_store(explanations)
        print("[OK] Vector database loaded successfully")
    except Exception as e:
        print(f"[WARNING] Vector database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        RAG_AVAILABLE = False


@app.route('/')
def landing():
    """
    Landing page route - entry point of the application
    Displays hero section, search, and feature highlights
    """
    try:
        # Load properties data to get cities for search dropdown
        df = load_properties_data()
        cities = sorted(df['city'].unique().tolist()) if not df.empty and 'city' in df.columns else []
        return render_template('landing.html', cities=cities)
    except Exception as e:
        print(f"Landing page error: {e}")
        return render_template('landing.html', cities=[])


@app.route('/dashboard')
def dashboard():
    """
    Main dashboard route
    Displays KPIs, charts, and property listings with optional filters
    """
    try:
        # Load properties data
        df = load_properties_data()
        
        # Get filter parameters from query string
        filters = {
            'city': request.args.get('city', 'all'),
            'min_budget': request.args.get('min_budget', type=float),
            'max_budget': request.args.get('max_budget', type=float),
            'bhk': request.args.get('bhk', type=int),
            'decision': request.args.get('decision', 'all')
        }
        
        # Apply filters if data exists
        if not df.empty:
            df_filtered = analyzer.filter_properties(df, filters)
        else:
            df_filtered = df
        
        # Calculate summary metrics
        metrics = analyzer.get_summary_metrics(df_filtered)
        
        # Get available cities for filter dropdown
        cities = sorted(df['city'].unique().tolist()) if not df.empty and 'city' in df.columns else []
        
        # Prepare chart data
        chart_data = analyzer.get_chart_data(df_filtered)
        
        # Convert filtered dataframe to list of dicts for template
        properties = df_filtered.to_dict('records') if not df_filtered.empty else []
        
        return render_template(
            'dashboard.html',
            metrics=metrics,
            properties=properties,
            cities=cities,
            chart_data=chart_data,
            active_filters=filters
        )
        
    except Exception as e:
        print(f"Dashboard error: {e}")
        # Return empty dashboard on error
        return render_template(
            'dashboard.html',
            metrics={
                'total_properties': 0,
                'buy_recommendations': 0,
                'rent_recommendations': 0,
                'avg_price': 0,
                'avg_roi': 0,
                'avg_emi': 0
            },
            properties=[],
            cities=[],
            chart_data={
                'roi_labels': [],
                'roi_values': [],
                'buy_vs_rent_labels': ['Buy', 'Rent'],
                'buy_vs_rent_values': [0, 0],
                'price_distribution_labels': [],
                'price_distribution_values': [],
                'city_properties_labels': [],
                'city_properties_values': [],
                'location_price_labels': [],
                'location_price_values': [],
                'price_per_sqft_labels': [],
                'price_per_sqft_values': []
            },
            active_filters={}
        )


@app.route('/api/calculate', methods=['POST'])
def calculate_analysis():
    """
    API endpoint for custom analysis calculations
    Accepts property details and returns buy vs rent analysis
    """
    try:
        data = request.json
        
        # Extract parameters
        property_price = float(data.get('property_price', 0))
        monthly_rent = float(data.get('monthly_rent', 0))
        
        # Optional custom parameters
        custom_params = {
            'down_payment_percent': float(data.get('down_payment_percent', 20)),
            'loan_rate': float(data.get('loan_rate', 8.5)),
            'loan_tenure_years': int(data.get('loan_tenure_years', 20)),
            'appreciation_rate': float(data.get('appreciation_rate', 5)),
            'rent_escalation': float(data.get('rent_escalation', 5)),
            'investment_return_rate': float(data.get('investment_return_rate', 10)),
            'monthly_savings': float(data.get('monthly_savings', 15000))
        }
        
        # Perform analysis
        result = analyzer.buy_vs_rent_analysis(
            property_price=property_price,
            monthly_rent=monthly_rent,
            params=custom_params
        )
        
        # Calculate ROI
        roi_data = analyzer.calculate_roi(
            property_price=property_price,
            rent=monthly_rent,
            appreciation_rate=custom_params['appreciation_rate']
        )
        
        return jsonify({
            'success': True,
            'analysis': result,
            'roi': roi_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/properties', methods=['GET'])
def get_properties():
    """
    API endpoint to fetch filtered properties
    Supports pagination and filtering
    """
    try:
        df = load_properties_data()
        
        if df.empty:
            return jsonify({
                'success': True,
                'properties': [],
                'total': 0,
                'page': 1,
                'pages': 0
            })
        
        # Get filters
        filters = {
            'city': request.args.get('city', 'all'),
            'min_budget': request.args.get('min_budget', type=float),
            'max_budget': request.args.get('max_budget', type=float),
            'bhk': request.args.get('bhk', type=int),
            'decision': request.args.get('decision', 'all')
        }
        
        # Apply filters
        df_filtered = analyzer.filter_properties(df, filters)
        
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        total = len(df_filtered)
        start = (page - 1) * per_page
        end = start + per_page
        
        properties = df_filtered.iloc[start:end].to_dict('records')
        
        return jsonify({
            'success': True,
            'properties': properties,
            'total': total,
            'page': page,
            'pages': (total + per_page - 1) // per_page
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    API endpoint to fetch summary metrics
    Can be used for dynamic dashboard updates
    """
    try:
        df = load_properties_data()
        metrics = analyzer.get_summary_metrics(df)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


# ============================================================================
# PROPERTIES BROWSE PAGE - Magicbricks/99acres Style Property Browser
# ============================================================================

@app.route('/properties')
def properties_browse():
    """
    Standalone Properties browse page - Magicbricks/99acres style
    Independent of dashboard, uses structured data only (no RAG)
    """
    try:
        df = load_properties_data()
        
        # Get unique cities for filter dropdown
        cities = sorted(df['city'].unique().tolist()) if not df.empty and 'city' in df.columns else []
        
        return render_template(
            'properties.html',
            cities=cities
        )
        
    except Exception as e:
        print(f"Properties page error: {e}")
        return render_template(
            'properties.html',
            cities=[]
        )


@app.route('/api/properties/browse', methods=['GET'])
def api_properties_browse():
    """
    API endpoint for Properties browse page
    Returns all properties with buy_advantage calculation for frontend filtering/sorting
    """
    try:
        df = load_properties_data()
        
        if df.empty:
            return jsonify({
                'success': True,
                'properties': [],
                'total': 0
            })
        
        # Calculate buy_advantage for each property
        # buy_advantage = (wealth_buying - wealth_renting) / wealth_renting * 100
        df_copy = df.copy()
        df_copy['buy_advantage'] = df_copy.apply(
            lambda row: ((row['wealth_buying'] - row['wealth_renting']) / row['wealth_renting'] * 100)
            if row['wealth_renting'] > 0 else 0,
            axis=1
        )
        
        # Convert to list of dicts
        properties = df_copy.to_dict('records')
        
        return jsonify({
            'success': True,
            'properties': properties,
            'total': len(properties)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/properties/filter', methods=['POST'])
def api_properties_filter():
    """
    API endpoint for filtered property search
    Accepts filter criteria and returns matching properties
    """
    try:
        df = load_properties_data()
        
        if df.empty:
            return jsonify({
                'success': True,
                'properties': [],
                'total': 0
            })
        
        # Get filter criteria from request
        data = request.json or {}
        
        # Calculate buy_advantage
        df['buy_advantage'] = df.apply(
            lambda row: ((row['wealth_buying'] - row['wealth_renting']) / row['wealth_renting'] * 100)
            if row['wealth_renting'] > 0 else 0,
            axis=1
        )
        
        # Apply filters
        filtered_df = df.copy()
        
        # City filter
        if data.get('city') and data['city'] != 'all':
            filtered_df = filtered_df[filtered_df['city'] == data['city']]
        
        # Location search (partial match)
        if data.get('location'):
            search_term = data['location'].lower()
            filtered_df = filtered_df[filtered_df['location'].str.lower().str.contains(search_term, na=False)]
        
        # Price range
        if data.get('minPrice'):
            filtered_df = filtered_df[filtered_df['price'] >= float(data['minPrice'])]
        if data.get('maxPrice'):
            filtered_df = filtered_df[filtered_df['price'] <= float(data['maxPrice'])]
        
        # BHK filter
        if data.get('bhk') and data['bhk'] != 'all':
            filtered_df = filtered_df[filtered_df['bhk'] == int(data['bhk'])]
        
        # Area range
        if data.get('minArea'):
            filtered_df = filtered_df[filtered_df['area_sqft'] >= float(data['minArea'])]
        if data.get('maxArea'):
            filtered_df = filtered_df[filtered_df['area_sqft'] <= float(data['maxArea'])]
        
        # Decision filter
        if data.get('decision') and data['decision'] != 'all':
            if data['decision'] == 'buy':
                filtered_df = filtered_df[filtered_df['decision'].str.lower().str.contains('buy', na=False)]
            elif data['decision'] == 'rent':
                filtered_df = filtered_df[filtered_df['decision'].str.lower().str.contains('rent', na=False)]
        
        # Sorting
        sort_by = data.get('sortBy', 'price_asc')
        if sort_by == 'price_asc':
            filtered_df = filtered_df.sort_values('price', ascending=True)
        elif sort_by == 'price_desc':
            filtered_df = filtered_df.sort_values('price', ascending=False)
        elif sort_by == 'price_sqft_asc':
            filtered_df = filtered_df.sort_values('price_per_sqft', ascending=True)
        elif sort_by == 'price_sqft_desc':
            filtered_df = filtered_df.sort_values('price_per_sqft', ascending=False)
        elif sort_by == 'area_desc':
            filtered_df = filtered_df.sort_values('area_sqft', ascending=False)
        elif sort_by == 'area_asc':
            filtered_df = filtered_df.sort_values('area_sqft', ascending=True)
        elif sort_by == 'buy_advantage':
            filtered_df = filtered_df.sort_values('buy_advantage', ascending=False)
        
        # Convert to list
        properties = filtered_df.to_dict('records')
        
        return jsonify({
            'success': True,
            'properties': properties,
            'total': len(properties)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    RAG-powered chat endpoint - Data-grounded responses only.
    
    Flow:
    1. Classify query intent
    2. Extract entities (cities, BHK)
    3. Retrieve relevant data via SQL and/or vector search
    4. Generate response strictly from retrieved data
    """
    try:
        data = request.json
        user_query = data.get('message', '').strip()
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        # Check if RAG is available
        if not RAG_AVAILABLE or not vector_db:
            return jsonify({
                'success': False,
                'error': 'AI chat is currently unavailable. Please ensure the vector database is set up by running: python run_rag.py'
            }), 503
        
        try:
            # ================================================================
            # PRECISION FIX: Check for specific property query FIRST
            # This must run before regular intent classification to ensure
            # named property queries get exact/fuzzy matching, not broad search
            # ================================================================
            property_names = get_all_property_names()
            is_specific, extracted_name = detect_specific_property_query(user_query, property_names)
            
            if is_specific and extracted_name:
                print(f"🎯 Specific Property Query detected: '{extracted_name}'")
                
                # Use high-precision lookup
                matched_prop, match_type, similar_props = find_property_by_name(extracted_name, threshold=0.5)
                
                if matched_prop:
                    # Found a match - return ONLY this property
                    response = format_single_property(matched_prop)
                    
                    # Add clarification if there were similar properties
                    if similar_props and match_type == 'fuzzy':
                        response += f"\n\n💡 Similar properties: {', '.join(similar_props)}"
                    
                    return jsonify({
                        'success': True,
                        'response': response,
                        'source': f'Exact Match ({match_type})'
                    })
                
                elif similar_props:
                    # No exact match but found similar - ask for clarification
                    response = f"I couldn't find an exact match for \"{extracted_name}\".\n\n"
                    response += "Did you mean one of these?\n"
                    for i, prop_name in enumerate(similar_props, 1):
                        response += f"  {i}. {prop_name}\n"
                    response += "\nPlease try again with the exact property name."
                    
                    return jsonify({
                        'success': True,
                        'response': response,
                        'source': 'Clarification Needed'
                    })
                
                else:
                    # No match found at all
                    response = f"I couldn't find a property named \"{extracted_name}\" in the database.\n\n"
                    response += "Please check the spelling or try searching for properties in a specific city, e.g., 'properties in Mumbai'."
                    
                    return jsonify({
                        'success': True,
                        'response': response,
                        'source': 'No Match'
                    })
            
            # ================================================================
            # Standard intent classification (for non-specific queries)
            # ================================================================
            intent = classify_intent(user_query)
            available_cities = get_available_cities()
            
            # Step 2: Extract entities from query
            detected_cities = extract_cities_from_query(user_query, available_cities)
            detected_bhk = extract_bhk_from_query(user_query)
            
            print(f"🎯 Query: '{user_query}'")
            print(f"   Intent: {intent} | Cities: {detected_cities} | BHK: {detected_bhk}")
            
            # Step 3: Build context based on intent
            # FIX: FILTER, LOCATION, and AGGREGATE queries now return directly without LLM
            context_parts = []
            retrieval_source = []
            
            # 3a. For AGGREGATE queries, return SQL stats directly (NO LLM)
            # FIX: Skip LLM for simple aggregate queries to prevent timeouts
            if intent == "AGGREGATE":
                city = None
                stats = None
                if detected_cities:
                    city = detected_cities[0]
                    stats = get_city_stats(city)
                else:
                    # Try to extract city from query text directly
                    q_lower = user_query.lower()
                    for c in available_cities:
                        if c in q_lower:
                            city = c
                            stats = get_city_stats(city)
                            break
                
                if stats and "error" not in stats:
                    # FIX: Return directly without LLM call
                    response = generate_aggregate_response(stats, city)
                    return jsonify({
                        'success': True,
                        'response': response,
                        'source': f'SQL Stats ({city.title() if city else "all cities"})'
                    })
                # If no stats found, fall through to vector search
            
            # 3b. For LOCATION queries, return locations list directly (NO LLM)
            # FIX: Skip LLM for location queries to prevent timeouts
            elif intent == "LOCATION":
                city = None
                locations = None
                if detected_cities:
                    city = detected_cities[0]
                    locations = get_locations_in_city(city)
                else:
                    # Try to extract city from query text directly
                    q_lower = user_query.lower()
                    for c in available_cities:
                        if c in q_lower:
                            city = c
                            locations = get_locations_in_city(city)
                            break
                
                if locations:
                    # FIX: Return directly without LLM call
                    response = generate_location_response(locations, city)
                    return jsonify({
                        'success': True,
                        'response': response,
                        'source': f'SQL Locations ({city.title() if city else "database"})'
                    })
                # If no locations found, fall through to vector search
            
            # 3c. For COMPARE queries, get comparison stats
            elif intent == "COMPARE" and len(detected_cities) >= 2:
                comparison = get_comparison_stats(detected_cities)
                for city, stats in comparison.items():
                    context_parts.append(format_city_stats_for_context(stats))
                    retrieval_source.append(f"Stats for {city}")
            
            # 3d. For FILTER queries, return SQL data directly (no LLM needed)
            # FIX: FILTER intent now returns immediately with formatted SQL results
            elif intent == "FILTER":
                city = detected_cities[0] if detected_cities else None
                properties = filter_properties(city=city, bhk=detected_bhk, limit=8)
                # FIX: Use pre-imported generate_filter_response (no LLM call)
                response = generate_filter_response(properties, city, detected_bhk)
                return jsonify({
                    'success': True,
                    'response': response,
                    'source': f'SQL Filter ({len(properties) if properties else 0} results)'
                })
            
            # ================================================================
            # NEW ENHANCED INTENT HANDLERS
            # ================================================================
            
            # 3e. For ADVISORY queries, provide investment analysis
            elif intent == "ADVISORY":
                from src.rag.rag_engine import generate_advisory_response
                city = detected_cities[0] if detected_cities else None
                
                # Try to find a specific property first
                properties = filter_properties(city=city, bhk=detected_bhk, limit=1)
                if properties:
                    city_stats = get_city_stats(city) if city else {}
                    response = generate_advisory_response(properties[0], city_stats)
                    return jsonify({
                        'success': True,
                        'response': response,
                        'source': 'Investment Analysis'
                    })
                # Fall through to vector search if no property found
            
            # 3f. For CITY_PROFILE queries, return city investment profile
            elif intent == "CITY_PROFILE":
                from src.rag.rag_engine import generate_city_profile_response
                city = detected_cities[0] if detected_cities else None
                
                if city:
                    response = generate_city_profile_response(city)
                    return jsonify({
                        'success': True,
                        'response': response,
                        'source': f'City Profile ({city.title()})'
                    })
                # If no city detected, try to extract from query
                q_lower = user_query.lower()
                for c in available_cities:
                    if c in q_lower:
                        response = generate_city_profile_response(c)
                        return jsonify({
                            'success': True,
                            'response': response,
                            'source': f'City Profile ({c.title()})'
                        })
            
            # 3g. For RISK queries, return risk assessment
            elif intent == "RISK":
                from src.rag.rag_engine import generate_risk_response
                city = detected_cities[0] if detected_cities else None
                response = generate_risk_response(city)
                return jsonify({
                    'success': True,
                    'response': response,
                    'source': f'Risk Assessment ({city.title() if city else "Market"})'
                })
            
            # 3h. For SCENARIO queries, return scenario analysis
            elif intent == "SCENARIO":
                from src.rag.rag_engine import generate_scenario_response
                from src.rag.intent_classifier import extract_scenario_from_query
                
                # Try to find property price/rent from query context
                city = detected_cities[0] if detected_cities else None
                properties = filter_properties(city=city, bhk=detected_bhk, limit=1)
                
                if properties:
                    prop = properties[0]
                    price = prop.get('price', 10000000)
                    rent = prop.get('estimated_rent', price * 0.003)
                    scenario = extract_scenario_from_query(user_query)
                    response = generate_scenario_response(price, rent, scenario)
                    return jsonify({
                        'success': True,
                        'response': response,
                        'source': 'Scenario Analysis'
                    })
                else:
                    # Use default values for general scenario query
                    response = generate_scenario_response(10000000, 30000)  # 1 Cr, 30K rent
                    return jsonify({
                        'success': True,
                        'response': response,
                        'source': 'Scenario Analysis (default values)'
                    })
            
            # 3i. For RECOMMEND queries, get filtered properties for LLM analysis
            # NOTE: RECOMMEND still uses LLM for buy/rent advice
            elif intent == "RECOMMEND":
                city = detected_cities[0] if detected_cities else None
                properties = filter_properties(city=city, bhk=detected_bhk, limit=8)
                if properties:
                    context_parts.append(format_properties_for_context(properties))
                    retrieval_source.append(f"Properties ({len(properties)} results)")
            
            # Step 4: Vector search for semantic context (for intents that need LLM)
            docs_with_scores = []
            if vector_db and (intent in ["RECOMMEND", "EDUCATIONAL", "COMPARE"] or not context_parts):
                try:
                    docs_with_scores = similarity_search_with_score(vector_db, user_query, k=5)
                except Exception as e:
                    print(f"Vector search error: {e}")
            
            if docs_with_scores:
                vector_context = [doc.page_content for doc, score in docs_with_scores]
                context_parts.extend(vector_context)
                retrieval_source.append(f"Vector search ({len(docs_with_scores)} docs)")
            
            # Step 5: Handle empty results gracefully
            if not context_parts:
                response = generate_no_data_response(user_query, available_cities)
                return jsonify({
                    'success': True,
                    'response': response,
                    'source': 'no_data_found'
                })
            
            # Step 6: Generate data-grounded response
            response = generate_rag_response(context_parts, user_query, intent)
            
            # FIX: Ensure we always have a valid response string - never return None
            if not response or not isinstance(response, str) or len(response.strip()) == 0:
                response = "I found some data but couldn't generate a summary. Please try rephrasing your question or ask about a specific city like Mumbai, Pune, or Delhi."
            
            source_str = ", ".join(retrieval_source) if retrieval_source else "RAG"
            return jsonify({
                'success': True,
                'response': response,
                'source': f'RAG ({source_str})'
            })
            
        except Exception as e:
            print(f"RAG error: {e}")
            traceback.print_exc()
            # Return a graceful error with suggestions
            return jsonify({
                'success': True,
                'response': f"I encountered an issue processing your query. Please try rephrasing or ask about:\n• Property prices in a specific city (Mumbai, Pune, Delhi, etc.)\n• Buy vs rent recommendations\n• Locations in a city\n\nError: {str(e)}",
                'source': 'error_fallback'
            })
        
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Real Estate Analyzer',
        'rag_available': RAG_AVAILABLE and vector_db is not None
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Development server configuration
    # In production, use gunicorn or similar WSGI server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Disabled debug to avoid reload issues
    )
