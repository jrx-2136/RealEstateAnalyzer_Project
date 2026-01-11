"""
Real Estate Investment Analysis Service
Handles all financial calculations, ROI analysis, and buy vs rent comparisons
"""

import pandas as pd
from typing import Dict, List, Any, Optional


class RealEstateAnalyzer:
    """
    Core analytics engine for real estate investment analysis
    Integrates with existing Parameters modules for calculations
    """
    
    def __init__(self):
        """Initialize analyzer with default parameters"""
        self.default_params = {
            'down_payment_percent': 20,
            'loan_rate': 8.5,
            'loan_tenure_years': 20,
            'property_tax_rate': 0.1,
            'maintenance_rate': 0.5,
            'appreciation_rate': 5,
            'rent_escalation': 5,
            'investment_return_rate': 10,
            'monthly_savings': 15000
        }
    
    def calculate_emi(self, principal: float, rate_percent: float, tenure_years: int) -> float:
        """
        Calculate monthly EMI (Equated Monthly Installment)
        
        Args:
            principal: Loan amount
            rate_percent: Annual interest rate (%)
            tenure_years: Loan tenure in years
        
        Returns:
            Monthly EMI amount
        """
        if principal <= 0 or rate_percent <= 0 or tenure_years <= 0:
            return 0
        
        monthly_rate = rate_percent / (12 * 100)
        num_months = tenure_years * 12
        
        if monthly_rate == 0:
            return principal / num_months
        
        emi = principal * monthly_rate * (1 + monthly_rate) ** num_months / \
              ((1 + monthly_rate) ** num_months - 1)
        
        return round(emi, 2)
    
    def calculate_roi(self, property_price: float, rent: float, appreciation_rate: float, 
                     years: int = 10) -> Dict[str, float]:
        """
        Calculate Return on Investment for a property
        
        Args:
            property_price: Purchase price of property
            rent: Monthly rental income
            appreciation_rate: Annual appreciation rate (%)
            years: Investment period
        
        Returns:
            Dictionary with ROI metrics
        """
        # Calculate future property value
        future_value = property_price * (1 + appreciation_rate / 100) ** years
        
        # Calculate total rental income (with 5% escalation)
        total_rent = 0
        current_rent = rent
        for year in range(years):
            total_rent += current_rent * 12
            current_rent *= (1 + 0.05)  # 5% annual escalation
        
        # Calculate total returns
        total_return = (future_value - property_price) + total_rent
        roi_percent = (total_return / property_price) * 100
        
        return {
            'initial_investment': property_price,
            'future_value': round(future_value, 2),
            'rental_income': round(total_rent, 2),
            'total_return': round(total_return, 2),
            'roi_percent': round(roi_percent, 2),
            'annual_roi': round(roi_percent / years, 2)
        }
    
    def buy_vs_rent_analysis(self, property_price: float, monthly_rent: float, 
                            params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Compare buying vs renting over 20 years
        
        METHODOLOGY:
        -----------
        - BUYING: Calculate total cost (down payment + EMIs), then project property value
          with appreciation. Wealth = Future Value - Total Cost Paid.
        
        - RENTING: Invest the down payment + monthly savings (EMI - Rent) in equity.
          Wealth = Investment Corpus after 20 years.
        
        - DECISION: If Wealth(Buying) > Wealth(Renting), recommend BUY.
        
        ASSUMPTIONS (see INVESTMENT_METRICS.md for details):
        - Down payment: 20% of property price
        - Loan rate: 8.5% p.a. (average home loan rate in India)
        - Appreciation: 5% p.a. (conservative estimate)
        - Investment return: 10% p.a. (index fund benchmark)
        - Rent escalation: 5% p.a.
        
        Args:
            property_price: Property purchase price
            monthly_rent: Monthly rent amount
            params: Optional custom parameters
        
        Returns:
            Dictionary with comparison metrics and recommendation
        """
        # Use default params if not provided
        p = params or self.default_params
        
        # ============================
        # BUYING SCENARIO CALCULATIONS
        # ============================
        
        # Step 1: Calculate down payment and loan amount
        down_payment = property_price * (p['down_payment_percent'] / 100)
        loan_amount = property_price - down_payment
        
        # Step 2: Calculate EMI using standard formula
        monthly_emi = self.calculate_emi(loan_amount, p['loan_rate'], p['loan_tenure_years'])
        
        # Step 3: Total cost = Down payment + All EMI payments over tenure
        total_emi_paid = monthly_emi * p['loan_tenure_years'] * 12
        total_buying_cost = down_payment + total_emi_paid
        
        # Step 4: Project property value with compound appreciation
        years = 20
        future_property_value = property_price * (1 + p['appreciation_rate'] / 100) ** years
        
        # Step 5: Net wealth from buying = Asset value - Cost paid
        # NOTE: Does not include property taxes, maintenance, or stamp duty (simplified model)
        wealth_buying = future_property_value - total_buying_cost
        
        # ============================
        # RENTING SCENARIO CALCULATIONS
        # ============================
        
        # Step 1: Calculate total rent over 20 years (with annual escalation)
        total_rent_paid = 0
        current_rent = monthly_rent
        for year in range(years):
            total_rent_paid += current_rent * 12
            current_rent *= (1 + p['rent_escalation'] / 100)
        
        # Step 2: Invest down payment in equity/mutual funds
        # Future Value = Principal * (1 + rate)^years
        investment_corpus = down_payment * (1 + p['investment_return_rate'] / 100) ** years
        
        # Step 3: Monthly savings = (EMI - Rent) invested as SIP
        # If renting is cheaper, the savings are invested monthly
        monthly_savings_investment = 0
        if monthly_emi > monthly_rent:
            savings_per_month = monthly_emi - monthly_rent
            # SIP Future Value formula: P * [(1+r)^n - 1] / r * (1+r)
            monthly_return_rate = p['investment_return_rate'] / (12 * 100)
            months = years * 12
            if monthly_return_rate > 0:
                monthly_savings_investment = savings_per_month * \
                    (((1 + monthly_return_rate) ** months - 1) / monthly_return_rate) * \
                    (1 + monthly_return_rate)
        
        # Step 4: Total wealth from renting = Investment corpus + SIP corpus
        wealth_renting = investment_corpus + monthly_savings_investment
        
        # ============================
        # COMPARISON & RECOMMENDATION
        # ============================
        wealth_difference = wealth_buying - wealth_renting
        
        # Decision logic: Higher final wealth wins
        recommendation = "Buy" if wealth_buying > wealth_renting else "Rent"
        
        # Calculate break-even point (simplified)
        break_even_years = 0
        if wealth_difference != 0:
            break_even_years = min(years, max(5, int(abs(wealth_difference) / (property_price * 0.05))))
        
        return {
            # Buying metrics
            'buy_down_payment': round(down_payment, 2),
            'buy_loan_amount': round(loan_amount, 2),
            'buy_monthly_emi': round(monthly_emi, 2),
            'buy_total_cost': round(total_buying_cost, 2),
            'buy_future_value': round(future_property_value, 2),
            'buy_wealth': round(wealth_buying, 2),
            
            # Renting metrics
            'rent_monthly': monthly_rent,
            'rent_total_paid': round(total_rent_paid, 2),
            'rent_investment_corpus': round(investment_corpus, 2),
            'rent_savings_investment': round(monthly_savings_investment, 2),
            'rent_wealth': round(wealth_renting, 2),
            
            # Comparison
            'wealth_difference': round(wealth_difference, 2),
            'recommendation': recommendation,
            'break_even_years': break_even_years,
            'monthly_cash_flow_buy': round(-monthly_emi, 2),
            'monthly_cash_flow_rent': round(-monthly_rent, 2)
        }
    
    def get_summary_metrics(self, properties_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate summary KPIs for dashboard
        
        Args:
            properties_df: DataFrame with analyzed properties
        
        Returns:
            Dictionary with KPI metrics
        """
        # Use sample data if no dataframe provided
        if properties_df is None or properties_df.empty:
            return {
                'total_properties': 0,
                'buy_recommendations': 0,
                'rent_recommendations': 0,
                'avg_price': 0,
                'avg_roi': 0,
                'avg_emi': 0,
                'avg_rental_yield': 0
            }
        
        total = len(properties_df)
        
        # Count buy vs rent recommendations (handle full text decisions like "RENTING is financially better")
        buy_count = 0
        rent_count = 0
        if 'decision' in properties_df.columns:
            decision_lower = properties_df['decision'].str.lower()
            buy_count = len(properties_df[decision_lower.str.contains('buy', na=False)])
            rent_count = len(properties_df[decision_lower.str.contains('rent', na=False)])
        
        avg_price = properties_df['price'].mean() if 'price' in properties_df.columns else 0
        avg_emi = properties_df.get('monthly_emi', pd.Series([0])).mean()
        
        # Calculate average ROI if available
        avg_roi = 0
        if 'roi_percent' in properties_df.columns:
            avg_roi = properties_df['roi_percent'].mean()
        
        # Calculate rental yield
        avg_rental_yield = 0
        if 'price' in properties_df.columns and 'estimated_rent' in properties_df.columns:
            properties_df['rental_yield'] = (properties_df['estimated_rent'] * 12 / properties_df['price']) * 100
            avg_rental_yield = properties_df['rental_yield'].mean()
        
        return {
            'total_properties': int(total),
            'buy_recommendations': int(buy_count),
            'rent_recommendations': int(rent_count),
            'avg_price': round(avg_price, 2),
            'avg_roi': round(avg_roi, 2),
            'avg_emi': round(avg_emi, 2),
            'avg_rental_yield': round(avg_rental_yield, 2)
        }
    
    def filter_properties(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter properties based on user criteria
        
        Args:
            df: Properties DataFrame
            filters: Dictionary with filter criteria
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Apply city filter
        if filters.get('city') and filters['city'] != 'all':
            filtered_df = filtered_df[filtered_df['city'].str.lower() == filters['city'].lower()]
        
        # Apply budget filter
        if filters.get('min_budget'):
            filtered_df = filtered_df[filtered_df['price'] >= filters['min_budget']]
        if filters.get('max_budget'):
            filtered_df = filtered_df[filtered_df['price'] <= filters['max_budget']]
        
        # Apply BHK filter
        if filters.get('bhk'):
            filtered_df = filtered_df[filtered_df['bhk'] == filters['bhk']]
        
        # Apply decision filter (handles both "Buy"/"Rent" and full text like "RENTING is financially better")
        if filters.get('decision') and filters['decision'] != 'all':
            decision_filter = filters['decision'].lower()
            if decision_filter == 'buy':
                filtered_df = filtered_df[filtered_df['decision'].str.lower().str.contains('buy', na=False)]
            elif decision_filter == 'rent':
                filtered_df = filtered_df[filtered_df['decision'].str.lower().str.contains('rent', na=False)]
            else:
                filtered_df = filtered_df[filtered_df['decision'].str.lower().str.contains(decision_filter, na=False)]
        
        return filtered_df
    
    def get_chart_data(self, properties_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare data for Chart.js visualizations
        
        Args:
            properties_df: Properties DataFrame
        
        Returns:
            Dictionary with chart data
        """
        if properties_df.empty:
            return {
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
            }
        
        # ROI comparison (top 10 properties)
        # Show properties with best "Buy vs Rent" advantage (wealth_buying - wealth_renting)
        # This is the key investment metric showing where buying makes most sense
        if 'roi_percent' in properties_df.columns:
            top_roi = properties_df.nlargest(10, 'roi_percent')
            roi_labels = top_roi['location'].tolist() if 'location' in top_roi.columns else [f"Property {i+1}" for i in range(len(top_roi))]
            roi_values = top_roi['roi_percent'].tolist()
        elif 'wealth_buying' in properties_df.columns and 'wealth_renting' in properties_df.columns:
            # Calculate Buy Advantage = (wealth_buying - wealth_renting) / wealth_renting * 100
            # Positive = buying is better, Negative = renting is better
            # This shows the relative advantage/disadvantage of buying
            df_copy = properties_df.copy()
            df_copy['buy_advantage'] = df_copy.apply(
                lambda row: ((row['wealth_buying'] - row['wealth_renting']) / row['wealth_renting'] * 100)
                if row['wealth_renting'] > 0 else 0,
                axis=1
            )
            # Get top 10 properties where buying has the best relative advantage
            # Sort descending so highest advantage (or smallest disadvantage) comes first
            top_roi = df_copy.nlargest(10, 'buy_advantage')
            
            roi_labels = top_roi['location'].tolist() if 'location' in top_roi.columns else [f"Property {i+1}" for i in range(len(top_roi))]
            # Round values for cleaner display
            roi_values = [round(v, 2) for v in top_roi['buy_advantage'].tolist()]
        else:
            roi_labels = []
            roi_values = []
        
        # Buy vs Rent distribution (handle full text decisions)
        buy_count = 0
        rent_count = 0
        if 'decision' in properties_df.columns:
            decision_lower = properties_df['decision'].str.lower()
            buy_count = int(len(properties_df[decision_lower.str.contains('buy', na=False)]))
            rent_count = int(len(properties_df[decision_lower.str.contains('rent', na=False)]))
        
        # Price distribution by city (average price)
        if 'city' in properties_df.columns:
            city_prices = properties_df.groupby('city')['price'].mean().sort_values(ascending=False).head(8)
            price_labels = city_prices.index.tolist()
            price_values = city_prices.values.tolist()
        else:
            price_labels = []
            price_values = []
        
        # City properties count
        city_properties_labels = []
        city_properties_values = []
        if 'city' in properties_df.columns:
            city_counts = properties_df['city'].value_counts().head(8)
            city_properties_labels = city_counts.index.tolist()
            city_properties_values = city_counts.values.tolist()
        
        # Top locations by average price
        location_price_labels = []
        location_price_values = []
        if 'location' in properties_df.columns and 'price' in properties_df.columns:
            location_prices = properties_df.groupby('location')['price'].mean().sort_values(ascending=False).head(10)
            location_price_labels = location_prices.index.tolist()
            location_price_values = location_prices.values.tolist()
        
        # Price per sqft by city
        price_per_sqft_labels = []
        price_per_sqft_values = []
        if 'city' in properties_df.columns and 'price_per_sqft' in properties_df.columns:
            city_ppsqft = properties_df.groupby('city')['price_per_sqft'].mean().sort_values(ascending=False).head(8)
            price_per_sqft_labels = city_ppsqft.index.tolist()
            price_per_sqft_values = city_ppsqft.values.tolist()
        
        return {
            'roi_labels': roi_labels,
            'roi_values': roi_values,
            'buy_vs_rent_labels': ['Buy', 'Rent'],
            'buy_vs_rent_values': [buy_count, rent_count],
            'price_distribution_labels': price_labels,
            'price_distribution_values': price_values,
            'city_properties_labels': city_properties_labels,
            'city_properties_values': city_properties_values,
            'location_price_labels': location_price_labels,
            'location_price_values': location_price_values,
            'price_per_sqft_labels': price_per_sqft_labels,
            'price_per_sqft_values': price_per_sqft_values
        }


# Utility functions for quick access
def load_properties_data() -> pd.DataFrame:
    """
    Load properties data from CSV
    Returns empty DataFrame if file doesn't exist
    """
    try:
        df = pd.read_csv('data/outputs/analyzed_properties.csv')
        return df
    except FileNotFoundError:
        # Return sample data structure
        return pd.DataFrame(columns=['location', 'city', 'price', 'area_sqft', 'bhk', 
                                    'price_per_sqft', 'decision', 'wealth_buying', 'wealth_renting'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def estimate_rent(price: float, area_sqft: float) -> float:
    """
    Estimate monthly rent based on property characteristics
    Rule of thumb: 0.3-0.5% of property value per month
    """
    # More sophisticated estimation based on area
    rent_per_sqft = 20 if area_sqft > 0 else 0
    estimated_rent = area_sqft * rent_per_sqft
    
    # Cap between 0.2% to 0.4% of price
    min_rent = price * 0.002
    max_rent = price * 0.004
    
    return round(max(min_rent, min(estimated_rent, max_rent)), 2)
