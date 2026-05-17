import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# 1. Page Configurations for a professional look
st.set_page_config(page_title="Bengaluru Real Estate AI", page_icon="🏡", layout="wide")

# Custom UI styling to give the app a modern dark theme with neon accents
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
    }
    h1 {
        color: #00ffcc;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        text-shadow: 0 0 10px #00ffcc;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00ffcc, #0077ff);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 15px rgba(0, 255, 204, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 20px rgba(0, 255, 204, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏡 Bengaluru Property Valuation Engine")
st.write("Input structural metrics below to evaluate estimated market property values computed by machine learning.")

# Initialize session state for storing results
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
    st.session_state.results_data = {}

# 1. Load exported model artifacts safely
@st.cache_resource
def load_saved_artifacts():
    model = joblib.load('house_price_model.pkl')
    
    # We grab the TRUE feature names straight from the trained model brain itself!
    # This guarantees the capitalization matches 100% perfectly.
    try:
        data_columns = list(model.feature_names_in_)
    except AttributeError:
        # Fallback to file if model attributes aren't accessible
        with open("columns.json", "r") as f:
            data_columns = json.load(f)['data_columns']
            
    return model, data_columns

try:
    model, data_columns = load_saved_artifacts()
    
    # Parse the true column formats
    locations = []
    for col in data_columns:
        # Match features regardless of whether they start with 'location_' or 'Location_'
        if col.lower().startswith('location_') and col.lower() != 'location_other':
            # Clean it up for the UI dropdown presentation
            clean_name = col.split('_', 1)[1].title()
            locations.append(clean_name)
            
    if not locations:
        locations = ["Electronic City", "Whitefield", "HSR Layout", "Yelahanka", "Marathahalli"]
        
    locations = sorted(list(set(locations)))
except Exception as e:
    st.error(f"Error parsing structural model features: {e}")
    st.stop()

# Two-column layout: Left for inputs, Right for results
left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.subheader("📋 Property Details")
    
    # Structural Features Section
    st.write("**Core Structural Metrics**")
    
    col1, col2 = st.columns(2)
    with col1:
        ui_location = st.selectbox("Select Target Neighborhood/Location:", ["Select a location..."] + locations)
        sqft = st.number_input("Total Square Feet Area:", min_value=300, max_value=15000, value=1200, step=50)
    
    with col2:
        bhk = st.slider("Total BHK Layout Construction:", min_value=1, max_value=10, value=2)
        bathrooms = st.slider("Total Bathrooms Planned:", min_value=1, max_value=10, value=2)
    
    # Additional Property Features Section
    st.markdown("---")
    st.write("**Additional Attributes**")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        property_age = st.selectbox(
            "Property Age & Condition:",
            ["Brand New / Under Construction", "1-5 Years Old", "5-10 Years Old", "10+ Years Old"]
        )
    
    with col4:
        furnishing = st.selectbox(
            "Furnishing Status:",
            ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
        )
    
    with col5:
        amenities = st.multiselect(
            "Luxury Amenities:",
            ["24/7 Gated Security", "Swimming Pool & Gym", "Reserved Covered Parking", "Power Backup Generator"]
        )
    
    # Calculate button
    st.markdown("---")
    calc_button = st.button("🚀 Calculate Property Valuation", use_container_width=True)

with right_col:
    st.subheader("📊 Valuation Results")
    
    # Display results if available
    if st.session_state.show_results:
        results = st.session_state.results_data
        
        # Create metric cards in a grid
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                label="Base ML Valuation",
                value=f"₹{results['predicted_price_display']}"
            )
        
        with metric_col2:
            st.metric(
                label="Price per Sqft",
                value=f"₹{results['price_per_sqft']:,.0f}/sqft"
            )
        
        st.markdown("---")
        
        # Display final adjusted valuation
        st.markdown(f"<div style='text-align: center; padding: 20px; background-color: rgba(0, 255, 204, 0.1); border-radius: 10px; border: 2px solid #00ffcc;'><h2 style='color: #00ffcc; margin: 0; text-shadow: 0 0 10px #00ffcc;'>💰 Final Valuation</h2><h1 style='color: #00ffcc; margin: 10px 0 0 0; text-shadow: 0 0 10px #00ffcc;'>{results['adjusted_price_display']}</h1></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display adjustments
        st.write("**Price Adjustments Applied:**")
        for adjustment in results['adjustments_breakdown']:
            st.write(adjustment)
    else:
        st.info("👈 Enter property details and click 'Calculate' to see results here")

# 4. Valuation Prediction Processing Execution
if calc_button:
    if ui_location == "Select a location...":
        st.warning("Please specify a valid neighborhood location to calculate values.")
    else:
        # Create an input vector initialized to 0 matching our columns length
        input_vector = np.zeros(len(data_columns))
        
        # Mapping standard feature scalar indexes
        input_vector[0] = sqft
        input_vector[1] = bathrooms
        input_vector[2] = bhk
        
        # Locate corresponding location column index matching dataset case style
        loc_index = None
        
        # Try exact match, standard lowercase match, or title case match against data_columns
        possible_matches = [
            f"location_{ui_location}",
            f"location_{ui_location.lower()}",
            f"location_{ui_location.title()}",
            f"Location_{ui_location}"
        ]
        
        for match in possible_matches:
            if match in data_columns:
                loc_index = data_columns.index(match)
                break
                
        if loc_index is not None:
            input_vector[loc_index] = 1
        elif 'other' in data_columns:
            input_vector[data_columns.index('other')] = 1
        elif 'location_other' in data_columns:
            input_vector[data_columns.index('location_other')] = 1

        # Reshape data array into a DataFrame for prediction
        input_df = pd.DataFrame([input_vector], columns=data_columns)
        predicted_price = model.predict(input_df)[0]
        
        # Apply business logic adjustments to the predicted price
        adjusted_price = predicted_price
        adjustments_breakdown = []
        
        # 1. Furnishing adjustment
        if furnishing == "Fully Furnished":
            adjusted_price += 5
            adjustments_breakdown.append("✅ Fully Furnished: +₹5 Lakhs")
        elif furnishing == "Semi-Furnished":
            adjusted_price += 2
            adjustments_breakdown.append("✅ Semi-Furnished: +₹2 Lakhs")
        else:
            adjustments_breakdown.append("• Unfurnished: No adjustment")
        
        # 2. Property age depreciation
        if property_age == "1-5 Years Old":
            depreciation = adjusted_price * 0.02
            adjusted_price -= depreciation
            adjustments_breakdown.append(f"📉 1-5 Years Old: -₹{depreciation:.2f} Lakhs (2% depreciation)")
        elif property_age == "5-10 Years Old":
            depreciation = adjusted_price * 0.05
            adjusted_price -= depreciation
            adjustments_breakdown.append(f"📉 5-10 Years Old: -₹{depreciation:.2f} Lakhs (5% depreciation)")
        elif property_age == "10+ Years Old":
            depreciation = adjusted_price * 0.10
            adjusted_price -= depreciation
            adjustments_breakdown.append(f"📉 10+ Years Old: -₹{depreciation:.2f} Lakhs (10% depreciation)")
        else:
            adjustments_breakdown.append("• Brand New: No depreciation")
        
        # 3. Luxury amenities bonus
        amenity_bonus = len(amenities) * 1.5
        if amenities:
            adjusted_price += amenity_bonus
            adjustments_breakdown.append(f"⭐ Amenities ({len(amenities)}): +₹{amenity_bonus:.2f} Lakhs")

        # Calculate price per sqft
        price_per_sqft = (adjusted_price * 100000) / sqft
        
        # Store results in session state
        st.session_state.results_data = {
            'predicted_price': predicted_price,
            'predicted_price_display': f"{predicted_price/100:.2f} Crores" if predicted_price >= 100 else f"{predicted_price:.2f} Lakhs",
            'adjusted_price': adjusted_price,
            'adjusted_price_display': f"₹{adjusted_price/100:.2f} Crores" if adjusted_price >= 100 else f"₹{adjusted_price:.2f} Lakhs",
            'price_per_sqft': price_per_sqft,
            'adjustments_breakdown': adjustments_breakdown,
            'ui_location': ui_location,
            'sqft': sqft
        }
        st.session_state.show_results = True
        st.rerun()

# Display benchmark comparison chart if results are available
if st.session_state.show_results:
    st.markdown("---")
    st.subheader("📈 Market Benchmark Comparison")
    
    # Create benchmark data
    benchmark_data = pd.DataFrame({
        'Location': ['Your Property', 'Indiranagar', 'Whitefield', 'Electronic City'],
        'Price per Sqft (₹)': [
            st.session_state.results_data.get('price_per_sqft', 0),
            12000,
            7500,
            5500
        ]
    })
    
    # Display bar chart
    st.bar_chart(benchmark_data.set_index('Location'), use_container_width=True)
    
    st.write("**Benchmark Hub Details:**")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.write("🏘️ **Indiranagar**\n₹12,000/sqft\n(Prime Central Area)")
    with col_info2:
        st.write("💼 **Whitefield**\n₹7,500/sqft\n(IT Corridor)")
    with col_info3:
        st.write("🔧 **Electronic City**\n₹5,500/sqft\n(Software Hub)")
    
    # Display summary stats in footer
    st.markdown("---")
    st.info(f"✅ Your property in **{st.session_state.results_data.get('ui_location', 'Selected Location')}** at **₹{st.session_state.results_data.get('price_per_sqft', 0):,.0f}/sqft** compares well against major market benchmarks!")
    
    # ===== Financial Buyer Toolkit Section =====
    st.markdown("---")
    st.markdown("<h2 style='color: #00ffcc; text-align: center; text-shadow: 0 0 10px #00ffcc;'>💰 Buyer's Financial Dashboard</h2>", unsafe_allow_html=True)
    
    # Get the final adjusted price in lakhs for calculations
    final_price_lakhs = st.session_state.results_data.get('adjusted_price', 0)
    final_price_amount = final_price_lakhs * 100000  # Convert to actual amount
    
    # Create two columns for EMI Calculator and Fees Breakdown
    emi_col, fees_col = st.columns(2)
    
    with emi_col:
        st.subheader("🏦 EMI Calculator")
        st.write("*Calculate your monthly home loan payment*")
        
        # EMI Calculator Sliders
        loan_percent = st.slider(
            "Loan Amount (% of property value):",
            min_value=50,
            max_value=90,
            value=80,
            step=5,
            help="Select what percentage of the property price you want to finance"
        )
        
        loan_tenure = st.slider(
            "Loan Tenure (Years):",
            min_value=5,
            max_value=30,
            value=20,
            step=1,
            help="Choose the duration to repay the loan"
        )
        
        interest_rate = st.slider(
            "Interest Rate (% per annum):",
            min_value=7.0,
            max_value=12.0,
            value=9.5,
            step=0.1,
            help="Current market interest rates for home loans"
        )
        
        # EMI Calculation Logic
        loan_amount = (loan_percent / 100) * final_price_amount
        monthly_rate = interest_rate / 100 / 12
        num_months = loan_tenure * 12
        
        # EMI Formula: EMI = P * [r(1+r)^n] / [(1+r)^n - 1]
        if monthly_rate > 0:
            emi = loan_amount * (monthly_rate * (1 + monthly_rate)**num_months) / ((1 + monthly_rate)**num_months - 1)
        else:
            emi = loan_amount / num_months
        
        # Display EMI Results
        st.markdown("---")
        st.metric(label="Loan Amount", value=f"₹{loan_amount/100000:.2f} Lakhs")
        st.metric(label="Monthly EMI Payment", value=f"₹{emi:,.0f}")
        st.metric(label="Total Interest Payable", value=f"₹{(emi * num_months - loan_amount)/100000:.2f} Lakhs")
    
    with fees_col:
        st.subheader("📋 Statutory Fees Breakdown")
        st.write("*Karnataka closing costs estimate*")
        
        # Calculate statutory fees
        stamp_duty = final_price_amount * 0.05
        registration_fee = final_price_amount * 0.01
        legal_fees = 50000  # Flat fee
        
        # Display fee breakdown
        st.write("**Cost Breakdown:**")
        st.write(f"• Stamp Duty (5%): ₹{stamp_duty/100000:.2f} Lakhs")
        st.write(f"• Registration Fee (1%): ₹{registration_fee/100000:.2f} Lakhs")
        st.write(f"• Documentation & Legal: ₹{legal_fees/100000:.2f} Lakhs")
        
        st.markdown("---")
        
        # Total on-road cost
        total_on_road_cost = final_price_amount + stamp_duty + registration_fee + legal_fees
        
        st.markdown(f"<div style='text-align: center; padding: 15px; background-color: rgba(0, 255, 204, 0.15); border-radius: 8px; border: 2px solid #00ffcc;'><p style='color: #00ffcc; margin: 0;'>Total Estimated On-Road Cost</p><h3 style='color: #00ffcc; margin: 8px 0 0 0; text-shadow: 0 0 10px #00ffcc;'>₹{total_on_road_cost/100000:.2f} Lakhs</h3></div>", unsafe_allow_html=True)
        
        st.caption(f"Property Price: ₹{final_price_amount/100000:.2f} Lakhs + Closing Costs: ₹{(stamp_duty + registration_fee + legal_fees)/100000:.2f} Lakhs")