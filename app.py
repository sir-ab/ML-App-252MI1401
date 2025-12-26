import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime, time

# --- Page config ---
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MAPPINGS (Full Names vs Model Inputs) ---
# NOTE: Keys match what the model expects (from training data). 
# Values are what the user sees.

AIRLINE_MAPPING = {
    'Alaska': 'Alaska Airlines (AS)',
    'American': 'American Airlines (AA)',
    'Air Canada': 'Air Canada (AC)',
    'Aeromexico': 'Aeromexico (AM)',
    'Continental': 'Continental Airlines (CO)',
    'Delta': 'Delta Airlines (DL)',
    'FedEx': 'FedEx (FX)',
    'Hawaiian': 'Hawaiian Airlines (HA)',
    'Northwest': 'Northwest Airlines (NW)',
    'Polar': 'Polar Air Cargo (PO)',
    'Southwest': 'Southwest Airlines (WN)',
    'United': 'United Airlines (UA)',
    'UPS': 'United Parcel Service (5X)',
    'Virgin': 'Virgin Atlantic (VS)',
    'Viva': 'VivaAerobús (VB)',
    'WestJet': 'WestJet (WS)',
    'JetBlue': 'JetBlue Airways (B6)',
    'Spirit': 'Spirit Airlines (NK)',
    'Frontier': 'Frontier Airlines (F9)',
    'Unknown': 'Other / Unknown'
}

AIRPORT_MAPPING = {
    'ATL': 'ATL - Hartsfield-Jackson Atlanta Intl (GA)',
    'AUS': 'AUS - Austin-Bergstrom Intl (TX)',
    'BNA': 'BNA - Nashville Intl (TN)',
    'BOS': 'BOS - Boston Logan Intl (MA)',
    'BWI': 'BWI - Baltimore-Washington Intl (WA)',
    'CLT': 'CLT - Charlotte Douglas Intl (NC)',
    'DAL': 'DAL - Dallas Love Field (TX)',
    'DCA': 'DCA - Ronald Reagan Washington National (VA)',
    'DEN': 'DEN - Denver Intl (CO)',
    'DFW': 'DFW - Dallas/Fort Worth Intl (TX)',
    'DTW': 'DTW - Detroit Metropolitan (MI)',
    'EWR': 'EWR - Newark Liberty Intl (NJ)',
    'FLL': 'FLL - Fort Lauderdale–Hollywood Intl (FL)',
    'HNL': 'HNL - Daniel K. Inouye Intl (HI)',
    'HOU': 'HOU - William P. Hobby (TX)',
    'IAD': 'IAD - Dulles Intl (VA)',
    'IAH': 'IAH - George Bush Intercontinental (TX)',
    'JFK': 'JFK - John F. Kennedy Intl (NY)',
    'LAS': 'LAS - McCarran Intl (NV)',
    'LAX': 'LAX - Los Angeles Intl (CA)',
    'LGA': 'LGA - LaGuardia (NY)',
    'MCO': 'MCO - Orlando Intl (FL)',
    'MDW': 'MDW - Chicago Midway Intl (IL)',
    'MIA': 'MIA - Miami Intl (FL)',
    'MSP': 'MSP - Minneapolis–Saint Paul Intl (MN)',
    'MSY': 'MSY - Louis Armstrong New Orleans Intl (LA)',
    'OAK': 'OAK - Oakland Intl (CA)',
    'ORD': 'ORD - O\'Hare Intl (IL)',
    'PDX': 'PDX - Portland Intl (OR)',
    'PHL': 'PHL - Philadelphia Intl (PA)',
    'PHX': 'PHX - Phoenix Sky Harbor Intl (AZ)',
    'RDU': 'RDU - Raleigh-Durham Intl (NC)',
    'SAN': 'SAN - San Diego Intl (CA)',
    'SEA': 'SEA - Seattle–Tacoma Intl (WA)',
    'SFO': 'SFO - San Francisco Intl (CA)',
    'SJC': 'SJC - Norman Y. Mineta San Jose Intl (CA)',
    'SLC': 'SLC - Salt Lake City Intl (UT)',
    'SMF': 'SMF - Sacramento Intl (CA)',
    'STL': 'STL - St. Louis Lambert Intl (MO)',
    'TPA': 'TPA - Tampa Intl (FL)'
}

# --- Custom CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Global Headers (White on the purple background) */
    h1, h2, h3 {
        color: white !important;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    .sidebar-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #2c3e50;
        font-size: 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Force text inside sidebar card to be dark */
    .sidebar-card b, .sidebar-card div {
        color: #2c3e50 !important;
    }

    /* --- FORM STYLING --- */
    [data-testid="stForm"] {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        border: none;
    }
    [data-testid="stForm"] label {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    /* Force "Flight Information" header inside form to be dark */
    [data-testid="stForm"] h3 {
        color: #2c3e50 !important; 
        margin-top: 0;
    }
    .stSelectbox, .stNumberInput, .stTextInput, .stTimeInput {
        color: #2c3e50;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 18px;
        font-weight: bold;
        padding: 15px 0px;
        border-radius: 50px;
        border: none;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }

    /* --- RESULT BOXES --- */
    .result-box {
        padding: 25px;
        border-radius: 15px;
        color: white !important;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    /* Blue Gradient for On Time */
    .success-bg { 
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); 
    }
    /* Orange Gradient for Delay (Colorblind safe) */
    .warning-bg { 
        background: linear-gradient(135deg, #f12711 0%, #f5af19 100%); 
    }
    
    /* --- WHITE METRICS CARD --- */
    .metrics-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    /* CRITICAL: Force text black inside the white metrics card */
    .metrics-card h3, .metrics-card div {
        color: #2c3e50 !important;
    }

    .metric-container {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white !important;
        margin-top: 10px;
    }
    .metric-container div {
        color: white !important;
    }

    /* --- RECOMMENDATION BOX --- */
    .rec-box {
        background: rgba(255, 255, 255, 0.95);
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model and info ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("flight_delay_model.pkl")
        with open("model_info.json", "r") as f:
            info = json.load(f)
        return model, info
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        st.stop()
    except Exception as e:
        # Fallback for version mismatch if user didn't retrain
        st.error(f"Error loading model: {e}")
        st.stop()

model, model_info = load_model()

# --- Sidebar Info ---
with st.sidebar:
    st.markdown("### Model Information")
    st.markdown(f"""
    <div class='sidebar-card'>
        <b>Algo:</b> {model_info.get('model_name', 'N/A')}<br>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Historical Delay")
    delay_rate = model_info.get('delay_rate', 0.4) # Default 0.3 if missing
    st.metric("Avg Delay Rate", f"{delay_rate:.1%}")
    st.progress(delay_rate)
    st.markdown("---")
    st.markdown("### How It Works")
    st.info("""
    Enter your flight details and get an instant prediction on whether your flight is likely to be delayed.
    """)

# --- Header ---
st.markdown("""
    <div style='text-align: center; padding: 10px; margin-bottom: 20px;'>
        <h1 style='font-size: 42px; margin-bottom: 5px; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>Flight Delay Predictor</h1>
        <p style='font-size: 18px; color: rgba(255,255,255,0.9);'>AI-Powered Flight Status Forecasting</p>
    </div>
""", unsafe_allow_html=True)

# --- Main Form ---
# We use the native st.form which we styled with CSS to look like a card
# This removes the "White Bar" issue caused by nested divs
with st.form("flight_form"):
    st.markdown("### ✈️ Flight Information")
    
    # Get feature information
    numeric_features = model_info.get('features', {}).get('numeric', [])
    categorical_features = model_info.get('features', {}).get('categorical', [])
    
    col1, col2, col3 = st.columns(3)
    form_data = {}
    
    # Helper to determine where to place input
    cols = [col1, col2, col3]
    
    # 1. Airline (First)
    with col1:
        # Use keys from AIRLINE_MAPPING for options, but display values
        # We need to make sure we include options that the model knows but might not be in our map
        # For this example, we default to the map.
        airline_options = list(AIRLINE_MAPPING.keys())
        form_data['Airline'] = st.selectbox(
            "Airline Carrier", 
            options=airline_options,
            format_func=lambda x: AIRLINE_MAPPING.get(x, x)
        )

    # 2. Airports
    with col2:
        airport_options = list(AIRPORT_MAPPING.keys())
        form_data['AirportFrom'] = st.selectbox(
            "Origin Airport", 
            options=airport_options,
            format_func=lambda x: AIRPORT_MAPPING.get(x, x)
        )
    
    with col3:
        form_data['AirportTo'] = st.selectbox(
            "Destination Airport", 
            options=airport_options,
            format_func=lambda x: AIRPORT_MAPPING.get(x, x)
        )

    # 3. Other Features (Dynamic)
    # We filter out the ones we already placed manually
    remaining_features = [f for f in (categorical_features + numeric_features) 
                         if f not in ['Airline', 'AirportFrom', 'AirportTo']]
    
    for i, feature in enumerate(remaining_features):
        col = cols[i % 3]
        with col:
            display_name = feature.replace('_', ' ').title()
            
            if 'day' in feature.lower() and 'week' in feature.lower():
                form_data[feature] = st.selectbox(
                    display_name,
                    options=[1, 2, 3, 4, 5, 6, 7],
                    format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x-1]
                )
            
            elif 'time' in feature.lower() or 'hour' in feature.lower():
                time_val = st.time_input(display_name, value=time(6, 00))
                form_data[feature] = time_val.hour * 100 + time_val.minute
            
            elif 'length' in feature.lower():
                form_data[feature] = st.number_input(f"{display_name} (min)", 30, 600, 180, 15)
            
            elif feature in numeric_features:
                 form_data[feature] = st.number_input(display_name, value=100)
            
            else:
                form_data[feature] = st.text_input(display_name, value="Unknown")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the button
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        submit = st.form_submit_button("Predict Flight Status", use_container_width=True)

# --- Prediction Results ---
if submit:
    with st.spinner("Calculating probabilities..."):
        input_df = pd.DataFrame([form_data])
        
        try:
            # Get Probability
            proba = model.predict_proba(input_df)[0]
            prob_delay = proba[1]
            prob_ontime = proba[0]
            
            # Custom Threshold Logic
            THRESHOLD = 0.5
            is_delayed = prob_delay >= THRESHOLD
            
            # 1. Main Status Box
            if is_delayed:
                st.markdown(f"""
                <div class='result-box warning-bg'>
                    <div style='font-size: 32px; font-weight: bold;'>⚠️ HIGH DELAY RISK</div>
                    <div style='font-size: 18px; margin-top: 5px;'>Likelihood: {prob_delay:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-box success-bg'>
                    <div style='font-size: 32px; font-weight: bold;'>✅ LIKELY ON TIME</div>
                    <div style='font-size: 18px; margin-top: 5px;'>On-Time Probability: {prob_ontime:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            # 2. Detailed Metrics (White Card)
            # Added 'metrics-card' class to handle text color
            st.markdown(f"""
            <div class='metrics-card'>
                <h3 style='margin-top: 0; margin-bottom: 15px;'>Confidence Analysis</h3>
                <div style='display: flex; gap: 20px; justify-content: center;'>
                    <div style='flex: 1; background: #2193b0; padding: 15px; border-radius: 10px; text-align: center; color: white;'>
                        <div style='font-size: 14px; opacity: 0.9; color: white !important;'>On Time</div>
                        <div style='font-size: 28px; font-weight: bold; color: white !important;'>{prob_ontime:.1%}</div>
                    </div>
                    <div style='flex: 1; background: #f5af19; padding: 15px; border-radius: 10px; text-align: center; color: white;'>
                        <div style='font-size: 14px; opacity: 0.9; color: white !important;'>Delayed</div>
                        <div style='font-size: 28px; font-weight: bold; color: white !important;'>{prob_delay:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. Recommendations (Custom styled box)
            if is_delayed:
                st.markdown("""
                <div class='rec-box'>
                    <b>Recommendation:</b> This flight has a high risk of delay.<br>
                    Please check your connection times, bring extra entertainment, and monitor the airline app.
                </div>
                """, unsafe_allow_html=True)
            else:
                 st.markdown("""
                <div class='rec-box' style='border-left-color: #2193b0;'>
                    <b>Recommendation:</b> Flight looks good.<br>
                    Arrive at the airport 2 hours early and have a safe trip!
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")