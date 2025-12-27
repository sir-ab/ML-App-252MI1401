import streamlit as st
import pandas as pd
import joblib
import json
import io
from datetime import datetime, time

# --- Page config ---
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
# This ensures results don't disappear when you click other buttons (like Download)
if 'bulk_results' not in st.session_state:
    st.session_state['bulk_results'] = None

# --- MAPPINGS ---
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
    
    /* Global Headers */
    h1, h2, h3, h4, h5 {
        color: white !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255,255,255,0.1);
        padding: 10px 10px 0px 10px;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.2);
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #667eea !important;
        font-weight: bold;
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
    /* Force headers inside the form to be dark */
    [data-testid="stForm"] h3 {
        color: #2c3e50 !important; 
        margin-top: 0;
    }
    .stSelectbox, .stNumberInput, .stTextInput, .stTimeInput {
        color: #2c3e50;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 16px;
        font-weight: bold;
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
    .success-bg { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
    .warning-bg { background: linear-gradient(135deg, #f12711 0%, #f5af19 100%); }
    
    /* --- METRICS & TABLES --- */
    .metrics-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    /* FORCE TEXT COLOR BLACK/DARK INSIDE METRICS CARD */
    .metrics-card h1, 
    .metrics-card h2, 
    .metrics-card h3, 
    .metrics-card h4, 
    .metrics-card h5, 
    .metrics-card div, 
    .metrics-card p,
    .metrics-card span {
        color: #2c3e50 !important;
    }
    
    /* Recommendation Box */
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
        st.error(f"Error loading model: {e}")
        st.stop()

model, model_info = load_model()
numeric_features = model_info.get('features', {}).get('numeric', [])
categorical_features = model_info.get('features', {}).get('categorical', [])
all_features = numeric_features + categorical_features

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Model Information")
    st.markdown(f"""
    <div class='sidebar-card'>
        <b>Algo:</b> {model_info.get('model_name', 'N/A')}<br>
    </div>
    """, unsafe_allow_html=True)
    delay_rate = model_info.get('delay_rate', 0.4)
    st.metric("Avg Delay Rate", f"{delay_rate:.1%}")
    st.progress(delay_rate)
    st.markdown("---")
    st.info("Use the tabs to switch between Single Flight prediction and Bulk CSV Upload.")

# --- Header ---
st.markdown("""
    <div style='text-align: center; padding: 10px; margin-bottom: 20px;'>
        <h1 style='font-size: 42px; margin-bottom: 5px; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>⋆｡ﾟ☁︎｡✈︎⋆｡☾Flight Delay Predictor ⋆｡ﾟ☁︎｡✈︎⋆｡☾</h1>
        <p style='font-size: 18px; color: rgba(255,255,255,0.9);'>AI-Powered Flight Status Forecasting</p>
    </div>
""", unsafe_allow_html=True)

# --- TABS Interface ---
tab1, tab2 = st.tabs(["🎫 Single Flight Prediction", "📂 Bulk Prediction (Upload)"])

# ==========================================
# TAB 1: SINGLE PREDICTION
# ==========================================
with tab1:
    with st.form("flight_form"):
        st.markdown("### ✈️ Flight Information")
        
        col1, col2, col3 = st.columns(3)
        form_data = {}
        cols = [col1, col2, col3]
        
        # 1. Airline
        with col1:
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

        # 3. Dynamic Features
        remaining_features = [f for f in all_features if f not in ['Airline', 'AirportFrom', 'AirportTo']]
        
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
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            submit = st.form_submit_button("Predict Status", use_container_width=True)

    if submit:
        with st.spinner("Analyzing flight data..."):
            input_df = pd.DataFrame([form_data])
            try:
                proba = model.predict_proba(input_df)[0]
                prob_delay = proba[1]
                prob_ontime = proba[0]
                is_delayed = prob_delay >= 0.5
                
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

                st.markdown(f"""
                <div class='metrics-card'>
                    <h3 style='margin-top: 0; margin-bottom: 15px;'>Confidence Analysis</h3>
                    <div style='display: flex; gap: 20px; justify-content: center;'>
                        <div style='flex: 1; background: #2193b0; padding: 15px; border-radius: 10px; text-align: center;'>
                            <div style='font-size: 14px; opacity: 0.9; color: white !important;'>On Time</div>
                            <div style='font-size: 28px; font-weight: bold; color: white !important;'>{prob_ontime:.1%}</div>
                        </div>
                        <div style='flex: 1; background: #f5af19; padding: 15px; border-radius: 10px; text-align: center;'>
                            <div style='font-size: 14px; opacity: 0.9; color: white !important;'>Delayed</div>
                            <div style='font-size: 28px; font-weight: bold; color: white !important;'>{prob_delay:.1%}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

# ==========================================
# TAB 2: BULK PREDICTION
# ==========================================
with tab2:
    st.markdown("### 📂 Bulk CSV Upload")
    
    with st.expander("ℹ️ Instructions & Template", expanded=True):
        st.markdown("""
        Upload a CSV file containing multiple flights. The file must match the required feature columns.
        
        **Required Columns:**
        """)
        st.code(", ".join(all_features), language="text")
        st.markdown("Download a sample template below to get started:")
        
        # Create Template
        template_data = {
            'Airline': ['Delta', 'Southwest', 'United'],
            'AirportFrom': ['ATL', 'SFO', 'JFK'],
            'AirportTo': ['SFO', 'LAX', 'ORD'],
            'DayOfWeek': [1, 5, 7],
            'Time': [800, 1430, 2100],
            'Length': [300, 90, 150]
        }
        for f in all_features:
            if f not in template_data:
                template_data[f] = [0, 0, 0]
                
        template_df = pd.DataFrame(template_data)
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Download CSV Template",
            data=csv_template,
            file_name="flight_prediction_template.csv",
            mime="text/csv"
        )

    # File Uploader with specific key to help state stability
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="bulk_upload_widget")

    # Clear results if the user clears the file uploader
    if uploaded_file is None:
        st.session_state['bulk_results'] = None

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            # Validation
            missing_cols = [col for col in all_features if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns in CSV: {', '.join(missing_cols)}")
            else:
                st.success(f"✅ Successfully loaded {len(input_df)} flights.")
                
                # Predict Button
                if st.button("Predict All Flights", use_container_width=True):
                    with st.spinner("Processing batch predictions..."):
                        prediction_df = input_df[all_features].copy()
                        probas = model.predict_proba(prediction_df)
                        
                        results_df = input_df.copy()
                        results_df['Probability_Delayed'] = probas[:, 1]
                        results_df['Prediction'] = ["Delayed" if p >= 0.5 else "On Time" for p in probas[:, 1]]
                        
                        # Save to session state so it persists
                        st.session_state['bulk_results'] = results_df
                        
                # --- DISPLAY RESULTS FROM SESSION STATE ---
                if st.session_state['bulk_results'] is not None:
                    final_df = st.session_state['bulk_results']
                    
                    st.markdown("### 📊 Results")
                    
                    st.dataframe(
                        final_df,
                        column_config={
                            "Probability_Delayed": st.column_config.ProgressColumn(
                                "Delay Risk",
                                help="Probability of delay",
                                format="%.2f",
                                min_value=0,
                                max_value=1,
                            ),
                            "Prediction": st.column_config.TextColumn(
                                "Status",
                                help="Predicted Status",
                            )
                        },
                        use_container_width=True
                    )
                    
                    # Download Button
                    csv_result = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Predictions",
                        data=csv_result,
                        file_name="flight_predictions_result.csv",
                        mime="text/csv"
                    )
                    
                    # Metric Summary (Dark Text fixed via CSS)
                    total = len(final_df)
                    delays = len(final_df[final_df['Prediction'] == 'Delayed'])
                    
                    st.markdown(f"""
                    <div class='metrics-card' style='text-align:center;'>
                        <h4>Batch Summary</h4>
                        <p>{delays} out of {total} flights are predicted to be delayed ({delays/total:.1%})</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")