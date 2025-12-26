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

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .prediction-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.3);
    }
    .warning-box {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(238, 9, 121, 0.3);
    }
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px 40px;
        border-radius: 50px;
        border: none;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    .stSelectbox, .stNumberInput {
        background: white;
        border-radius: 10px;
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

model, model_info = load_model()

# --- Header ---
st.markdown("""
    <div style='text-align: center; padding: 20px; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; margin-bottom: 10px;'>Flight Delay Predictor</h1>
        <p style='font-size: 20px; color: rgba(255,255,255,0.9);'>
            Predict flight delays using machine learning
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Info ---
with st.sidebar:
    st.markdown("### Model Information")
    st.markdown(f"""
    <div class='info-card'>
        <b>Model:</b> {model_info.get('model_name', 'N/A')}<br>
        <b>Accuracy:</b> {model_info.get('metrics', {}).get('Accuracy', 0):.2%}<br>
        <b>F1 Score:</b> {model_info.get('metrics', {}).get('F1 Score', 0):.2%}<br>
        <b>Training Samples:</b> {model_info.get('training_samples', 'N/A'):,}<br>
        <b>Last Updated:</b> {model_info.get('training_date', 'N/A')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### How It Works")
    st.info("""
    Enter your flight details and get an instant prediction on whether your flight is likely to be delayed.
    """)
    
    st.markdown("---")
    
    st.markdown("### Statistics")
    delay_rate = model_info.get('delay_rate', 0)
    st.metric("Historical Delay Rate", f"{delay_rate:.1%}")
    st.progress(delay_rate)

# --- Main Form ---
st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)

# Get feature information
numeric_features = model_info.get('features', {}).get('numeric', [])
categorical_features = model_info.get('features', {}).get('categorical', [])

with st.form("flight_form"):
    st.markdown("### Flight Information")
    
    col1, col2, col3 = st.columns(3)
    
    # Dictionary to store form inputs
    form_data = {}
    
    # Dynamically create inputs based on features
    for i, feature in enumerate(categorical_features + numeric_features):
        col = [col1, col2, col3][i % 3]
        
        with col:
            # Format feature name for display
            display_name = feature.replace('_', ' ').title()
            
            if feature in categorical_features:
                # For categorical features, provide common options
                if 'airline' in feature.lower():
                    options = ['Southwest', 'Delta', 'American', 'United', 'JetBlue', 'Spirit', 'Alaska', 'Frontier']
                    form_data[feature] = st.selectbox(display_name, options)
                
                elif 'airport' in feature.lower() and 'from' in feature.lower():
                    options = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'MCO', 'LAS', 'PHX', 'MIA', 'JFK', 'SFO', 'SEA', 'BOS', 'EWR']
                    form_data[feature] = st.selectbox(f"Origin {display_name.replace('Airport From', '')}", options)
                
                elif 'airport' in feature.lower() and 'to' in feature.lower():
                    options = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'MCO', 'LAS', 'PHX', 'MIA', 'JFK', 'SFO', 'SEA', 'BOS', 'EWR']
                    form_data[feature] = st.selectbox(f"Destination {display_name.replace('Airport To', '')}", options)
                
                else:
                    form_data[feature] = st.text_input(display_name, value="Unknown")
            
            elif feature in numeric_features:
                # For numeric features, provide appropriate ranges
                if 'day' in feature.lower() and 'week' in feature.lower():
                    form_data[feature] = st.selectbox(
                        display_name,
                        options=[1, 2, 3, 4, 5, 6, 7],
                        format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x-1]
                    )
                
                elif 'time' in feature.lower() or 'hour' in feature.lower():
                    time_val = st.time_input(display_name, value=time(14, 30))
                    form_data[feature] = time_val.hour * 100 + time_val.minute
                
                elif 'length' in feature.lower() or 'duration' in feature.lower():
                    form_data[feature] = st.number_input(
                        f"{display_name} (minutes)",
                        min_value=30,
                        max_value=600,
                        value=180,
                        step=15
                    )
                
                elif 'distance' in feature.lower():
                    form_data[feature] = st.number_input(
                        f"{display_name} (miles)",
                        min_value=100,
                        max_value=3000,
                        value=500,
                        step=50
                    )
                
                else:
                    form_data[feature] = st.number_input(
                        display_name,
                        min_value=0.0,
                        value=100.0,
                        step=10.0
                    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit = st.form_submit_button("Predict Flight Status", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction Results ---
if submit:
    with st.spinner("Analyzing flight data..."):
        # Create DataFrame for prediction
        input_data = pd.DataFrame([form_data])
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Display results
            if prediction == 0:
                st.markdown(f"""
                <div class='success-box'>
                    Flight On Time
                    <div style='font-size: 16px; margin-top: 10px; font-weight: normal;'>
                        This flight is predicted to depart on schedule
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='warning-box'>
                    Flight Delayed
                    <div style='font-size: 16px; margin-top: 10px; font-weight: normal;'>
                        This flight may experience delays
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability visualization
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            st.markdown("### Prediction Confidence")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class='metric-container' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
                    <h3 style='margin: 0; color: white;'>On Time</h3>
                    <h1 style='margin: 10px 0; color: white;'>{probability[0]:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
                st.progress(probability[0])
            
            with col2:
                st.markdown(f"""
                <div class='metric-container' style='background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);'>
                    <h3 style='margin: 0; color: white;'>Delayed</h3>
                    <h1 style='margin: 10px 0; color: white;'>{probability[1]:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
                st.progress(probability[1])
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Travel tips
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            st.markdown("### Recommendations")
            
            if prediction == 1:
                st.warning("""
                **Your flight may be delayed. Consider these steps:**
                - Check with your airline for real-time updates
                - Have backup accommodation information ready
                - Bring entertainment and snacks
                - Arrive early at the airport
                - Keep airline customer service contact handy
                """)
            else:
                st.success("""
                **Your flight looks good. General travel tips:**
                - Arrive 2 hours early for domestic flights
                - Check in online to save time
                - Review baggage allowances beforehand
                - Download the airline app for gate updates
                - Plan your airport transportation in advance
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all required fields are filled correctly.")

# --- Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: rgba(255,255,255,0.7); padding: 20px;'>
        <p>Built with Streamlit and Scikit-learn</p>
        <p style='font-size: 12px;'>Predictions are based on historical data and machine learning models. 
        Always verify with your airline for official updates.</p>
    </div>
""", unsafe_allow_html=True)