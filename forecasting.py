import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from prophet.plot import plot_plotly

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Time Series Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Streamlit Time Series Forecast")
st.markdown("Use Prophet to forecast time series data from a CSV file.")

# --- SIDEBAR: DATA UPLOAD ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data loaded successfully!")

    # --- SIDEBAR: COLUMN SELECTION ---
    st.sidebar.header("2. Configure Data")
    
    # Get all column names for selection
    cols = data.columns.tolist()
    
    # Select Date Column (Prophet requires it to be named 'ds')
    date_col = st.sidebar.selectbox("Select Date Column", cols)
    
    # Select Target Column (Prophet requires it to be named 'y')
    target_col = st.sidebar.selectbox("Select Target Column (Value to Forecast)", cols)
    
    # Drop columns not needed for forecasting and rename for Prophet
    prophet_data = data[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    
    # Convert 'ds' column to datetime
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
    
    st.subheader("Raw Data Preview")
    st.dataframe(prophet_data.head())
    
    # --- MAIN PANEL: FORECASTING PARAMETERS ---
    st.header("🎯 Forecast Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Forecast Horizon
        periods = st.number_input(
            "Periods to Forecast (e.g., days, months)",
            min_value=1,
            max_value=365,
            value=90,
            step=1
        )
        
    with col2:
        # Seasonality (Prophet default is yearly)
        yearly_seasonality = st.checkbox("Include Yearly Seasonality?", value=True)
        
    # --- FORECASTING EXECUTION ---
    if st.button("Run Forecast", type="primary"):
        st.info(f"Running Prophet model to forecast {periods} periods...")
        
        try:
            # 1. Initialize and Fit the Model
            m = Prophet(
                yearly_seasonality=yearly_seasonality,
                # Add more parameters here if needed, like daily_seasonality=True
            )
            m.fit(prophet_data)
            
            # 2. Make Future DataFrame
            future = m.make_future_dataframe(periods=periods)
            
            # 3. Predict
            forecast = m.predict(future)
            
            st.success("Forecast completed!")
            
            # --- RESULTS VISUALIZATION ---
            st.subheader("Forecast Plot")
            
            # Use Prophet's built-in Plotly function
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1, use_container_width=True)
            
            # --- COMPONENTS PLOT ---
            st.subheader("Model Components")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)
            
            # --- RAW FORECAST DATA ---
            st.subheader("Forecast Data (Last 5 periods)")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            
        except Exception as e:
            st.error(f"An error occurred during forecasting: {e}")

else:
    st.warning("Please upload a CSV file in the sidebar to begin.") 
