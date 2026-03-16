import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="RF Pairs Trading", layout="wide")

st.title("🌲 Random Forest: Pairs Trading Dashboard")
st.markdown("Interactive analysis of rolling 60-day RF model predictions.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('rf_final_results.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

try:
    df = load_data()
    pairs = df['pair'].unique()

    # Sidebar
    st.sidebar.header("Navigation")
    selected_pair = st.sidebar.selectbox("Select Pair", pairs)
    
    # Filter Data
    pair_df = df[df['pair'] == selected_pair].sort_values('date')

    # Metrics
    mae = np.mean(np.abs(pair_df['target'] - pair_df['prediction']))
    mape = np.mean(np.abs((pair_df['target'] - pair_df['prediction']) / pair_df['target'])) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Pair", selected_pair)
    c2.metric("Mean Absolute Error", f"{mae:.4f}")
    c3.metric("MAPE", f"{mape:.2f}%")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pair_df['date'], y=pair_df['target'], name="Actual Spread"))
    fig.add_trace(go.Scatter(x=pair_df['date'], y=pair_df['prediction'], name="RF Prediction", line=dict(dash='dash')))
    
    fig.update_layout(title=f"Spread Prediction: {selected_pair}", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {e}")
