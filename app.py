import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Paynext | Fintech Research",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR "PAYNEXT" UI/UX ---
# This injects the purple gradient sidebar and card-like styling from your image
st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background-color: #F3F4F6;
        }
        
        /* Sidebar Styling - Deep Purple Gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2E1065 0%, #4C1D95 100%);
            color: white;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
            color: white !important;
        }
        
        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid #E5E7EB;
        }
        
        /* Custom "Card" Container Class */
        .css-card {
            background-color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            color: #1F2937;
        }
        
        /* AI Assistant Box */
        .ai-box {
            background: linear-gradient(135deg, #7C4DFF 0%, #6200EA 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Paynext")
    st.markdown("### Fintech Workspace")
    
    # Input Widgets
    ticker = st.text_input("Enter Stock Symbol", value="AAPL").upper()
    
    # Date Slider
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    end_date = st.date_input("End Date", value=datetime.now())
    
    st.markdown("---")
    st.write("Current Mode: **Research**")
    st.info("AI Assistant Active 🟢")

# --- DATA FETCHING ---
@st.cache_data
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        return data
    except Exception as e:
        return None

data = load_data(ticker, start_date, end_date)

# --- MAIN DASHBOARD ---

# Header Section
col1, col2 = st.columns([3, 1])
with col1:
    st.title(f"Financial Overview: {ticker}")
    st.markdown("Real-time data, technical analysis, and AI insights.")
with col2:
    st.markdown(f"**Last Updated:**\n{datetime.now().strftime('%Y-%m-%d')}")

if data is not None and not data.empty:
    # 1. KPI METRICS ROW
    # We use a container to mimic the white dashboard cards
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    delta = current_price - prev_price
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Current Price", f"${current_price:.2f}", f"{delta:.2f}")
    with col_m2:
        st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
    with col_m3:
        st.metric("High (52W)", f"${data['High'].max():.2f}")
    with col_m4:
        st.metric("Low (52W)", f"${data['Low'].min():.2f}")

    st.markdown("---")

    # 2. CHARTS SECTION
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Technical Analysis")
        # Plotly Chart using Paynext Colors
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        increasing_line_color= '#00C805', 
                        decreasing_line_color= '#FF3B30')])
        
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Market News & AI")
        
        # Simulating the AI Assistant Feature mentioned in your text
        st.markdown("""
        <div class="ai-box">
            <b>🤖 Paynext AI Insight</b><br>
            Based on recent moving averages, AAPL is showing strong bullish momentum. 
            RSI indicates the stock is approaching overbought territory.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Recent Headlines")
        st.markdown("""
        * *Apple announces new AI features for iOS...*
        * *Q3 Earnings beat expectations by 15%...*
        * *Analyst upgrade: Price target raised to $200...*
        """)

    # 3. FINANCIAL DATA TABLE
    st.subheader("Historical Data View")
    with st.expander("View Detailed Dataframe", expanded=True):
        st.dataframe(data.sort_index(ascending=False), use_container_width=True)

else:
    st.error("No data found. Please check the stock symbol.")
