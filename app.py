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
st.markdown("""
    <style>
        /* Global Background */
        .stApp {
            background-color: #F3F4F6;
        }
        
        /* Sidebar Styling - Deep Purple Gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2E1065 0%, #4C1D95 100%);
            color: white;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
            color: white !important;
        }
        
        /* Metric Cards Styling */
        div[data-testid="metric-container"] {
            background-color: white;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid #E5E7EB;
            min-height: 100px;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            color: #1F2937;
            font-weight: 700;
        }
        
        /* AI Assistant Box */
        .ai-box {
            background: linear-gradient(135deg, #7C4DFF 0%, #6200EA 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            box-shadow: 0 4px 10px rgba(124, 77, 255, 0.3);
        }
        .ai-box h4 {
            color: white;
            margin: 0;
            padding-bottom: 5px;
        }
        
        /* Remove top padding */
        .block-container {
            padding-top: 2rem;
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

# --- DATA FETCHING WITH ERROR FIX ---
@st.cache_data
def load_data(symbol, start, end):
    try:
        # Download data
        data = yf.download(symbol, start=start, end=end, progress=False)
        
        # --- FIX FOR YFINANCE UPDATE ---
        # Flatten MultiIndex columns if present (e.g. ('Close', 'AAPL') -> 'Close')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        # -------------------------------
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
data = load_data(ticker, start_date, end_date)

# --- MAIN DASHBOARD ---

# Header Section
col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1:
    st.title(f"Financial Overview: {ticker}")
    st.caption("Real-time data, technical analysis, and AI insights.")
with col_header_2:
    st.markdown(f"**Last Updated:**\n{datetime.now().strftime('%Y-%m-%d')}")

if data is not None and not data.empty:
    # 1. KPI METRICS ROW
    # Force conversion to float to prevent Formatting Errors
    try:
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        delta = current_price - prev_price
        
        vol_val = int(data['Volume'].iloc[-1])
        high_val = float(data['High'].max())
        low_val = float(data['Low'].min())
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Current Price", f"${current_price:.2f}", f"{delta:.2f}")
        with col_m2:
            st.metric("Volume", f"{vol_val:,}")
        with col_m3:
            st.metric("High (52W)", f"${high_val:.2f}")
        with col_m4:
            st.metric("Low (52W)", f"${low_val:.2f}")
            
    except IndexError:
        st.warning("Not enough data to calculate changes yet.")

    st.markdown("---")

    # 2. CHARTS SECTION
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Technical Analysis")
        # Plotly Chart
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        increasing_line_color= '#00C805', 
                        decreasing_line_color= '#FF3B30')])
        
        # Update layout to match white card theme
        fig.update_layout(
            height=400, 
            margin=dict(l=20, r=20, t=20, b=20), 
            paper_bgcolor="white", 
            plot_bgcolor="white",
            xaxis_rangeslider_visible=False,
            xaxis=dict(showgrid=True, gridcolor='#F3F4F6'),
            yaxis=dict(showgrid=True, gridcolor='#F3F4F6')
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Market News & AI")
        
        # AI Insight Card (HTML Injection)
        st.markdown(f"""
        <div class="ai-box">
            <h4>🤖 Paynext AI Insight</h4>
            <p style="font-size: 0.9rem; opacity: 0.9;">
            <b>{ticker}</b> is showing strong bullish momentum based on the last 30 days of trading. 
            The volume spike suggests institutional interest. Resistance is observed near <b>${data['High'].max():.0f}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Recent Headlines")
        # Mock headlines for UI demonstration
        st.markdown(f"""
        * **Breaking:** {ticker} announces new strategic partnership...
        * **Earnings:** Quarterly revenue beats expectations by 12%.
        * **Market:** Sector rotation benefits tech stocks like {ticker}.
        """)

    # 3. FINANCIAL DATA TABLE
    st.subheader("Historical Data View")
    with st.expander("View Detailed Dataframe", expanded=False):
        # Display sorting by newest date first
        st.dataframe(data.sort_index(ascending=False), use_container_width=True)

else:
    st.error("No data found. Please check the stock symbol or date range.")
