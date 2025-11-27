import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Paynext | Pro Research",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #F3F4F6; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #2E1065 0%, #4C1D95 100%); color: white; }
        [data-testid="stSidebar"] * { color: white !important; }
        
        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border: 1px solid #E5E7EB;
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: white;
            border-radius: 5px;
            color: #4C1D95;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stTabs [aria-selected="true"] {
            background-color: #7C4DFF;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(symbol, start, end):
    """Fetch history and company info safely."""
    try:
        ticker_obj = yf.Ticker(symbol)
        
        # 1. Get History
        df = ticker_obj.history(start=start, end=end)
        
        # Flatten MultiIndex if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 2. Get Info (Fundamentals)
        info = ticker_obj.info
        
        # 3. Get News
        news = ticker_obj.news
        
        return df, info, news
    except Exception as e:
        return None, None, None

def calculate_technical_indicators(df):
    """Calculate SMA and EMA manually to avoid heavy dependencies."""
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.title("Paynext Pro")
    
    symbol = st.text_input("Stock Symbol", "AAPL").upper()
    
    # Date Range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End", datetime.now())
        
    st.markdown("---")
    st.subheader("Chart Overlay")
    show_sma = st.checkbox("Show SMA (50 & 200)")
    show_ema = st.checkbox("Show EMA (20)")
    
    st.markdown("---")
    st.caption("Paynext Financial Studio v2.0")

# --- MAIN LOGIC ---
df, info, news = load_data(symbol, start_date, end_date)

if df is not None and not df.empty:
    df = calculate_technical_indicators(df)
    
    # --- HEADER & FUNDAMENTALS ---
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title(f"{info.get('longName', symbol)}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
    with col_h2:
        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}", 
                  f"{df['Close'].iloc[-1] - df['Close'].iloc[-2]:.2f}")

    # --- METRICS ROW ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Cap", f"${info.get('marketCap', 0):,}")
    m2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
    m3.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
    m4.metric("Volume", f"{df['Volume'].iloc[-1]:,}")

    st.markdown("---")

    # --- TABS FOR ORGANIZED VIEW ---
    tab_chart, tab_news, tab_data = st.tabs(["📈 Technical Chart", "📰 Latest News", "📊 Historical Data"])

    # TAB 1: ADVANCED CHART
    with tab_chart:
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'],
                        name='Price'))

        # Add Indicators based on User Selection
        if show_sma:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'))
        
        if show_ema:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='purple', width=1), name='EMA 20'))

        fig.update_layout(
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Simulated AI Insight based on Simple Logic
        last_close = df['Close'].iloc[-1]
        sma_200 = df['SMA_200'].iloc[-1] if not pd.isna(df['SMA_200'].iloc[-1]) else last_close
        trend = "Bullish 🟢" if last_close > sma_200 else "Bearish 🔴"
        
        st.info(f"**AI Technical Check:** The stock is currently trading **{trend}** relative to its 200-day moving average.")

    # TAB 2: NEWS FEED
    with tab_news:
        st.subheader(f"News for {symbol}")
        if news:
            for article in news[:5]:  # Show top 5 news
                with st.container():
                    st.markdown(f"**[{article['title']}]({article['link']})**")
                    st.caption(f"Publisher: {article.get('publisher', 'Unknown')} | {datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%d')}")
                    st.markdown("---")
        else:
            st.write("No news found directly via API.")

    # TAB 3: DATA & DOWNLOAD
    with tab_data:
        st.subheader("Raw Data")
        
        # Download Button
        csv = df.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{symbol}_data.csv",
            mime="text/csv",
        )
        
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)

else:
    st.error("Error loading data. Please check the symbol.")
