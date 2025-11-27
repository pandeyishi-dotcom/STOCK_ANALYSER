import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FOREX | Pro Terminal",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DARK & NEON THEME ---
st.markdown("""
    <style>
        /* Global Background & Text */
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #111111;
            border-right: 1px solid #2C2C2C;
        }
        
        /* Metric Cards styling to match 'FOREX' dashboard */
        div[data-testid="metric-container"] {
            background-color: #2C2C2C;
            padding: 20px;
            border-radius: 15px;
            border-left: 3px solid #C6F221; /* Neon accent border */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        /* Metric Label */
        div[data-testid="metric-container"] label {
            color: #A0A0A0 !important; /* Lighter gray for labels */
        }
        /* Metric Value */
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #2C2C2C;
            border-radius: 8px;
            color: #A0A0A0;
            border: 1px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: transparent;
            color: #C6F221; /* Neon text for active tab */
            border: 1px solid #C6F221;
        }
        
        /* AI Insight Box */
        .ai-box {
            background: linear-gradient(135deg, #2C2C2C 0%, #1E1E1E 100%);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #C6F221;
            box-shadow: 0 0 15px rgba(198, 242, 33, 0.2); /* Neon glow */
        }
        .ai-box h4 { color: #C6F221; margin: 0 0 10px 0; }
        .ai-box p { color: #E0E0E0; margin: 0; }

        /* Adjusting header colors */
        h1, h2, h3 { color: #FFFFFF !important; }
        .stCaption { color: #A0A0A0 !important; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(symbol, start, end):
    """Fetch history and company info safely."""
    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(start=start, end=end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        try: info = ticker_obj.info
        except: info = {}
        try: news = ticker_obj.news
        except: news = []
        return df, info, news
    except Exception:
        return None, None, None

def calculate_technical_indicators(df):
    if len(df) > 50: df['SMA_50'] = df['Close'].rolling(window=50).mean()
    if len(df) > 200: df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

# --- SIDEBAR ---
with st.sidebar:
    # Using the green accent for the title to pop
    st.markdown("<h1 style='color: #C6F221;'>FOREX Terminal</h1>", unsafe_allow_html=True)
    
    symbol = st.text_input("Symbol", "EURUSD=X").upper()
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start", datetime.now() - timedelta(days=365))
    end_date = col2.date_input("End", datetime.now())
        
    st.markdown("---")
    st.subheader("Indicators")
    show_sma = st.checkbox("Show SMA (50 & 200)")
    show_ema = st.checkbox("Show EMA (20)")
    
    st.markdown("---")
    st.caption("v2.1 | Dark Mode Active 🟢")

# --- MAIN LOGIC ---
df, info, news = load_data(symbol, start_date, end_date)

if df is not None and not df.empty:
    df = calculate_technical_indicators(df)
    
    # --- HEADER ---
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        # Use shortName for forex/crypto if longName is missing
        name = info.get('longName', info.get('shortName', symbol))
        st.title(name)
        st.caption(f"Sector: {info.get('sector', 'Financial')} | Industry: {info.get('industry', 'Currency/Market')}")
    with col_h2:
        try:
            current = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            delta = current - prev
            # Custom formatting for Forex (more decimal places) vs Stocks
            fmt = ".4f" if "USD" in symbol or "=X" in symbol else ".2f"
            st.metric("Current Price", f"${current:{fmt}}", f"{delta:{fmt}}")
        except:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")

    # --- METRICS ROW ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Day High", f"${df['High'].iloc[-1]:.4f}")
    m2.metric("Day Low", f"${df['Low'].iloc[-1]:.4f}")
    # Handle missing 52W data gracefully
    hi_52 = info.get('fiftyTwoWeekHigh', df['High'].max())
    lo_52 = info.get('fiftyTwoWeekLow', df['Low'].min())
    m3.metric("52W High", f"${hi_52:.4f}")
    m4.metric("Volume", f"{df['Volume'].iloc[-1]:,}")

    st.markdown("---")

    # --- TABS ---
    tab_chart, tab_news, tab_data = st.tabs(["📈 Chart", "📰 News", "📊 Data"])

    # TAB 1: CHART
    with tab_chart:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price',
                                     increasing_line_color='#C6F221', decreasing_line_color='#FF3B30')) # Neon green for up candles

        if show_sma and 'SMA_50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#00B8D4', width=1), name='SMA 50'))
        if show_sma and 'SMA_200' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#FFD600', width=1), name='SMA 200'))
        if show_ema:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='#AB47BC', width=1), name='EMA 20'))

        # Update chart layout for dark theme
        fig.update_layout(
            height=500,
            template="plotly_dark", # Use Plotly's built-in dark theme
            paper_bgcolor='#2C2C2C',
            plot_bgcolor='#2C2C2C',
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor='#444444'),
            yaxis=dict(showgrid=True, gridcolor='#444444')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Insight Box
        last_close = df['Close'].iloc[-1]
        sma_200 = df['SMA_200'].iloc[-1] if 'SMA_200' in df.columns and not pd.isna(df['SMA_200'].iloc[-1]) else last_close
        trend = "BULLISH 🟢" if last_close > sma_200 else "BEARISH 🔴"
        
        st.markdown(f"""
        <div class="ai-box">
            <h4>🤖 AI Trend Analysis</h4>
            <p>Based on the 200-day moving average, the long-term trend for <b>{symbol}</b> is currently <b>{trend}</b>. 
            Monitor for price action near key support/resistance levels indicated on the chart.</p>
        </div>
        """, unsafe_allow_html=True)

    # TAB 2: NEWS
    with tab_news:
        st.subheader(f"Latest News: {symbol}")
        if news:
            count = 0
            for article in news:
                if count >= 5: break
                title = article.get('title', '')
                link = article.get('link', '#')
                publisher = article.get('publisher', 'Unknown')
                if title:
                    with st.container():
                        # Use the neon color for links
                        st.markdown(f"**[{title}]({link})**", unsafe_allow_html=True)
                        st.caption(f"Source: {publisher}")
                        st.markdown("---")
                    count += 1
        else:
            st.write("No specific news found.")

    # TAB 3: DATA
    with tab_data:
        st.subheader("Raw Price Data")
        csv = df.to_csv().encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name=f"{symbol}_data.csv", mime="text/csv")
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)

else:
    st.error("Data not found. Try a different symbol (e.g., EURUSD=X, BTC-USD, AAPL).")
