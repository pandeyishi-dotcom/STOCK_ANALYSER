import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# --- NLTK SETUP ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', download_dir=nltk_data_path)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FOREX | Pro Terminal",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #121212; color: #FFFFFF; }
        [data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #333; }
        .neon-text { color: #C6F221; font-weight: bold; }
        
        /* Custom Metric Card */
        div[data-testid="metric-container"] {
            background-color: #1E1E1E;
            border-left: 4px solid #C6F221;
            padding: 10px;
            border-radius: 5px;
        }
        
        /* Ticker Tape */
        .ticker-wrap {
            width: 100%; overflow: hidden; background-color: #000;
            padding: 10px 0; white-space: nowrap; border-bottom: 1px solid #333;
        }
        .ticker { display: inline-block; animation: marquee 40s linear infinite; }
        .ticker-item { display: inline-block; padding: 0 2rem; font-size: 1.1rem; color: #C6F221; font-family: monospace; }
        @keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_ticker_tape():
    try:
        tickers = ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'BTC-USD', 'ETH-USD', 'GC=F']
        data = yf.download(tickers, period="1d", interval="1d", progress=False)['Close'].iloc[-1]
        html = ""
        if isinstance(data, pd.Series):
             for t, p in data.items(): html += f"<span class='ticker-item'>{t}: ${p:,.4f}</span>"
        else: html = f"<span class='ticker-item'>Loading Market Data...</span>"
        return f"<div class='ticker-wrap'><div class='ticker'>{html}{html}</div></div>"
    except: return ""

@st.cache_data
def get_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

def analyze_sentiment(news_list):
    if not news_list: return 0, "Neutral ⚪"
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(a.get('title',''))['compound'] for a in news_list]
    avg = sum(scores) / len(scores) if scores else 0
    if avg > 0.05: return avg, "Bullish 🟢"
    elif avg < -0.05: return avg, "Bearish 🔴"
    else: return avg, "Neutral ⚪"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 class='neon-text'>FOREX Terminal</h1>", unsafe_allow_html=True)
    symbol = st.text_input("Symbol", "EURUSD=X").upper()
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start", datetime.now() - timedelta(days=365))
    end_date = col2.date_input("End", datetime.now())
    st.markdown("### 🛠 Tools")
    show_rsi = st.checkbox("Show RSI (14)", value=True)
    compare_sym = st.text_input("Compare (e.g. GBPUSD=X)")

st.markdown(get_ticker_tape(), unsafe_allow_html=True)

# --- MAIN LOGIC ---
df = get_data(symbol, start_date, end_date)
ticker_obj = yf.Ticker(symbol)

if df is not None and not df.empty:
    try: info = ticker_obj.info
    except: info = {}
    try: news = ticker_obj.news
    except: news = []
    
    # DETECT ASSET TYPE
    # If quoteType is missing, assume it's NOT a stock (safest default)
    quote_type = info.get('quoteType', 'CURRENCY') 
    is_stock = quote_type == 'EQUITY'
    
    # 1. HEADER
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title(info.get('shortName', symbol))
        st.caption(f"Type: {quote_type} | Sector: {info.get('sector', 'N/A')}")
    with c2:
        score, label = analyze_sentiment(news)
        st.metric("AI Sentiment", label, f"{score:.2f}")

    # 2. TABS (ADAPTIVE)
    # Only insert 'Fundamentals' tab if it's actually a stock
    tabs = ["📈 Chart", "📰 News & AI"]
    if is_stock: tabs.insert(1, "🏗 Fundamentals")
    
    active_tabs = st.tabs(tabs)
    
    # TAB: CHART (Index 0)
    with active_tabs[0]:
        rows = 2 if show_rsi else 1
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3] if show_rsi else [1])
        
        # Main Price
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                     name="Price", increasing_line_color='#C6F221', decreasing_line_color='#FF3B30'), row=1, col=1)
        
        # Comparison
        if compare_sym:
            comp_df = get_data(compare_sym, start_date, end_date)
            if comp_df is not None:
                fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Close'], line=dict(color='white', width=1, dash='dot'),
                                         name=f"Vs {compare_sym}"), row=1, col=1)

        # RSI
        if show_rsi:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#AB47BC', width=2), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#1E1E1E')
        st.plotly_chart(fig, use_container_width=True)

    # TAB: FUNDAMENTALS (Only if Stock)
    if is_stock:
        with active_tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Balance Sheet")
                # Show last 2 reports
                st.dataframe(ticker_obj.balance_sheet.iloc[:, :2], height=300, use_container_width=True)
            with c2:
                st.subheader("Insider Trading")
                st.dataframe(ticker_obj.insider_transactions.head(10), height=300, use_container_width=True)
            
            st.markdown("---")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
            k2.metric("Market Cap", f"${info.get('marketCap', 0):,}")
            k3.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%")
            k4.metric("Rev Growth", f"{info.get('revenueGrowth', 0)*100:.2f}%")

    # TAB: NEWS (Always Last)
    news_tab_index = 2 if is_stock else 1
    with active_tabs[news_tab_index]:
        if news:
            for article in news[:7]:
                score = SentimentIntensityAnalyzer().polarity_scores(article.get('title',''))['compound']
                mood = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
                with st.container():
                    st.markdown(f"**{mood} [{article.get('title')}]({article.get('link')})**")
                    st.caption(f"Source: {article.get('publisher')} | Score: {score:.2f}")
                    st.markdown("---")
        else:
            st.warning("No specific news found for this symbol (Common for Crypto/Forex).")

else:
    st.error("Data not found. Try 'AAPL' for Stocks or 'BTC-USD' for Crypto.")
