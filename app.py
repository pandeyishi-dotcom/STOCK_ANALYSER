import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- SETUP NLTK ---
# Download VADER lexicon for sentiment analysis (only runs once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

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
        /* Global & Dark Theme */
        .stApp { background-color: #121212; color: #FFFFFF; }
        [data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #333; }
        
        /* Neon Accents */
        .neon-text { color: #C6F221; font-weight: bold; }
        .metric-card {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #C6F221;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        /* Ticker Tape Animation */
        .ticker-wrap {
            width: 100%;
            overflow: hidden;
            background-color: #000;
            padding: 10px 0;
            white-space: nowrap;
            box-sizing: border-box;
        }
        .ticker {
            display: inline-block;
            animation: marquee 30s linear infinite;
        }
        .ticker-item {
            display: inline-block;
            padding: 0 2rem;
            font-size: 1.2rem;
            color: #C6F221;
        }
        @keyframes marquee {
            0%   { transform: translate(0, 0); }
            100% { transform: translate(-100%, 0); }
        }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_ticker_tape():
    """Fetches simple live prices for a marquee."""
    # List of major pairs to scroll
    tickers = ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'BTC-USD', 'ETH-USD', 'GC=F', 'CL=F']
    data = yf.download(tickers, period="1d", interval="1d", progress=False)['Close'].iloc[-1]
    
    html_content = ""
    for ticker, price in data.items():
        symbol = ticker if isinstance(ticker, str) else ticker[0] # Handle MultiIndex if present
        val = price.iloc[0] if isinstance(price, pd.Series) else price
        html_content += f"<span class='ticker-item'>{symbol}: ${val:,.4f}</span>"
        
    return f"""
    <div class="ticker-wrap">
        <div class="ticker">
            {html_content} {html_content}
        </div>
    </div>
    """

@st.cache_data
def load_data(symbol, start, end):
    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(start=start, end=end)
        
        # Safe handling of different yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df, ticker_obj
    except Exception as e:
        return None, None

def analyze_sentiment(news_list):
    """Analyze news titles using NLTK VADER."""
    if not news_list: return 0, "Neutral"
    
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    
    for article in news_list:
        title = article.get('title', '')
        score = analyzer.polarity_scores(title)['compound']
        scores.append(score)
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    if avg_score > 0.05: return avg_score, "Bullish 🟢"
    elif avg_score < -0.05: return avg_score, "Bearish 🔴"
    else: return avg_score, "Neutral ⚪"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 class='neon-text'>FOREX Terminal</h1>", unsafe_allow_html=True)
    symbol = st.text_input("Symbol", "EURUSD=X").upper()
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start", datetime.now() - timedelta(days=365))
    end_date = col2.date_input("End", datetime.now())
    
    st.markdown("### 🛠 Tools")
    show_rsi = st.checkbox("Show RSI (14)")
    compare_sym = st.text_input("Compare (e.g. GBPUSD=X)")

# --- TICKER TAPE ---
try:
    st.markdown(get_ticker_tape(), unsafe_allow_html=True)
except:
    st.warning("Ticker tape offline (Data Error)")

# --- MAIN LOGIC ---
df, ticker_obj = load_data(symbol, start_date, end_date)

if df is not None and not df.empty:
    info = ticker_obj.info
    news = ticker_obj.news
    
    # 1. HEADER & SENTIMENT
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title(info.get('shortName', symbol))
        st.caption(f"{info.get('sector', 'Unknown')} | {info.get('industry', 'Unknown')}")
    with c2:
        # Sentiment Analysis Calculation
        sent_score, sent_label = analyze_sentiment(news)
        st.metric("AI Sentiment Score", sent_label, f"{sent_score:.2f}")

    # 2. CHARTING
    st.markdown("---")
    tab_chart, tab_fund, tab_news = st.tabs(["📈 Technical Chart", "🏗 Fundamentals", "📰 News & AI"])

    with tab_chart:
        # Create Subplots (Row 1: Price, Row 2: RSI)
        rows = 2 if show_rsi else 1
        row_heights = [0.7, 0.3] if show_rsi else [1.0]
        
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            row_heights=row_heights)

        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                     low=df['Low'], close=df['Close'], name="Price",
                                     increasing_line_color='#C6F221', decreasing_line_color='#FF3B30'), row=1, col=1)

        # Comparison Line
        if compare_sym:
            comp_df, _ = load_data(compare_sym, start_date, end_date)
            if comp_df is not None:
                 fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Close'], 
                                          line=dict(color='white', width=1, dash='dot'), 
                                          name=f"Vs {compare_sym}"), row=1, col=1)

        # RSI Calculation & Plot
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

    with tab_fund:
        f1, f2 = st.columns(2)
        with f1:
            st.subheader("Balance Sheet")
            # Show standard balance sheet or handle error
            try:
                st.dataframe(ticker_obj.balance_sheet.T.head(5), height=300)
            except:
                st.info("Balance sheet data unavailable for this asset.")
        with f2:
            st.subheader("Insider Transactions")
            try:
                insiders = ticker_obj.insider_transactions
                if insiders is not None and not insiders.empty:
                    st.dataframe(insiders.head(10), height=300)
                else:
                    st.info("No insider transactions found.")
            except:
                st.info("Insider data unavailable.")
                
        st.subheader("Company Ratios")
        # Quick metrics
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
        r2.metric("PEG Ratio", info.get('pegRatio', 'N/A'))
        r3.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%")
        r4.metric("Debt to Equity", info.get('debtToEquity', 'N/A'))

    with tab_news:
        st.subheader("Market Sentiment Analysis")
        for article in news[:5]:
            title = article.get('title', 'No Title')
            link = article.get('link', '#')
            # Calculate score for individual article
            score = SentimentIntensityAnalyzer().polarity_scores(title)['compound']
            mood = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
            
            with st.container():
                st.markdown(f"**{mood} [{title}]({link})**")
                st.caption(f"Sentiment Score: {score} | Publisher: {article.get('publisher', 'Unknown')}")
                st.markdown("---")

else:
    st.error("Data not found. Please check symbol.")
