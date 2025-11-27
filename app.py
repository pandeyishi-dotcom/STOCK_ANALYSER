import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from fpdf import FPDF
import numpy as np

# --- NLTK SETUP ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Define local path for NLTK data
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
    page_title="FinTerminal India | AI Research",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (DARK THEME) ---
st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Roboto', sans-serif; }
        [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
        h1, h2, h3 { color: #C6F221 !important; }
        .stCaption { color: #8B949E !important; }
        hr { border: 0; border-top: 1px solid #30363D; }
        
        /* Neon Accents */
        .neon-text { color: #C6F221; font-weight: bold; text-shadow: 0 0 5px rgba(198, 242, 33, 0.5); }
        
        /* Metric Card */
        div[data-testid="metric-container"] {
            background-color: #161B22;
            border-left: 4px solid #C6F221;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        /* Report Generator Styling */
        .report-container {
            background-color: #161B22;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #30363D;
            margin-bottom: 20px;
        }
        .rating-badge { padding: 8px 20px; border-radius: 20px; font-weight: 900; font-size: 1.1rem; display: inline-block; }
        .buy { background-color: #238636; color: white; }
        .sell { background-color: #DA3633; color: white; }
        .hold { background-color: #D29922; color: black; }
        
        /* Ticker Tape */
        .ticker-wrap {
            width: 100%; overflow: hidden; background-color: #000;
            padding: 8px 0; white-space: nowrap; border-bottom: 2px solid #C6F221;
        }
        .ticker { display: inline-block; animation: marquee 45s linear infinite; }
        .ticker-item { display: inline-block; padding: 0 2rem; font-size: 1rem; color: #C6F221; font-family: 'Courier New', monospace; }
        @keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }
    </style>
""", unsafe_allow_html=True)

# =========================================
# --- HELPER: INDIAN MARKET MAPPING ---
# =========================================
# This dictionary maps easy names to Ticker Symbols
INDIAN_MARKET_MAP = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "State Bank of India (SBI)": "SBIN.NS",
    "Hindustan Unilever (HUL)": "HINDUNILVR.NS",
    "ITC Limited": "ITC.NS",
    "Larsen & Toubro (L&T)": "LT.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Axis Bank": "AXISBANK.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Titan Company": "TITAN.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Wipro": "WIPRO.NS",
    "Coal India": "COALINDIA.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Zomato": "ZOMATO.NS",
    "Paytm (One97)": "PAYTM.NS",
    "Ola Electric": "OLAELEC.NS",
    "Vodafone Idea": "IDEA.NS",
    "Yes Bank": "YESBANK.NS"
}

def get_currency_symbol(ticker):
    """Returns ₹ for Indian stocks, $ for others."""
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "₹"
    return "$"

# =========================================
# --- LOGIC ENGINES ---
# =========================================

class ResearchEngine:
    def __init__(self, df, info, currency_sym):
        self.df = df
        self.info = info
        self.currency = currency_sym
        self.close = df['Close'].iloc[-1] if not df.empty else 0
        self.sma200 = df['Close'].rolling(200).mean().iloc[-1] if not df.empty else 0
    
    def calculate_dcf(self):
        try:
            eps = self.info.get('trailingEps')
            growth_rate = self.info.get('earningsGrowth', 0.10) # Assume 10% if missing
            terminal_pe = 15 
            if eps is None or eps <= 0: return 0
            future_eps = eps * ((1 + growth_rate) ** 5)
            fair_value = future_eps * terminal_pe
            return fair_value
        except: return 0

    def get_rating(self, intrinsic_value):
        score = 0
        if self.close > self.sma200: score += 1
        if self.info.get('forwardPE', 100) < 30: score += 0.5
        if self.info.get('profitMargins', 0) > 0.10: score += 0.5
        if intrinsic_value > 0 and self.close < intrinsic_value * 0.9: score += 2
        
        if score >= 3: return "STRONG BUY", "buy"
        elif score >= 1.5: return "BUY", "buy"
        elif score >= 0.5: return "HOLD", "hold"
        else: return "SELL", "sell"

    def generate_thesis(self):
        reasons = []
        pe = self.info.get('trailingPE')
        if pe: reasons.append(f"Trading at a P/E of {pe:.1f}x.")
        if self.close > self.sma200: reasons.append("Technically in an uptrend (Above 200 DMA).")
        else: reasons.append("Technically in a downtrend (Below 200 DMA).")
        margins = self.info.get('profitMargins', 0)
        reasons.append(f"Net Profit Margin is {margins*100:.1f}%.")
        return " ".join(reasons)

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Equity Research Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, f"  {title}", 0, 1, 'L', 1)
        self.ln(4)
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

# =========================================
# --- HELPER FUNCTIONS ---
# =========================================
@st.cache_data(ttl=3600)
def get_ticker_tape_data():
    try:
        # Focusing on Indian Indices for tape
        tickers = ['^NSEI', '^BSESN', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INR=X']
        data = yf.download(tickers, period="1d", interval="1m", progress=False)['Close'].iloc[-1]
        html = ""
        if isinstance(data, pd.Series):
             for t, p in data.items(): 
                 # Pretty names
                 name = "NIFTY 50" if t == '^NSEI' else "SENSEX" if t == '^BSESN' else "USD/INR" if t == 'INR=X' else t.replace('.NS','')
                 html += f"<span class='ticker-item'>{name}: {p:,.2f}</span>"
        return f"<div class='ticker-wrap'><div class='ticker'>{html}{html}</div></div>"
    except: return ""

@st.cache_data
def load_historical_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

def analyze_sentiment_vader(news_list):
    if not news_list: return 0, "Neutral ⚪"
    sia = SentimentIntensityAnalyzer()
    scores = []
    for article in news_list:
        title = article.get('title')
        if not title: continue
        scores.append(sia.polarity_scores(title)['compound'])
    avg = np.mean(scores) if scores else 0
    if avg >= 0.05: return avg, "Bullish 🟢"
    elif avg <= -0.05: return avg, "Bearish 🔴"
    else: return avg, "Neutral ⚪"

def create_pdf_bytes(ticker, info, rating, thesis, intrinsic_val, currency_sym):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f"{info.get('longName', ticker)}", 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Price: {currency_sym}{info.get('currentPrice', 'N/A')} | Recommendation: {rating}", 0, 1, 'L')
    if intrinsic_val > 0:
        pdf.cell(0, 8, f"Intrinsic Value (DCF): {currency_sym}{intrinsic_val:.2f}", 0, 1, 'L')
    pdf.ln(5)
    pdf.chapter_title("Investment Thesis")
    pdf.chapter_body(thesis)
    pdf.chapter_title("Key Fundamentals")
    pdf.set_font('Courier', '', 10)
    metrics = [
        f"P/E Ratio: {info.get('trailingPE', 'N/A')}",
        f"ROE:       {info.get('returnOnEquity', 0)*100:.2f}%",
        f"Profit Mgn:{info.get('profitMargins', 0)*100:.2f}%",
    ]
    for m in metrics: pdf.cell(0, 5, m, 0, 1)
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 5, "Disclaimer: Automated report. Not financial advice.")
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# =========================================
# --- MAIN APP UI ---
# =========================================

with st.sidebar:
    st.markdown("<h1 class='neon-text'>FinTerminal India</h1>", unsafe_allow_html=True)
    mode = st.radio("Mode:", ["📊 Dashboard", "📑 Report Gen"], label_visibility="collapsed")
    st.markdown("---")
    
    # --- FIX 1: Searchable Dropdown for Indian Stocks ---
    st.markdown("### 🔍 Select Company")
    search_mode = st.checkbox("Manual Ticker Search", value=False)
    
    if search_mode:
        symbol = st.text_input("Enter Ticker (e.g. RELIANCE.NS)", "RELIANCE.NS").upper()
    else:
        # Dropdown with names
        selected_name = st.selectbox("Popular Stocks", options=list(INDIAN_MARKET_MAP.keys()))
        symbol = INDIAN_MARKET_MAP[selected_name]
        st.caption(f"Ticker: {symbol}")

    # Detect Currency Symbol
    currency_sym = get_currency_symbol(symbol)
    
    if mode == "📊 Dashboard":
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start", datetime.now() - timedelta(days=365))
        end_date = col2.date_input("End", datetime.now())
        st.markdown("### 📈 Settings")
        show_rsi = st.checkbox("Show RSI", value=True)
        show_sma = st.checkbox("Show SMA", value=True)
        
    else: # Report Mode
        generate_btn = st.button("🚀 Generate Report", type="primary")

# Top Ticker Tape
st.markdown(get_ticker_tape_data(), unsafe_allow_html=True)

ticker_obj = yf.Ticker(symbol)
try: info = ticker_obj.info
except: info = {}
try: news = ticker_obj.news
except: news = []

# =========================================
# DASHBOARD MODE
# =========================================
if mode == "📊 Dashboard":
    df = load_historical_data(symbol, start_date, end_date)
    if df is not None and not df.empty:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title(info.get('shortName', symbol))
            st.caption(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
        with c2:
            # --- FIX 2: CURRENCY IN METRICS ---
            current_price = df['Close'].iloc[-1]
            delta = current_price - df['Close'].iloc[-2]
            st.metric("Current Price", f"{currency_sym}{current_price:,.2f}", f"{delta:.2f}")

        # Charts
        tab1, tab2, tab3 = st.tabs(["📈 Chart", "🏗 Fundamentals", "📰 News"])
        
        with tab1:
            fig = make_subplots(rows=2 if show_rsi else 1, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3] if show_rsi else [1])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=symbol, increasing_line_color='#C6F221', decreasing_line_color='#FF3B30'), row=1, col=1)
            if show_sma:
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), line=dict(color='#FFC107', width=1), name='200 DMA'), row=1, col=1)
            if show_rsi:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#AB47BC', width=2), name="RSI"), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.update_layout(template="plotly_dark", height=600, paper_bgcolor='#161B22', plot_bgcolor='#161B22')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Market Cap", f"{currency_sym}{info.get('marketCap', 0)/1e7:,.0f} Cr")
            c2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            c3.metric("52W High", f"{currency_sym}{info.get('fiftyTwoWeekHigh', 0):,.2f}")
            st.markdown("---")
            st.subheader("Balance Sheet")
            try: st.dataframe(ticker_obj.balance_sheet.iloc[:, :2], use_container_width=True)
            except: st.info("Data unavailable")

        with tab3:
            if news:
                valid = 0
                for article in news:
                    title = article.get('title')
                    link = article.get('link')
                    if not title or not link: continue
                    if valid >= 7: break
                    score = SentimentIntensityAnalyzer().polarity_scores(title)['compound']
                    mood = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
                    st.markdown(f"{mood} **[{title}]({link})**")
                    st.divider()
                    valid += 1
                if valid == 0: st.info("No text news found.")
            else: st.warning("No news found.")
    else: st.error("Data not found.")

# =========================================
# REPORT MODE
# =========================================
elif mode == "📑 Report Gen":
    if generate_btn:
        with st.spinner("Analyzing..."):
            df_rep = load_historical_data(symbol, datetime.now()-timedelta(days=400), datetime.now())
            if df_rep is not None:
                engine = ResearchEngine(df_rep, info, currency_sym)
                ival = engine.calculate_dcf()
                rating, r_cls = engine.get_rating(ival)
                thesis = engine.generate_thesis()
                
                st.markdown(f"""
                <div class="report-container">
                    <h1>{info.get('longName', symbol)}</h1>
                    <span class="rating-badge {r_cls}">{rating}</span>
                    <hr>
                    <div style="display:flex; justify-content:space-between;">
                        <div>Current Price<br><b>{currency_sym}{info.get('currentPrice',0):,.2f}</b></div>
                        <div>Intrinsic Value (DCF)<br><b>{currency_sym}{ival:,.2f}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns([2,1])
                with c1:
                    st.markdown('<div class="report-container">', unsafe_allow_html=True)
                    st.subheader("Investment Thesis")
                    st.write(thesis)
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="report-container">', unsafe_allow_html=True)
                    st.subheader("Key Ratios")
                    st.write(f"**P/E:** {info.get('trailingPE', 'N/A')}")
                    st.write(f"**ROE:** {info.get('returnOnEquity', 0)*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                pdf_data = create_pdf_bytes(symbol, info, rating, thesis, ival, currency_sym)
                st.download_button("Download PDF", pdf_data, f"{symbol}_Report.pdf", "application/pdf", type='primary')
