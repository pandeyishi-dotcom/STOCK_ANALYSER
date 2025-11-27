import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from fpdf import FPDF
import numpy as np

# --- NLTK SETUP ( robust for cloud environments ) ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Define local path for NLTK data to avoid permission errors
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
    page_title="FinTerminal India",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* Global Dark Theme */
        .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Roboto', sans-serif; }
        [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
        
        /* Headers & Text */
        h1, h2, h3 { color: #C6F221 !important; }
        .stCaption { color: #8B949E !important; }
        hr { border: 0; border-top: 1px solid #30363D; }
        .neon-text { color: #C6F221; font-weight: bold; text-shadow: 0 0 5px rgba(198, 242, 33, 0.5); }
        
        /* Custom Components */
        div[data-testid="metric-container"] {
            background-color: #161B22; border-left: 4px solid #C6F221;
            padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .report-container {
            background-color: #161B22; padding: 25px; border-radius: 12px;
            border: 1px solid #30363D; margin-bottom: 20px;
        }
        .rating-badge { padding: 8px 20px; border-radius: 20px; font-weight: 900; font-size: 1.1rem; display: inline-block; }
        .buy { background-color: #238636; color: white; }
        .sell { background-color: #DA3633; color: white; }
        .hold { background-color: #D29922; color: black; }
        
        /* Ticker Tape Animation */
        .ticker-wrap {
            width: 100%; overflow: hidden; background-color: #000;
            padding: 8px 0; white-space: nowrap; border-bottom: 2px solid #C6F221;
        }
        .ticker { display: inline-block; animation: marquee 45s linear infinite; }
        .ticker-item { display: inline-block; padding: 0 2rem; font-size: 1rem; color: #C6F221; font-family: 'Courier New', monospace; }
        @keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }
    </style>
""", unsafe_allow_html=True)

# --- MARKET MAPPING (Common Indian Stocks) ---
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
    "Mahindra & Mahindra": "M&M.NS",
    "Wipro": "WIPRO.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Zomato": "ZOMATO.NS",
    "Paytm": "PAYTM.NS",
    "Vodafone Idea": "IDEA.NS",
    "Yes Bank": "YESBANK.NS"
}

# --- HELPER FUNCTIONS ---

def get_currency_symbol(ticker):
    """Detect if INR or USD based on ticker suffix."""
    if ticker.endswith(".NS") or ticker.endswith(".BO"): return "₹"
    return "$"

def sanitize_text(text):
    """Cleans text for PDF generation to prevent encoding errors."""
    if not isinstance(text, str): return str(text)
    # Replace Rupee symbol with 'Rs.' (Latin-1 doesn't support ₹)
    text = text.replace("₹", "Rs. ")
    # Replace common problematic chars
    text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-")
    return text.encode('latin-1', 'replace').decode('latin-1')

@st.cache_data(ttl=3600)
def get_ticker_tape_data():
    """Fetches market indices for the top banner, handles NaNs."""
    try:
        tickers = ['^NSEI', '^BSESN', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        # Download 1 day data
        data = yf.download(tickers, period="1d", interval="1m", progress=False)['Close'].iloc[-1]
        html = ""
        if isinstance(data, pd.Series):
             for t, p in data.items(): 
                 # Formatting Name
                 name = "NIFTY 50" if '^NSEI' in t else "SENSEX" if '^BSESN' in t else t.replace('.NS','')
                 # Formatting Price (Handle NaN)
                 if pd.isna(p):
                     price_display = "N/A"
                 else:
                     price_display = f"{p:,.2f}"
                 
                 html += f"<span class='ticker-item'>{name}: {price_display}</span>"
        return f"<div class='ticker-wrap'><div class='ticker'>{html}{html}</div></div>"
    except: return ""

@st.cache_data
def load_historical_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        # Flatten MultiIndex columns (Fix for yfinance update)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

# --- LOGIC ENGINES ---

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
            growth = self.info.get('earningsGrowth', 0.10)
            if eps is None or eps <= 0: return 0
            # Simple DCF: EPS * (1+g)^5 * Terminal Multiple (15)
            return eps * ((1 + growth) ** 5) * 15
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
        if pe: reasons.append(f"Current P/E Ratio is {pe:.1f}x.")
        
        if self.close > self.sma200: reasons.append("Stock is technically in an Uptrend (Above 200 DMA).")
        else: reasons.append("Stock is technically in a Downtrend (Below 200 DMA).")
        
        margins = self.info.get('profitMargins', 0)
        reasons.append(f"Net Profit Margin remains at {margins*100:.1f}%.")
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

def create_pdf_bytes(ticker, info, rating, thesis, intrinsic_val, currency_sym):
    pdf = PDFReport()
    pdf.add_page()
    
    # Sanitize Currency for PDF (Convert ₹ -> Rs. to avoid crash)
    safe_curr = "Rs. " if currency_sym == "₹" else "$"
    
    # Title & Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, sanitize_text(f"{info.get('longName', ticker)}"), 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    price_txt = f"Price: {safe_curr}{info.get('currentPrice', 'N/A')} | Rec: {rating}"
    pdf.cell(0, 8, sanitize_text(price_txt), 0, 1, 'L')
    
    if intrinsic_val > 0:
        val_txt = f"Intrinsic Value (DCF): {safe_curr}{intrinsic_val:.2f}"
        pdf.cell(0, 8, sanitize_text(val_txt), 0, 1, 'L')
    pdf.ln(5)
    
    # Content
    pdf.chapter_title("Investment Thesis")
    pdf.chapter_body(sanitize_text(thesis))
    
    pdf.chapter_title("Key Fundamentals")
    pdf.set_font('Courier', '', 10)
    metrics = [
        f"P/E Ratio: {info.get('trailingPE', 'N/A')}",
        f"ROE:       {info.get('returnOnEquity', 0)*100:.2f}%",
        f"Profit Mgn:{info.get('profitMargins', 0)*100:.2f}%",
    ]
    for m in metrics: pdf.cell(0, 5, sanitize_text(m), 0, 1)
    
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 5, "Disclaimer: Automated report generated by AI. Not financial advice.")
    
    return pdf.output(dest='S').encode('latin-1', 'replace')

# =========================================
# --- MAIN APP UI ---
# =========================================

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 class='neon-text'>FinTerminal India</h1>", unsafe_allow_html=True)
    mode = st.radio("Mode:", ["📊 Dashboard", "📑 Report Gen"], label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("### 🔍 Select Company")
    search_mode = st.checkbox("Manual Ticker Search", value=False)
    
    if search_mode:
        symbol = st.text_input("Enter Ticker", "RELIANCE.NS").upper()
    else:
        # Dropdown for easier user experience
        selected_name = st.selectbox("Popular Stocks", options=list(INDIAN_MARKET_MAP.keys()))
        symbol = INDIAN_MARKET_MAP[selected_name]
        st.caption(f"Ticker: {symbol}")
    
    currency_sym = get_currency_symbol(symbol)
    
    if mode == "📊 Dashboard":
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start", datetime.now() - timedelta(days=365))
        end_date = col2.date_input("End", datetime.now())
        st.markdown("### 📈 Settings")
        show_rsi = st.checkbox("Show RSI", value=True)
        show_sma = st.checkbox("Show SMA", value=True)
    else:
        generate_btn = st.button("🚀 Generate Report", type="primary")

# Display Ticker Tape
st.markdown(get_ticker_tape_data(), unsafe_allow_html=True)

# Fetch Data Live
ticker_obj = yf.Ticker(symbol)
try: info = ticker_obj.info
except: info = {}
try: news = ticker_obj.news
except: news = []

# =========================================
# MODE 1: DASHBOARD
# =========================================
if mode == "📊 Dashboard":
    df = load_historical_data(symbol, start_date, end_date)
    
    if df is not None and not df.empty:
        # Header
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title(info.get('shortName', symbol))
            st.caption(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
        with c2:
            # Price Metric
            current_price = df['Close'].iloc[-1]
            delta = current_price - df['Close'].iloc[-2]
            st.metric("Price", f"{currency_sym}{current_price:,.2f}", f"{delta:.2f}")

        # Tabs
        t1, t2, t3 = st.tabs(["📈 Chart", "🏗 Fundamentals", "📰 News"])
        
        # TAB 1: CHART
        with t1:
            fig = make_subplots(rows=2 if show_rsi else 1, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3] if show_rsi else [1])
            # Candlestick
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=symbol, increasing_line_color='#C6F221', decreasing_line_color='#FF3B30'), row=1, col=1)
            # SMA
            if show_sma: 
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), line=dict(color='#FFC107', width=1), name='200 DMA'), row=1, col=1)
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
            fig.update_layout(template="plotly_dark", height=600, paper_bgcolor='#161B22', plot_bgcolor='#161B22')
            st.plotly_chart(fig, use_container_width=True)
            
        # TAB 2: FUNDAMENTALS
        with t2:
            st.subheader("Balance Sheet")
            try: st.dataframe(ticker_obj.balance_sheet.iloc[:, :2], use_container_width=True)
            except: st.info("Balance Sheet data unavailable")
        
        # TAB 3: NEWS (With Filter Fix)
        with t3:
            if news:
                valid_articles = 0
                for article in news:
                    title = article.get('title')
                    link = article.get('link')
                    publisher = article.get('publisher', 'Unknown')
                    
                    # Skip empty articles
                    if not title or not link: continue
                    if valid_articles >= 10: break

                    # Sentiment
                    try:
                        score = SentimentIntensityAnalyzer().polarity_scores(title)['compound']
                        mood = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
                    except: 
                        mood = "⚪"; score = 0.0

                    st.markdown(f"{mood} **[{title}]({link})**")
                    st.caption(f"Source: {publisher} | AI Score: {score:.2f}")
                    st.divider()
                    valid_articles += 1
                
                # Fallback if Yahoo returns only bad data
                if valid_articles == 0:
                    st.warning("Yahoo Finance returned incomplete data.")
                    google_url = f"https://www.google.com/search?q={symbol}+stock+news&tbm=nws"
                    st.markdown(f"👉 **[Click here for Google News: {symbol}]({google_url})**")
            else:
                st.info("No news feed available.")
                google_url = f"https://www.google.com/search?q={symbol}+stock+news&tbm=nws"
                st.markdown(f"👉 **[Click here for Google News: {symbol}]({google_url})**")
                
    else: st.error("No historical data found. Please check the ticker.")

# =========================================
# MODE 2: REPORT GEN
# =========================================
elif mode == "📑 Report Gen":
    if generate_btn:
        with st.spinner("Analyzing..."):
            df_rep = load_historical_data(symbol, datetime.now()-timedelta(days=400), datetime.now())
            if df_rep is not None:
                # Run Engines
                engine = ResearchEngine(df_rep, info, currency_sym)
                ival = engine.calculate_dcf()
                rating, r_cls = engine.get_rating(ival)
                thesis = engine.generate_thesis()
                
                # Report UI
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
                    st.markdown(f"**Investment Thesis:** {thesis}")
                with c2:
                    st.write(f"**P/E:** {info.get('trailingPE', 'N/A')}")
                    st.write(f"**ROE:** {info.get('returnOnEquity', 0)*100:.2f}%")
                
                # PDF Generation
                try:
                    pdf_data = create_pdf_bytes(symbol, info, rating, thesis, ival, currency_sym)
                    st.download_button("Download PDF Report", pdf_data, f"{symbol}_Report.pdf", "application/pdf", type='primary')
                except Exception as e:
                    st.error(f"Could not generate PDF: {e}")
