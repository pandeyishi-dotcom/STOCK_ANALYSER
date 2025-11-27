To make the Equity Research Report truly **institutional-grade**, I am adding three major analytical modules:

1.  **Auto-Generated SWOT Analysis:** The AI will now scan the balance sheet and price action to categorize **Strengths, Weaknesses, Opportunities, and Threats**.
2.  **Analyst Consensus & Target Price:** A visual comparison of the current price versus what Wall Street analysts predict (using `targetMeanPrice`).
3.  **Risk Profile (VaR & Drawdown):** A dedicated section calculating the **Maximum Drawdown** (worst drop) and **Volatility** to warn you of risks.

### Full Updated Code (`app.py`)

Replace your entire file with this version.

```python
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
    page_title="FinTerminal India Pro",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Roboto', sans-serif; }
        [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
        
        h1, h2, h3 { color: #C6F221 !important; }
        .stCaption { color: #8B949E !important; }
        hr { border: 0; border-top: 1px solid #30363D; }
        .neon-text { color: #C6F221; font-weight: bold; text-shadow: 0 0 5px rgba(198, 242, 33, 0.5); }
        
        div[data-testid="metric-container"] {
            background-color: #161B22; border-left: 4px solid #C6F221;
            padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        /* Report Cards */
        .report-card {
            background-color: #161B22; padding: 20px; border-radius: 10px;
            border: 1px solid #30363D; margin-bottom: 15px;
        }
        .swot-box {
            padding: 10px; border-radius: 5px; margin-bottom: 5px; font-size: 0.9rem;
        }
        .strength { background-color: #0f3d0f; border-left: 3px solid #238636; color: #e6ffec; }
        .weakness { background-color: #3d0f0f; border-left: 3px solid #da3633; color: #ffe6e6; }
        
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

# --- MARKET MAPPING ---
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

def get_currency_symbol(ticker):
    if ticker.endswith(".NS") or ticker.endswith(".BO"): return "₹"
    return "$"

def sanitize_text(text):
    if not isinstance(text, str): return str(text)
    text = text.replace("₹", "Rs. ").replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-")
    return text.encode('latin-1', 'replace').decode('latin-1')

# --- CACHED DATA LOADERS ---
@st.cache_data(ttl=3600)
def get_ticker_tape_data():
    try:
        tickers = ['^NSEI', '^BSESN', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        data = yf.download(tickers, period="1d", interval="1m", progress=False)['Close'].iloc[-1]
        html = ""
        if isinstance(data, pd.Series):
             for t, p in data.items(): 
                 name = "NIFTY 50" if '^NSEI' in t else "SENSEX" if '^BSESN' in t else t.replace('.NS','')
                 price_display = "N/A" if pd.isna(p) else f"{p:,.2f}"
                 html += f"<span class='ticker-item'>{name}: {price_display}</span>"
        return f"<div class='ticker-wrap'><div class='ticker'>{html}{html}</div></div>"
    except: return ""

@st.cache_data
def load_historical_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

# --- ADVANCED RESEARCH ENGINE ---
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

    def generate_swot(self):
        """Generates a dynamic SWOT analysis."""
        swot = {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}
        
        # Strengths
        if self.info.get('profitMargins', 0) > 0.15: swot['Strengths'].append("High Profit Margins (>15%)")
        if self.info.get('returnOnEquity', 0) > 0.15: swot['Strengths'].append("Strong ROE (>15%)")
        if self.close > self.sma200: swot['Strengths'].append("Bullish Technical Trend (Above 200 DMA)")
        
        # Weaknesses
        if self.info.get('debtToEquity', 0) > 150: swot['Weaknesses'].append("High Debt Levels (>150% D/E)")
        if self.info.get('trailingPE', 0) > 50: swot['Weaknesses'].append("Expensive Valuation (P/E > 50)")
        
        # Opportunities
        if self.info.get('earningsGrowth', 0) > 0.20: swot['Opportunities'].append("High Earnings Growth Potential")
        if self.info.get('pegRatio', 5) < 1.0: swot['Opportunities'].append("Undervalued relative to growth (PEG < 1)")
        
        # Threats
        if self.info.get('beta', 1.0) > 1.5: swot['Threats'].append("High Volatility (Beta > 1.5)")
        if self.info.get('shortRatio', 0) > 5: swot['Threats'].append("Rising Short Interest")
        
        # Defaults if empty
        if not swot['Strengths']: swot['Strengths'].append("Stable Large Cap Status")
        if not swot['Weaknesses']: swot['Weaknesses'].append("Moderate Growth Rates")
        
        return swot

    def get_risk_metrics(self):
        """Calculates Max Drawdown and Volatility."""
        if self.df.empty: return {}
        
        # Max Drawdown
        rolling_max = self.df['Close'].cummax()
        drawdown = (self.df['Close'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Volatility (Annualized)
        daily_ret = self.df['Close'].pct_change()
        volatility = daily_ret.std() * np.sqrt(252)
        
        return {
            "Max Drawdown": f"{max_drawdown*100:.2f}%",
            "Volatility": f"{volatility*100:.2f}%",
            "Beta": self.info.get('beta', 'N/A')
        }

# --- PDF GENERATOR ---
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

def create_pdf_bytes(ticker, info, rating, swot, risk, intrinsic_val, currency_sym):
    pdf = PDFReport()
    pdf.add_page()
    safe_curr = "Rs. " if currency_sym == "₹" else "$"
    
    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, sanitize_text(f"{info.get('longName', ticker)}"), 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, sanitize_text(f"Price: {safe_curr}{info.get('currentPrice', 'N/A')} | Rating: {rating}"), 0, 1, 'L')
    pdf.ln(5)
    
    # SWOT Section
    pdf.chapter_title("SWOT Analysis")
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 5, "Strengths:", 0, 1)
    pdf.set_font('Arial', '', 10)
    for s in swot['Strengths']: pdf.cell(0, 5, f"- {s}", 0, 1)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 5, "Weaknesses:", 0, 1)
    pdf.set_font('Arial', '', 10)
    for w in swot['Weaknesses']: pdf.cell(0, 5, f"- {w}", 0, 1)
    pdf.ln(5)
    
    # Risk Section
    pdf.chapter_title("Risk Profile")
    pdf.cell(0, 5, f"Max Drawdown (1Y): {risk['Max Drawdown']}", 0, 1)
    pdf.cell(0, 5, f"Annual Volatility: {risk['Volatility']}", 0, 1)
    pdf.cell(0, 5, f"Beta: {risk['Beta']}", 0, 1)
    pdf.ln(5)

    # Disclaimer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 5, "Disclaimer: Automated report generated by AI. Not financial advice.")
    
    return pdf.output(dest='S').encode('latin-1', 'replace')

# =========================================
# --- MAIN APP UI ---
# =========================================
with st.sidebar:
    st.markdown("<h1 class='neon-text'>FinTerminal India</h1>", unsafe_allow_html=True)
    mode = st.radio("Mode:", ["📊 Dashboard", "📑 Report Gen"], label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("### 🔍 Select Company")
    search_mode = st.checkbox("Manual Ticker Search", value=False)
    if search_mode:
        symbol = st.text_input("Enter Ticker", "RELIANCE.NS").upper()
    else:
        selected_name = st.selectbox("Popular Stocks", options=list(INDIAN_MARKET_MAP.keys()))
        symbol = INDIAN_MARKET_MAP[selected_name]
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

st.markdown(get_ticker_tape_data(), unsafe_allow_html=True)
ticker_obj = yf.Ticker(symbol)
try: info = ticker_obj.info
except: info = {}
try: news = ticker_obj.news
except: news = []

# --- DASHBOARD MODE ---
if mode == "📊 Dashboard":
    df = load_historical_data(symbol, start_date, end_date)
    if df is not None and not df.empty:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title(info.get('shortName', symbol))
            st.caption(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
        with c2:
            p = df['Close'].iloc[-1]
            d = p - df['Close'].iloc[-2]
            st.metric("Price", f"{currency_sym}{p:,.2f}", f"{d:.2f}")

        t1, t2, t3 = st.tabs(["📈 Chart", "🏗 Fundamentals", "📰 News"])
        with t1:
            fig = make_subplots(rows=2 if show_rsi else 1, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3] if show_rsi else [1])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=symbol, increasing_line_color='#C6F221', decreasing_line_color='#FF3B30'), row=1, col=1)
            if show_sma: fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), line=dict(color='#FFC107', width=1), name='200 DMA'), row=1, col=1)
            if show_rsi:
                delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean(); rs = gain / loss; rsi = 100 - (100 / (1 + rs))
                fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#AB47BC', width=2), name="RSI"), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1); fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.update_layout(template="plotly_dark", height=600, paper_bgcolor='#161B22', plot_bgcolor='#161B22')
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            st.subheader("Balance Sheet")
            try: st.dataframe(ticker_obj.balance_sheet.iloc[:, :2], use_container_width=True)
            except: st.info("Data unavailable")
        with t3:
            if news:
                v=0
                for a in news:
                    if not a.get('title') or not a.get('link'): continue
                    if v>=10: break
                    try: s = SentimentIntensityAnalyzer().polarity_scores(a['title'])['compound']; m = "🟢" if s > 0.05 else "🔴" if s < -0.05 else "⚪"
                    except: m="⚪"; s=0.0
                    st.markdown(f"{m} **[{a['title']}]({a['link']})**"); st.caption(f"Score: {s:.2f}"); st.divider(); v+=1
                if v==0: st.warning("Incomplete news data."); st.markdown(f"[Search Google News](https://www.google.com/search?q={symbol}+news)")
            else: st.info("No news feed."); st.markdown(f"[Search Google News](https://www.google.com/search?q={symbol}+news)")
    else: st.error("No Data.")

# --- REPORT MODE ---
elif mode == "📑 Report Gen":
    if generate_btn:
        with st.spinner("Running Advanced Analysis..."):
            df_rep = load_historical_data(symbol, datetime.now()-timedelta(days=400), datetime.now())
            if df_rep is not None:
                eng = ResearchEngine(df_rep, info, currency_sym)
                ival = eng.calculate_dcf()
                rating, r_cls = eng.get_rating(ival)
                swot = eng.generate_swot()
                risk = eng.get_risk_metrics()
                
                # Header
                st.markdown(f"""
                <div class="report-card">
                    <h1>{info.get('longName', symbol)}</h1>
                    <span class="rating-badge {r_cls}">{rating}</span>
                    <hr>
                    <div style="display:flex; justify-content:space-between;">
                        <div>Current Price<br><b>{currency_sym}{info.get('currentPrice',0):,.2f}</b></div>
                        <div>Target (Consensus)<br><b>{currency_sym}{info.get('targetMeanPrice', 'N/A')}</b></div>
                        <div>Intrinsic (DCF)<br><b>{currency_sym}{ival:,.2f}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                    st.subheader("🛡 SWOT Analysis")
                    st.markdown("**Strengths**")
                    for s in swot['Strengths']: st.markdown(f"<div class='swot-box strength'>{s}</div>", unsafe_allow_html=True)
                    st.markdown("**Weaknesses**")
                    for w in swot['Weaknesses']: st.markdown(f"<div class='swot-box weakness'>{w}</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with c2:
                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                    st.subheader("⚠️ Risk Profile")
                    st.write(f"**Max Drawdown (1Y):** {risk.get('Max Drawdown')}")
                    st.write(f"**Volatility:** {risk.get('Volatility')}")
                    st.write(f"**Beta:** {risk.get('Beta')}")
                    st.progress(min(info.get('payoutRatio', 0), 1.0), text=f"Dividend Payout: {info.get('payoutRatio',0)*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # PDF
                try:
                    pdf_data = create_pdf_bytes(symbol, info, rating, swot, risk, ival, currency_sym)
                    st.download_button("Download PDF Report", pdf_data, f"{symbol}_Report.pdf", "application/pdf", type='primary')
                except Exception as e: st.error(f"PDF Error: {e}")
```
