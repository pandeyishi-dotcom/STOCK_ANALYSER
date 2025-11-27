import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from fpdf import FPDF
import numpy as np
import requests
import xml.etree.ElementTree as ET
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

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
    page_title="FinTerminal India | Hedge Fund Edition",
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
        .macro-box { background-color: #161B22; border: 1px solid #30363D; padding: 10px; border-radius: 5px; margin-bottom: 10px; text-align: center; }
        .macro-val { font-size: 1.1rem; font-weight: bold; color: #FAFAFA; }
        .macro-lbl { font-size: 0.8rem; color: #8B949E; }
        div[data-testid="metric-container"] { background-color: #161B22; border-left: 4px solid #C6F221; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        .dataframe { font-size: 0.8rem !important; }
        .report-card { background-color: #161B22; padding: 25px; border-radius: 10px; border: 1px solid #30363D; margin-bottom: 15px; }
        .rating-badge { padding: 8px 20px; border-radius: 20px; font-weight: 900; font-size: 1.1rem; display: inline-block; }
        .buy { background-color: #238636; color: white; } .sell { background-color: #DA3633; color: white; } .hold { background-color: #D29922; color: black; }
        .eic-box { background-color: #1E2329; padding: 15px; border-radius: 5px; border-left: 3px solid #00BCD4; margin-bottom: 10px; }
        .challenge-box { background-color: #2B1616; padding: 15px; border-radius: 5px; border-left: 3px solid #DA3633; margin-bottom: 10px; }
        .impact-box { background-color: #132F13; padding: 15px; border-radius: 5px; border-left: 3px solid #238636; margin-bottom: 10px; }
        .ticker-wrap { width: 100%; overflow: hidden; background-color: #000; padding: 8px 0; white-space: nowrap; border-bottom: 2px solid #C6F221; }
        .ticker { display: inline-block; animation: marquee 45s linear infinite; }
        .ticker-item { display: inline-block; padding: 0 2rem; font-size: 1rem; color: #C6F221; font-family: 'Courier New', monospace; }
        @keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'portfolio' not in st.session_state: st.session_state.portfolio = []
generate_btn = False 

# --- MAPS ---
UI_MAP = {
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS", "SBI": "SBIN.NS",
    "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS", "Airtel": "BHARTIARTL.NS", "HUL": "HINDUNILVR.NS",
    "ITC": "ITC.NS", "L&T": "LT.NS", "Tata Motors": "TATAMOTORS.NS", "Axis Bank": "AXISBANK.NS",
    "Sun Pharma": "SUNPHARMA.NS", "Maruti": "MARUTI.NS", "Titan": "TITAN.NS", "Bajaj Fin": "BAJFINANCE.NS",
    "Asian Paints": "ASIANPAINT.NS", "M&M": "M&M.NS", "Wipro": "WIPRO.NS", "Tata Steel": "TATASTEEL.NS",
    "Zomato": "ZOMATO.NS", "Paytm": "PAYTM.NS", "Idea": "IDEA.NS", "Yes Bank": "YESBANK.NS"
}
STOCK_SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking", "SBIN.NS": "Banking",
    "ICICIBANK.NS": "Banking", "INFY.NS": "IT", "BHARTIARTL.NS": "Telecom", "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG", "LT.NS": "Infra", "TATAMOTORS.NS": "Auto", "AXISBANK.NS": "Banking",
    "SUNPHARMA.NS": "Pharma", "MARUTI.NS": "Auto", "TITAN.NS": "Consumer", "BAJFINANCE.NS": "Finance",
    "ASIANPAINT.NS": "Consumer", "M&M.NS": "Auto", "WIPRO.NS": "IT", "TATASTEEL.NS": "Metals",
    "ZOMATO.NS": "Tech", "PAYTM.NS": "Tech", "IDEA.NS": "Telecom", "YESBANK.NS": "Banking"
}

def get_currency_symbol(ticker):
    return "₹" if ticker.endswith((".NS", ".BO")) else "$"

def sanitize_text(text):
    if not isinstance(text, str): return str(text)
    return text.replace("₹", "Rs. ").encode('latin-1', 'replace').decode('latin-1')

# --- DATA HELPERS ---
@st.cache_data(ttl=300)
def fetch_google_news(symbol):
    try:
        search_term = symbol.replace('.NS', '')
        url = f"https://news.google.com/rss/search?q={search_term}+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        root = ET.fromstring(response.content)
        news_items = []
        for item in root.findall('./channel/item')[:10]:
            title = item.find('title').text.split(' - ')[0]
            news_items.append({'title': title, 'link': item.find('link').text, 'publisher': 'Google News'})
        return news_items
    except: return []

@st.cache_data(ttl=3600)
def get_macro_data():
    try:
        data = yf.download(['INR=X', 'CL=F', 'GC=F'], period="1d", progress=False)['Close'].iloc[-1]
        return data
    except: return None

@st.cache_data(ttl=3600)
def get_ticker_tape_data():
    try:
        data = yf.download(['^NSEI', '^BSESN', 'RELIANCE.NS', 'HDFCBANK.NS'], period="1d", interval="1m", progress=False)['Close'].iloc[-1]
        html = ""
        if isinstance(data, pd.Series):
             for t, p in data.items(): 
                 name = "NIFTY" if '^NSEI' in t else "SENSEX" if '^BSESN' in t else t.replace('.NS','')
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

# --- BLACK SCHOLES ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call': return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- AI ENGINES ---
def run_sector_analysis():
    sector_perf = {}
    for ticker, sector in STOCK_SECTOR_MAP.items():
        try:
            df = yf.download(ticker, period="2d", progress=False)
            if len(df) >= 2:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                if sector not in sector_perf: sector_perf[sector] = []
                sector_perf[sector].append(change)
        except: pass
    data = []
    for s, c in sector_perf.items():
        data.append({"Sector": s, "Change": np.mean(c)})
    return pd.DataFrame(data)

def predict_stock_price(symbol, days=7):
    try:
        df = yf.download(symbol, period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df['Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
        X = df[['Ordinal']]; y = df['Close']
        model = LinearRegression().fit(X, y)
        future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, days+1)]
        future_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        preds = model.predict(future_ord)
        return df, pd.DataFrame({"Date": future_dates, "Predicted": preds})
    except: return None, None

def run_market_scanner():
    results = []
    prog = st.progress(0); total = len(UI_MAP)
    for i, (name, ticker) in enumerate(UI_MAP.items()):
        try:
            df = yf.download(ticker, period="1y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if not df.empty:
                close = df['Close'].iloc[-1]; high_52 = df['High'].max()
                sma50 = df['Close'].rolling(50).mean().iloc[-1]; sma200 = df['Close'].rolling(200).mean().iloc[-1]
                
                # RSI
                delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss; rsi = 100 - (100 / (1 + rs)); rsi_val = rsi.iloc[-1]
                
                sig = "NEUTRAL"
                if rsi_val < 30: sig = "OVERSOLD"
                elif close >= 0.95 * high_52: sig = "NEAR 52W HIGH"
                elif sma50 > sma200 and df['Close'].iloc[-10] < df['Close'].iloc[-1]: sig = "GOLDEN CROSS"
                results.append({"Stock": name, "Price": f"₹{close:.2f}", "RSI": round(rsi_val, 1), "Signal": sig})
        except: pass
        prog.progress((i + 1) / total)
    prog.empty()
    return pd.DataFrame(results)

# --- ADVANCED RESEARCH ENGINE (EIC + FINANCIALS) ---
class ResearchEngine:
    def __init__(self, df, info, ticker_obj, currency_sym):
        self.df = df
        self.info = info
        self.ticker_obj = ticker_obj
        self.currency = currency_sym
        self.close = df['Close'].iloc[-1] if not df.empty else 0
        self.sma200 = df['Close'].rolling(200).mean().iloc[-1] if not df.empty else 0
        self.ticker = info.get('symbol', 'UNKNOWN')

    def analyze_economy(self):
        """EIC: Economy Phase"""
        try:
            nifty = yf.download('^NSEI', period='1y', progress=False)
            if isinstance(nifty.columns, pd.MultiIndex): nifty.columns = nifty.columns.get_level_values(0)
            trend = "Bullish" if nifty['Close'].iloc[-1] > nifty['Close'].rolling(200).mean().iloc[-1] else "Bearish"
            return f"The broad market (NIFTY 50) is in a **{trend}** phase. Inflation and interest rates remain key monitorables."
        except: return "Macro data unavailable."

    def analyze_industry(self):
        """EIC: Industry Comparison"""
        sector = STOCK_SECTOR_MAP.get(self.ticker, "General")
        return f"Sector: **{sector}**. The company operates in a cyclical environment. Relative strength against sector peers is a key momentum driver."

    def analyze_company_challenges(self):
        """Analyzes Financials to find Challenges (Past Fiscal)."""
        challenges = []
        changes = []
        impact = []
        
        try:
            fin = self.ticker_obj.financials
            if not fin.empty:
                # 1. Revenue Check
                curr_rev = fin.loc['Total Revenue'][0]; prev_rev = fin.loc['Total Revenue'][1]
                rev_chg = (curr_rev - prev_rev) / prev_rev
                
                if rev_chg < 0:
                    challenges.append("Declining Revenue Growth year-over-year.")
                    changes.append("Management may focus on cost optimization or new product launches.")
                    impact.append(f"Top-line contracted by {abs(rev_chg)*100:.1f}%, impacting overall cash flows.")
                else:
                    impact.append(f"Revenue grew by {rev_chg*100:.1f}%, showing strong demand.")

                # 2. Net Income Check
                curr_ni = fin.loc['Net Income'][0]; prev_ni = fin.loc['Net Income'][1]
                ni_chg = (curr_ni - prev_ni) / prev_ni
                
                if ni_chg < 0:
                    challenges.append("Profitability squeeze due to rising input costs or interest expenses.")
                    changes.append("Likely operational restructuring or price hikes implemented.")
                    impact.append(f"Bottom-line (Net Income) fell by {abs(ni_chg)*100:.1f}%.")
                
                # 3. Debt Check (Simulated if BS data available)
                # (Simple placeholder for logic)
                if self.info.get('debtToEquity', 0) > 200:
                    challenges.append("High leverage ratios restricting further capex.")
        except:
            challenges.append("Financial data insufficient for granular analysis.")
            
        if not challenges: challenges.append("No major financial red flags detected in recent filings.")
        if not changes: changes.append("Standard operational efficiency measures likely in place.")
        
        return challenges, changes, impact

    def calculate_dcf(self):
        try:
            eps = self.info.get('trailingEps'); growth = self.info.get('earningsGrowth', 0.10)
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

    def get_risk_metrics(self):
        if self.df.empty: return {}
        rolling_max = self.df['Close'].cummax()
        drawdown = (self.df['Close'] - rolling_max) / rolling_max
        volatility = self.df['Close'].pct_change().std() * np.sqrt(252)
        return {"Max Drawdown": f"{drawdown.min()*100:.2f}%", "Volatility": f"{volatility*100:.2f}%", "Beta": self.info.get('beta', 'N/A')}

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15); self.cell(0, 10, 'Equity Research Report', 0, 1, 'C'); self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12); self.set_fill_color(200, 220, 255); self.cell(0, 6, f"  {title}", 0, 1, 'L', 1); self.ln(4)
    def chapter_body(self, body):
        self.set_font('Arial', '', 10); self.multi_cell(0, 5, body); self.ln()

def create_pdf_bytes(ticker, info, rating, challenges, changes, impact, risk, intrinsic_val, currency_sym, eic_text, business_sum):
    pdf = PDFReport(); pdf.add_page(); safe_curr = "Rs. " if currency_sym == "₹" else "$"
    pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, sanitize_text(f"{info.get('longName', ticker)}"), 0, 1, 'L')
    pdf.set_font('Arial', '', 12); pdf.cell(0, 8, sanitize_text(f"Price: {safe_curr}{info.get('currentPrice', 'N/A')} | Rating: {rating}"), 0, 1, 'L'); pdf.ln(5)
    
    # EIC
    pdf.chapter_title("EIC Framework Analysis")
    pdf.chapter_body(sanitize_text(f"Economy: {eic_text['E']}"))
    pdf.chapter_body(sanitize_text(f"Industry: {eic_text['I']}"))
    
    # Business & Challenges
    pdf.chapter_title("Strategic Review")
    pdf.set_font('Arial', 'B', 10); pdf.cell(0, 5, "Business Profile:", 0, 1); pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, sanitize_text(business_sum[:500] + "...")); pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 10); pdf.cell(0, 5, "Challenges faced in Past Fiscal:", 0, 1); pdf.set_font('Arial', '', 10)
    for c in challenges: pdf.cell(0, 5, sanitize_text(f"- {c}"), 0, 1)
    
    pdf.set_font('Arial', 'B', 10); pdf.cell(0, 5, "Impact on Financials:", 0, 1); pdf.set_font('Arial', '', 10)
    for i in impact: pdf.cell(0, 5, sanitize_text(f"- {i}"), 0, 1)
    
    # Risk
    pdf.chapter_title("Risk Profile"); pdf.cell(0, 5, f"Max Drawdown: {risk.get('Max Drawdown')}", 0, 1); pdf.cell(0, 5, f"Volatility: {risk.get('Volatility')}", 0, 1); pdf.ln(5)
    pdf.ln(10); pdf.set_font('Arial', 'I', 8); pdf.multi_cell(0, 5, "Disclaimer: Automated report generated by AI. Not financial advice."); return pdf.output(dest='S').encode('latin-1', 'replace')

# =========================================
# --- MAIN APP UI ---
# =========================================
with st.sidebar:
    st.markdown("<h1 class='neon-text'>FinTerminal India</h1>", unsafe_allow_html=True)
    
    macro = get_macro_data()
    if macro is not None:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='macro-box'><div class='macro-lbl'>USD/INR</div><div class='macro-val'>{macro['INR=X']:.2f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='macro-box'><div class='macro-lbl'>Crude</div><div class='macro-val'>{macro['CL=F']:.1f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='macro-box'><div class='macro-lbl'>Gold</div><div class='macro-val'>{macro['GC=F']:.0f}</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # NAVIGATION
    mode = st.radio("Menu:", ["📊 Dashboard", "🔮 AI Oracle & Sectors", "📉 Option Chain", "🧮 Option Calculator", "⚡ Market Scanner", "💼 Portfolio Sim", "📑 Report Gen"])
    st.markdown("---")
    
    # Stock Selection
    if mode not in ["⚡ Market Scanner", "💼 Portfolio Sim", "🧮 Option Calculator"]:
        search_mode = st.checkbox("Manual Ticker Search", value=False)
        if search_mode: symbol = st.text_input("Enter Ticker", "SBIN.NS").upper()
        else: selected_name = st.selectbox("Popular Stocks", list(UI_MAP.keys())); symbol = UI_MAP[selected_name]
        currency_sym = get_currency_symbol(symbol)
    
    # Alerts
    with st.expander("🔔 Price Alerts"):
        alert_price = st.number_input("Target Price", value=0.0)
        if alert_price > 0:
            st.caption(f"Alert active for {symbol if 'symbol' in locals() else 'Current'} > {alert_price}")
            
    # FIX: Define generate_btn in sidebar if in correct mode
    if mode == "📑 Report Gen":
        generate_btn = st.button("🚀 Generate Report", type="primary")

st.markdown(get_ticker_tape_data(), unsafe_allow_html=True)

# --- PRICE ALERT CHECK ---
if 'symbol' in locals() and alert_price > 0:
    try:
        curr = yf.Ticker(symbol).info.get('currentPrice', 0)
        if curr >= alert_price:
            st.toast(f"🚨 ALERT: {symbol} has crossed {alert_price}! Current: {curr}", icon="🔥")
    except: pass

# =========================================
# MODE 1: DASHBOARD
# =========================================
if mode == "📊 Dashboard":
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", datetime.now() - timedelta(days=365))
    end = c2.date_input("End", datetime.now())
    show_bb = st.checkbox("Bollinger Bands", True)
    show_macd = st.checkbox("MACD", True)
    
    ticker_obj = yf.Ticker(symbol); info = ticker_obj.info
    df = load_historical_data(symbol, start, end)
    
    if df is not None and not df.empty:
        c1, c2 = st.columns([3, 1])
        with c1: st.title(info.get('shortName', symbol)); st.caption(f"Sector: {info.get('sector', 'N/A')}")
        with c2: p = df['Close'].iloc[-1]; d = p - df['Close'].iloc[-2]; st.metric("Price", f"{currency_sym}{p:,.2f}", f"{d:.2f}")

        t1, t2, t3 = st.tabs(["📈 Pro Chart", "📊 Fundamentals", "📰 News"])
        
        with t1:
            rows = 2 if show_macd else 1
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3] if show_macd else [1])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=symbol, increasing_line_color='#C6F221', decreasing_line_color='#FF3B30'), row=1, col=1)
            
            if show_bb:
                sma20 = df['Close'].rolling(20).mean(); std = df['Close'].rolling(20).std()
                fig.add_trace(go.Scatter(x=df.index, y=sma20+2*std, line=dict(color='rgba(255,255,255,0.3)'), name="BB Upper"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=sma20-2*std, line=dict(color='rgba(255,255,255,0.3)'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name="BB Lower"), row=1, col=1)
            
            if show_macd:
                ema12 = df['Close'].ewm(span=12).mean(); ema26 = df['Close'].ewm(span=26).mean(); macd = ema12 - ema26; sig = macd.ewm(span=9).mean()
                fig.add_trace(go.Scatter(x=df.index, y=macd, line=dict(color='#00E5FF'), name="MACD"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=sig, line=dict(color='#FFEA00'), name="Signal"), row=2, col=1)
                fig.add_trace(go.Bar(x=df.index, y=macd-sig, name="Hist"), row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=600, paper_bgcolor='#161B22', plot_bgcolor='#161B22'); st.plotly_chart(fig, use_container_width=True)
        
        with t2:
            st.subheader("Financial Growth")
            try:
                fin = ticker_obj.financials
                if not fin.empty:
                    rev = fin.loc['Total Revenue'].iloc[::-1]; net = fin.loc['Net Income'].iloc[::-1]
                    fig_f = go.Figure(data=[go.Bar(name='Revenue', x=rev.index, y=rev.values, marker_color='#1E88E5'), go.Bar(name='Net Income', x=net.index, y=net.values, marker_color='#43A047')])
                    fig_f.update_layout(barmode='group', template="plotly_dark", height=400, paper_bgcolor='#161B22'); st.plotly_chart(fig_f, use_container_width=True)
                else: st.info("Financials unavailable")
            except: st.info("Unavailable")
        
        with t3:
            news = fetch_google_news(symbol)
            if news:
                for a in news:
                    try: s = SentimentIntensityAnalyzer().polarity_scores(a['title'])['compound']; m = "🟢" if s > 0.05 else "🔴" if s < -0.05 else "⚪"
                    except: m="⚪"
                    st.markdown(f"{m} **[{a['title']}]({a['link']})**"); st.caption(f"Source: {a.get('publisher')}"); st.divider()
            else: st.warning("No news.")

# =========================================
# MODE 2: AI ORACLE & SECTORS
# =========================================
elif mode == "🔮 AI Oracle & Sectors":
    st.title("🔮 AI Market Intelligence")
    t1, t2 = st.tabs(["📈 Price Prediction", "🗺️ Sector Heatmap"])
    
    with t1:
        st.subheader(f"AI Forecast: {symbol}")
        if st.button("Run Prediction Model"):
            with st.spinner("Training Model..."):
                hist_df, pred_df = predict_stock_price(symbol)
                if hist_df is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Close'], name='History', line=dict(color='#00BCD4')))
                    fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted'], name='AI Forecast', line=dict(color='#C6F221', dash='dash')))
                    fig.update_layout(template="plotly_dark", height=500); st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(pred_df)
    
    with t2:
        if st.button("Scan Sectors"):
            with st.spinner("Analyzing..."):
                sec_df = run_sector_analysis()
                fig = px.treemap(sec_df, path=['Sector'], values='Change', color='Change', color_continuous_scale='RdYlGn', title='Sector Heatmap')
                st.plotly_chart(fig, use_container_width=True)

# =========================================
# MODE 3: OPTION CHAIN
# =========================================
elif mode == "📉 Option Chain":
    st.title("📉 Option Chain Analysis")
    try:
        tk = yf.Ticker(symbol); exps = tk.options
        if exps:
            exp_date = st.selectbox("Expiry Date", exps)
            opt = tk.option_chain(exp_date); calls = opt.calls; puts = opt.puts
            curr = tk.info.get('currentPrice', calls['strike'].iloc[len(calls)//2])
            calls = calls[(calls['strike'] > curr*0.9) & (calls['strike'] < curr*1.1)]
            puts = puts[(puts['strike'] > curr*0.9) & (puts['strike'] < curr*1.1)]
            c1, c2 = st.columns(2)
            with c1: st.subheader("Calls (Res)"); st.dataframe(calls[['strike', 'lastPrice', 'openInterest']].set_index('strike'), use_container_width=True)
            with c2: st.subheader("Puts (Sup)"); st.dataframe(puts[['strike', 'lastPrice', 'openInterest']].set_index('strike'), use_container_width=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=calls['strike'], y=calls['openInterest'], name='Call OI', marker_color='#FF3B30'))
            fig.add_trace(go.Bar(x=puts['strike'], y=puts['openInterest'], name='Put OI', marker_color='#00C805'))
            fig.update_layout(template="plotly_dark", title="OI Distribution", barmode='group'); st.plotly_chart(fig, use_container_width=True)
        else: st.warning("No Options Data.")
    except Exception as e: st.error(f"Error: {e}")

# =========================================
# MODE 4: OPTION CALCULATOR
# =========================================
elif mode == "🧮 Option Calculator":
    st.title("🧮 Black-Scholes Calculator")
    c1, c2 = st.columns(2)
    S = c1.number_input("Spot Price", 100.0); K = c1.number_input("Strike", 100.0); T = c1.number_input("Days", 30)
    sig = c2.number_input("IV %", 20.0); r = c2.number_input("Risk-Free %", 5.0)
    if st.button("Calculate"):
        cp = black_scholes(S, K, T/365, r/100, sig/100, 'call'); pp = black_scholes(S, K, T/365, r/100, sig/100, 'put')
        c1.metric("Call Value", f"{cp:.2f}"); c2.metric("Put Value", f"{pp:.2f}")

# =========================================
# MODE 5: SCANNER
# =========================================
elif mode == "⚡ Market Scanner":
    st.title("⚡ AI Market Scanner")
    if st.button("Start Scan", type="primary"):
        with st.spinner("Scanning..."):
            df = run_market_scanner()
            st.dataframe(df.style.applymap(lambda x: 'color:#00FF00' if 'OVERSOLD' in str(x) else ('color:#FFA500' if 'HIGH' in str(x) else ''), subset=['Signal']), use_container_width=True, height=600)

# =========================================
# MODE 6: PORTFOLIO
# =========================================
elif mode == "💼 Portfolio Sim":
    st.title("💼 Portfolio Sim")
    with st.expander("Add Trade"):
        c1, c2, c3 = st.columns(3)
        n = c1.selectbox("Stock", list(UI_MAP.keys())); q = c2.number_input("Qty", 1); p = c3.number_input("Price", 1.0)
        if st.button("Add"): st.session_state.portfolio.append({"Symbol": UI_MAP[n], "Name": n, "Qty": q, "Buy Price": p}); st.success(f"Added {n}")
    
    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        curr = []; val = []
        for _, r in df.iterrows():
            try: c = yf.Ticker(r['Symbol']).history(period="1d")['Close'].iloc[-1]
            except: c = 0
            curr.append(c); val.append(c*r['Qty'])
        df['Current'] = curr; df['Value'] = val; df['P&L'] = df['Value'] - (df['Buy Price']*df['Qty'])
        st.dataframe(df, use_container_width=True); st.metric("Total P&L", f"₹{df['P&L'].sum():,.2f}")

# =========================================
# MODE 7: REPORT GEN (EIC INCLUDED)
# =========================================
elif mode == "📑 Report Gen":
    ticker_obj = yf.Ticker(symbol)
    try: info = ticker_obj.info
    except: info = {}
    
    if not generate_btn:
        st.info(f"👈 Click **'Generate Report'** in the sidebar to analyze {symbol}.")
    else:
        with st.spinner(f"Running Institutional Analysis on {symbol}..."):
            df_rep = load_historical_data(symbol, datetime.now()-timedelta(days=400), datetime.now())
            if df_rep is not None:
                # EIC Engines
                eng = ResearchEngine(df_rep, info, ticker_obj, currency_sym)
                ival = eng.calculate_dcf()
                rating, r_cls = eng.get_rating(ival)
                risk = eng.get_risk_metrics()
                
                # EIC Analysis
                eic_text = {
                    'E': eng.analyze_economy(),
                    'I': eng.analyze_industry()
                }
                
                # Financial Deep Dive
                challenges, changes, impact = eng.analyze_company_challenges()
                business_sum = info.get('longBusinessSummary', "No summary available.")
                
                st.markdown(f"""<div class="report-card"><h1>{info.get('longName', symbol)}</h1><span class="rating-badge {r_cls}">{rating}</span><hr><div style="display:flex; justify-content:space-between;"><div>Current Price<br><b>{currency_sym}{info.get('currentPrice',0):,.2f}</b></div><div>Target<br><b>{currency_sym}{info.get('targetMeanPrice', 'N/A')}</b></div><div>Intrinsic<br><b>{currency_sym}{ival:,.2f}</b></div></div></div>""", unsafe_allow_html=True)
                
                # EIC Section UI
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.subheader("🌏 EIC Framework Analysis")
                st.markdown(f"**Economy:** {eic_text['E']}")
                st.markdown(f"**Industry:** {eic_text['I']}")
                st.markdown("---")
                
                st.subheader("🏢 Business & Strategic Review")
                with st.expander("Company Profile (Click to Expand)", expanded=True):
                    st.write(business_sum[:1000] + "...")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Challenges Faced (Past Fiscal)**")
                    for c in challenges: st.markdown(f"<div class='challenge-box'>{c}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("**Impact on Financials**")
                    for i in impact: st.markdown(f"<div class='impact-box'>{i}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1: st.subheader("⚠️ Risk"); st.write(risk)
                
                try: pdf = create_pdf_bytes(symbol, info, rating, challenges, changes, impact, risk, ival, currency_sym, eic_text, business_sum); st.download_button("Download PDF", pdf, f"{symbol}_Report.pdf", "application/pdf", type='primary')
                except Exception as e: st.error(f"PDF Error: {e}")
