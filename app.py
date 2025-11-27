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
        .report-card { background-color: #161B22; padding: 20px; border-radius: 10px; border: 1px solid #30363D; margin-bottom: 15px; }
        .rating-badge { padding: 8px 20px; border-radius: 20px; font-weight: 900; font-size: 1.1rem; display: inline-block; }
        .buy { background-color: #238636; color: white; } .sell { background-color: #DA3633; color: white; } .hold { background-color: #D29922; color: black; }
        .ticker-wrap { width: 100%; overflow: hidden; background-color: #000; padding: 8px 0; white-space: nowrap; border-bottom: 2px solid #C6F221; }
        .ticker { display: inline-block; animation: marquee 45s linear infinite; }
        .ticker-item { display: inline-block; padding: 0 2rem; font-size: 1rem; color: #C6F221; font-family: 'Courier New', monospace; }
        @keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'portfolio' not in st.session_state: st.session_state.portfolio = []

# --- MAPS ---
UI_MAP = {
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS", "SBI": "SBIN.NS",
    "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS", "Airtel": "BHARTIARTL.NS", "HUL": "HINDUNILVR.NS",
    "ITC": "ITC.NS", "L&T": "LT.NS", "Tata Motors": "TATAMOTORS.NS", "Axis Bank": "AXISBANK.NS",
    "Sun Pharma": "SUNPHARMA.NS", "Maruti": "MARUTI.NS", "Titan": "TITAN.NS", "Bajaj Fin": "BAJFINANCE.NS",
    "Asian Paints": "ASIANPAINT.NS", "M&M": "M&M.NS", "Wipro": "WIPRO.NS", "Tata Steel": "TATASTEEL.NS",
    "Zomato": "ZOMATO.NS", "Paytm": "PAYTM.NS", "Idea": "IDEA.NS", "Yes Bank": "YESBANK.NS",
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK"
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
        return yf.download(['INR=X', 'CL=F', 'GC=F'], period="1d", progress=False)['Close'].iloc[-1]
    except: return None

@st.cache_data(ttl=3600)
def get_ticker_tape_data():
    try:
        data = yf.download(['^NSEI', '^BSESN', 'RELIANCE.NS', 'HDFCBANK.NS'], period="1d", interval="1m", progress=False)['Close'].iloc[-1]
        html = ""
        if isinstance(data, pd.Series):
             for t, p in data.items(): 
                 name = "NIFTY" if '^NSEI' in t else "SENSEX" if '^BSESN' in t else t.replace('.NS','')
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
    data = [{"Sector": s, "Change": np.mean(c)} for s, c in sector_perf.items()]
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

# --- PDF ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15); self.cell(0, 10, 'Equity Research Report', 0, 1, 'C'); self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
def create_pdf_bytes(ticker, info, rating, thesis, currency_sym):
    pdf = PDFReport(); pdf.add_page(); safe_curr = "Rs. " if currency_sym == "₹" else "$"
    pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, sanitize_text(f"{info.get('longName', ticker)}"), 0, 1, 'L')
    pdf.set_font('Arial', '', 12); pdf.cell(0, 8, sanitize_text(f"Price: {safe_curr}{info.get('currentPrice', 'N/A')} | Rating: {rating}"), 0, 1, 'L'); pdf.ln(5)
    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, "Investment Thesis", 0, 1)
    pdf.set_font('Arial', '', 10); pdf.multi_cell(0, 5, sanitize_text(thesis)); pdf.ln()
    return pdf.output(dest='S').encode('latin-1', 'replace')

# =========================================
# --- MAIN APP UI ---
# =========================================
with st.sidebar:
    st.markdown("<h1 class='neon-text'>FinTerminal India</h1>", unsafe_allow_html=True)
    
    # Macro
    macro = get_macro_data()
    if macro is not None:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='macro-box'><div class='macro-lbl'>USD/INR</div><div class='macro-val'>{macro['INR=X']:.2f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='macro-box'><div class='macro-lbl'>Crude</div><div class='macro-val'>{macro['CL=F']:.1f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='macro-box'><div class='macro-lbl'>Gold</div><div class='macro-val'>{macro['GC=F']:.0f}</div></div>", unsafe_allow_html=True)
    
    # Navigation
    menu_options = ["📊 Dashboard", "🔮 AI Oracle & Sectors", "📉 Option Chain", "🧮 Option Calculator", "⚡ Market Scanner", "💼 Portfolio Sim", "📑 Report Gen"]
    mode = st.selectbox("Menu:", menu_options)
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

st.markdown(get_ticker_tape_data(), unsafe_allow_html=True)

# --- PRICE ALERT CHECK LOGIC ---
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
    # Controls
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", datetime.now() - timedelta(days=365))
    end = c2.date_input("End", datetime.now())
    show_bb = st.checkbox("Bollinger Bands", True)
    show_macd = st.checkbox("MACD", True)
    
    # Data
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
    st.caption("Fetches Option Chain for selected Expiry. Note: Indian Option Chain via Yahoo is often limited/delayed.")
    
    try:
        tk = yf.Ticker(symbol)
        exps = tk.options
        if exps:
            exp_date = st.selectbox("Expiry Date", exps)
            opt = tk.option_chain(exp_date)
            calls = opt.calls; puts = opt.puts
            
            # Filter near money
            curr = tk.info.get('currentPrice', calls['strike'].iloc[len(calls)//2])
            calls = calls[(calls['strike'] > curr*0.9) & (calls['strike'] < curr*1.1)]
            puts = puts[(puts['strike'] > curr*0.9) & (puts['strike'] < curr*1.1)]
            
            c1, c2 = st.columns(2)
            with c1: st.subheader("Calls (Resistance)"); st.dataframe(calls[['strike', 'lastPrice', 'openInterest', 'volume']].set_index('strike'), use_container_width=True)
            with c2: st.subheader("Puts (Support)"); st.dataframe(puts[['strike', 'lastPrice', 'openInterest', 'volume']].set_index('strike'), use_container_width=True)
            
            # OI Chart
            fig = go.Figure()
            fig.add_trace(go.Bar(x=calls['strike'], y=calls['openInterest'], name='Call OI', marker_color='#FF3B30'))
            fig.add_trace(go.Bar(x=puts['strike'], y=puts['openInterest'], name='Put OI', marker_color='#00C805'))
            fig.update_layout(template="plotly_dark", title="Open Interest Distribution", barmode='group'); st.plotly_chart(fig, use_container_width=True)
        else: st.warning("No Option Chain data found for this ticker.")
    except Exception as e: st.error(f"Error fetching options: {e}")

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
# MODE 7: REPORT GEN
# =========================================
elif mode == "📑 Report Gen":
    if st.button("Generate Report"):
        info = yf.Ticker(symbol).info
        thesis = f"Trading at P/E {info.get('trailingPE','N/A')}. Margins {info.get('profitMargins',0)*100:.1f}%."
        rating = "BUY" if info.get('profitMargins',0) > 0.1 else "HOLD"
        try: pdf = create_pdf_bytes(symbol, info, rating, thesis, currency_sym); st.download_button("Download PDF", pdf, f"{symbol}_Report.pdf", "application/pdf")
        except: st.error("Error generating PDF")
