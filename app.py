import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from fpdf import FPDF
import numpy as np

# --- NLTK SETUP (Robust for Cloud) ---
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
    page_title="FinTerminal Pro | AI Research",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (DARK NEON THEME + REPORT STYLING) ---
st.markdown("""
    <style>
        /* Global Settings */
        .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Roboto', sans-serif; }
        [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
        h1, h2, h3 { color: #C6F221 !important; }
        .stCaption { color: #8B949E !important; }
        hr { border: 0; border-top: 1px solid #30363D; }
        
        /* Neon Accents */
        .neon-text { color: #C6F221; font-weight: bold; text-shadow: 0 0 5px rgba(198, 242, 33, 0.5); }
        
        /* Custom Metric Card used in Dashboard */
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
        .buy { background-color: #238636; color: white; box-shadow: 0 0 10px #238636; }
        .sell { background-color: #DA3633; color: white; box-shadow: 0 0 10px #DA3633;}
        .hold { background-color: #D29922; color: black; box-shadow: 0 0 10px #D29922;}
        
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

# =========================================
# --- CORE LOGIC ENGINES ---
# =========================================

class ResearchEngine:
    """Rules-based AI engine for generating insights."""
    def __init__(self, df, info):
        self.df = df
        self.info = info
        self.close = df['Close'].iloc[-1] if not df.empty else 0
        self.sma200 = df['Close'].rolling(200).mean().iloc[-1] if not df.empty else 0
    
    def calculate_dcf(self):
        """Simplified Intrinsic Value Calculation."""
        try:
            eps = self.info.get('trailingEps')
            # Use analyst estimate if available, else default to 8%
            growth_rate = self.info.get('earningsGrowth', 0.08)
            terminal_pe = 15 # Conservative multiple
            if eps is None or eps <= 0: return 0
            
            # 5-year growth projection
            future_eps = eps * ((1 + growth_rate) ** 5)
            fair_value = future_eps * terminal_pe
            return fair_value
        except: return 0

    def get_rating(self, intrinsic_value):
        score = 0
        # Technical Score
        if self.close > self.sma200: score += 1
        # Fundamental Score
        if self.info.get('forwardPE', 100) < 25: score += 0.5
        if self.info.get('profitMargins', 0) > 0.15: score += 0.5
        # Valuation Score (DCF)
        if intrinsic_value > 0 and self.close < intrinsic_value * 0.85: score += 2
        elif intrinsic_value > 0 and self.close > intrinsic_value * 1.15: score -= 1

        if score >= 3: return "STRONG BUY", "buy"
        elif score >= 1.5: return "BUY", "buy"
        elif score >= 0.5: return "HOLD", "hold"
        else: return "SELL", "sell"

    def generate_thesis(self):
        reasons = []
        # Valuation
        pe = self.info.get('forwardPE')
        if pe: reasons.append(f"Trading at a forward P/E of {pe:.1f}x.")
        
        # Technicals
        if self.close > self.sma200: reasons.append("The stock is in a long-term technical uptrend, trading above its 200-day moving average.")
        else: reasons.append("Technically, the stock remains in a downtrend below key moving averages.")
        
        # Fundamentals
        rev_growth = self.info.get('revenueGrowth', 0)
        if rev_growth > 0.10: reasons.append(f"Demonstrating strong top-line expansion with {rev_growth*100:.1f}% YoY revenue growth.")
        
        return " ".join(reasons) if reasons else "Insufficient data to generate thesis."

class PDFReport(FPDF):
    """PDF Generation Class"""
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, 'Equity Research Summary', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'AI-Generated Report | Page {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(35, 134, 54) # Green accent
        self.set_text_color(255, 255, 255)
        self.cell(0, 6, f"  {title}", 0, 1, 'L', 1)
        self.set_text_color(0, 0, 0)
        self.ln(4)
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

# =========================================
# --- HELPER FUNCTIONS ---
# =========================================
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_ticker_tape_data():
    """Fetches data for marquee once per hour."""
    try:
        tickers = ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'BTC-USD', 'ETH-USD', 'GC=F', '^GSPC']
        data = yf.download(tickers, period="1d", interval="1m", progress=False)['Close'].iloc[-1]
        html = ""
        if isinstance(data, pd.Series):
             for t, p in data.items(): 
                 symbol_clean = t.replace('=X','').replace('-USD','')
                 html += f"<span class='ticker-item'>{symbol_clean}: {p:,.2f}</span>"
        return f"<div class='ticker-wrap'><div class='ticker'>{html}{html}</div></div>"
    except: return ""

@st.cache_data
def load_historical_data(symbol, start, end):
    """Caches ONLY dataframe price data."""
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
        title = article.get('title','')
        # Simple cleanup to remove repetitive publisher names from headlines if needed
        scores.append(sia.polarity_scores(title)['compound'])
        
    avg = np.mean(scores) if scores else 0
    if avg >= 0.05: return avg, "Bullish 🟢"
    elif avg <= -0.05: return avg, "Bearish 🔴"
    else: return avg, "Neutral ⚪"

def create_pdf_bytes(ticker, info, rating, thesis, peers_df, intrinsic_val):
    pdf = PDFReport()
    pdf.add_page()
    
    # Header Info
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f"{info.get('longName', ticker)} ({ticker})", 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Price: ${info.get('currentPrice', 'N/A')} | Recommendation: {rating}", 0, 1, 'L')
    if intrinsic_val > 0:
        pdf.cell(0, 8, f"Estimated Intrinsic Value (DCF): ${intrinsic_val:.2f}", 0, 1, 'L')
    pdf.ln(5)
    
    # Thesis
    pdf.chapter_title("Investment Thesis")
    pdf.chapter_body(thesis)
    
    # Metrics Table Simulation in PDF
    pdf.chapter_title("Key Fundamentals")
    pdf.set_font('Courier', '', 10)
    metrics = [
        f"P/E Ratio: {info.get('trailingPE', 'N/A')}",
        f"Fwd P/E:   {info.get('forwardPE', 'N/A')}",
        f"PEG Ratio: {info.get('pegRatio', 'N/A')}",
        f"Profit Mgn:{info.get('profitMargins', 0)*100:.1f}%",
        f"Beta:      {info.get('beta', 'N/A')}"
    ]
    for m in metrics: pdf.cell(0, 5, m, 0, 1)
    pdf.ln(5)

    # Peer Comparison Summary
    if peers_df is not None and not peers_df.empty:
        pdf.chapter_title("Peer Comparison Summary")
        pdf.chapter_body(f"Compared against: {', '.join(peers_df['Ticker'].tolist())}")
        avg_pe = peers_df['P/E'].replace('N/A', np.nan).astype(float).mean()
        pdf.chapter_body(f"Peer Average P/E: {avg_pe:.1f}x")

    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 5, "Disclaimer: This report is generated by automated algorithmic analysis for informational purposes only. It is not financial advice.")
    
    return pdf.output(dest='S').encode('latin-1', 'ignore') # encode latin-1 for fpdf compatibility

# =========================================
# --- MAIN APP UI STARTS HERE ---
# =========================================

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 class='neon-text'>FinTerminal Pro</h1>", unsafe_allow_html=True)
    
    # View Mode Selection
    mode = st.radio("Select Mode:", ["📊 Live Dashboard", "📑 Research ReportGen"], label_visibility="collapsed")
    st.markdown("---")

    symbol = st.text_input("Main Ticker", "AAPL").upper()
    
    if mode == "📊 Live Dashboard":
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start", datetime.now() - timedelta(days=365))
        end_date = col2.date_input("End", datetime.now())
        st.markdown("### 📈 Technicals")
        show_rsi = st.checkbox("Show RSI (14)", value=True)
        show_sma = st.checkbox("Show SMA (50/200)", value=False)
        compare_sym = st.text_input("Compare Ticker")
    
    else: # Research Report Mode
        st.markdown("### ⚖️ Valuation Settings")
        peers_input = st.text_input("Peer Tickers (comma sep)", "MSFT, GOOGL, NVDA")
        generate_btn = st.button("🚀 Generate Full Report", type="primary")

# Display Ticker Tape (Top of page)
st.markdown(get_ticker_tape_data(), unsafe_allow_html=True)

# Initialize Ticker Object (Live connection)
ticker_obj = yf.Ticker(symbol)

# Fetch Info & News safely
try: info = ticker_obj.info
except: info = {}
try: news = ticker_obj.news
except: news = []

# Detect Asset Type (Important for hiding fundamentals on Crypto/Forex)
quote_type = info.get('quoteType', 'Unknown')
is_stock = quote_type == 'EQUITY'

# =========================================
# MODE 1: LIVE DASHBOARD VIEW
# =========================================
if mode == "📊 Live Dashboard":
    # Load Historical Data
    df = load_historical_data(symbol, start_date, end_date)

    if df is not None and not df.empty:
        # 1. Header & Sentiment
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title(info.get('shortName', symbol))
            st.caption(f"Type: {quote_type} | Sector: {info.get('sector', 'N/A')}")
        with c2:
            score, label = analyze_sentiment_vader(news)
            st.metric("AI News Sentiment", label, f"{score:.2f}")

        # 2. Tabs
        tabs_list = ["📈 Chart", "📰 News Wire"]
        if is_stock: tabs_list.insert(1, "🏗 Fundamentals")
        active_tabs = st.tabs(tabs_list)

        # Tab: Chart
        with active_tabs[0]:
            rows = 2 if show_rsi else 1
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3] if show_rsi else [1])
            # Price
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                         name=symbol, increasing_line_color='#C6F221', decreasing_line_color='#FF3B30'), row=1, col=1)
            # SMAs
            if show_sma:
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), line=dict(color='#00BCD4', width=1), name='SMA 50'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), line=dict(color='#FFC107', width=1), name='SMA 200'), row=1, col=1)
            # Comparison
            if compare_sym:
                comp_df = load_historical_data(compare_sym, start_date, end_date)
                if comp_df is not None:
                    fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Close'], line=dict(color='white', width=1, dash='dot'),
                                             name=f"vs {compare_sym}"), row=1, col=1)
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

            fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='#161B22', plot_bgcolor='#161B22')
            st.plotly_chart(fig, use_container_width=True)

        # Tab: Fundamentals (Stock Only)
        if is_stock:
            with active_tabs[1]:
                f1, f2 = st.columns(2)
                with f1:
                    st.subheader("📜 Balance Sheet (Recent)")
                    try: st.dataframe(ticker_obj.balance_sheet.iloc[:, :2], height=300, use_container_width=True)
                    except: st.info("Data unavailable.")
                with f2:
                    st.subheader("👥 Insider Activity")
                    try: st.dataframe(ticker_obj.insider_transactions.head(10), height=300, use_container_width=True)
                    except: st.info("Data unavailable.")

        # Tab: News
        news_idx = 2 if is_stock else 1
        with active_tabs[news_idx]:
            if news:
                for article in news[:7]:
                    score = SentimentIntensityAnalyzer().polarity_scores(article.get('title',''))['compound']
                    mood = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
                    st.markdown(f"{mood} **[{article.get('title')}]({article.get('link')})**")
                    st.caption(f"Source: {article.get('publisher')} | Score: {score:.2f}")
                    st.divider()
            else: st.warning("No news feed available.")
    else:
        st.error("Could not load data. Check symbol.")

# =========================================
# MODE 2: RESEARCH REPORT GENERATOR
# =========================================
elif mode == "📑 Research ReportGen":
    if not is_stock:
         st.warning("⚠️ Research Report mode is designed for equities (stocks). Valuation models may not apply to Crypto/Forex.")

    if generate_btn:
        with st.spinner("🤖 AI Analyst is crunching numbers..."):
            # Load data specifically for report (longer timeframe for SMAs)
            df_report = load_historical_data(symbol, datetime.now() - timedelta(days=400), datetime.now())
            
            if df_report is not None:
                # 1. Run AI Engine
                engine = ResearchEngine(df_report, info)
                intrinsic_val = engine.calculate_dcf()
                rating_text, rating_class = engine.get_rating(intrinsic_val)
                thesis_text = engine.generate_thesis()
                
                # 2. Run Peer Comparison
                peers_list = [x.strip().upper() for x in peers_input.split(',')]
                peer_data = []
                for t in [symbol] + peers_list:
                    try:
                        p_info = yf.Ticker(t).info
                        peer_data.append({
                            "Ticker": t,
                            "Price": p_info.get('currentPrice', 'N/A'),
                            "P/E": p_info.get('trailingPE', 'N/A'),
                            "Fwd P/E": p_info.get('forwardPE', 'N/A'),
                            "Profit Mgn": f"{p_info.get('profitMargins', 0)*100:.1f}%"
                        })
                    except: pass
                peer_df = pd.DataFrame(peer_data)

                # --- REPORT UI ---
                st.markdown(f"""
                <div class="report-container">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h1 style="margin:0; color:#C6F221;">{info.get('longName', symbol)}</h1>
                        <span class="rating-badge {rating_class}">{rating_text}</span>
                    </div>
                    <hr>
                    <div style="display:flex; justify-content:space-between; margin-top:15px;">
                         <div>Current Price<br><span style="font-size:1.5rem; font-weight:bold;">${info.get('currentPrice',0):.2f}</span></div>
                         <div>AI Intrinsic Value (DCF)<br><span style="font-size:1.5rem; font-weight:bold; color:{'#238636' if intrinsic_val > info.get('currentPrice',0) else '#DA3633'}">${intrinsic_val:.2f}</span></div>
                         <div>Analyst Target<br><span style="font-size:1.5rem;">${info.get('targetMeanPrice', 'N/A')}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown('<div class="report-container">', unsafe_allow_html=True)
                    st.subheader("🧐 Investment Thesis")
                    st.write(thesis_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="report-container">', unsafe_allow_html=True)
                    st.subheader("⚖️ Peer Comparison")
                    st.dataframe(peer_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with c2:
                    st.markdown('<div class="report-container">', unsafe_allow_html=True)
                    st.subheader("📊 Key Ratios")
                    metrics = {"P/E Ratio": info.get('trailingPE'), "Forward P/E": info.get('forwardPE'), "PEG Ratio": info.get('pegRatio'), "Price/Book": info.get('priceToBook'), "Beta": info.get('beta')}
                    for k, v in metrics.items():
                        st.markdown(f"**{k}:** <span style='float:right;'>{v if v else 'N/A'}</span>", unsafe_allow_html=True)
                        st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # --- PDF GENERATION & DOWNLOAD ---
                pdf_bytes = create_pdf_bytes(symbol, info, rating_text, thesis_text, peer_df, intrinsic_val)
                st.download_button(label="📄 Download PDF Research Report", data=pdf_bytes, file_name=f"{symbol}_Research_Report.pdf", mime="application/pdf", use_container_width=True, type='primary')
            
            else: st.error("Not enough historical data to generate report.")
    else:
        st.info("👈 Enter tickers in the sidebar and click 'Generate Full Report' to start the analysis.")
