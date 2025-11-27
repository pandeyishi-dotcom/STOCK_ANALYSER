# Paynext - Fintech Research Platform

![Paynext Banner](https://via.placeholder.com/1000x300/2E1065/FFFFFF?text=Paynext+Fintech+Platform)

## Overview
Our goal was to create a platform that allows retail investors to conduct detailed stock research with ease. We aimed to offer features such as real-time stock data, technical analysis charts, financial metrics, news, and many more features including an AI assistant. Streamlit’s intuitive and efficient framework made it an ideal choice for this project.

## Key Features

### Component-Based Design
* **Widgets and Controls:** Leveraged Streamlit’s built-in widgets (sliders, dropdowns, date pickers) for interactive filtering.
* **Layouts:** utilized `st.columns` and `st.container` for a clean, grid-based dashboard interface.

### Data Visualization
* **Interactive Charts:** Integration with Plotly for interactive candlestick and line charts.
* **Financial Metrics:** Real-time rendering of KPIs and Pandas dataframes for deep-dive analysis.

### State Management
* **Session State:** Maintains user selection (tickers, date ranges) across reruns to ensure a seamless UX.

## Tech Stack
* **Frontend/Backend:** Streamlit (Python)
* **Data Source:** yfinance
* **Visualization:** Plotly, Pandas

## Benefits of our Architecture
1.  **Shorter Development Times:** Rapid prototyping allowed us to move from concept to code in days, not weeks.
2.  **Unified Codebase:** No separation between frontend and backend reduces maintenance overhead.
3.  **Cost Effective:** Simplified deployment and infrastructure management.

## How to Run Locally

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
