import datetime
import re
import io
import json
import time
import yfinance as yf
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
from fpdf import FPDF  # pip install fpdf

# Download the VADER lexicon if needed.
nltk.download('vader_lexicon')

###############################
# Global Blacklist for Common Word Tickers
###############################
common_ticker_blacklist = {"IT", "ALL", "NOW", "ON", "SO", "BRO", "WELL", "PM"}

###############################
# Ticker Aliases Dictionary
###############################
default_ticker_aliases = {
    "AAPL": ["aapl", "apple", "apple inc"],
    "MSFT": ["msft", "microsoft", "microsoft corporation"],
    "NVDA": ["nvda", "nvidia", "nvidia corporation"],
    "AMZN": ["amzn", "amazon", "amazon.com"],
    "GOOGL": ["googl", "alphabet", "google"],
    "META": ["meta", "meta platforms", "facebook"],
    "TSLA": ["tsla", "tesla", "tesla inc"],
    "SPY": ["spy", "s&p500", "s&p 500", "sp500"],
    "QQQ": ["qqq", "nasdaq 100"],
    # ... add more as needed.
}
ticker_aliases = default_ticker_aliases.copy()

###############################
# Function to Load Ticker CSV
###############################
def load_ticker_csv(file) -> dict:
    df = pd.read_csv(file)
    aliases = {}
    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip()
        name = str(row["Name"]).strip()
        if symbol and len(symbol) > 1 and symbol.upper() not in common_ticker_blacklist:
            aliases[symbol] = [symbol.lower(), name.lower()]
    return aliases

###############################
# Parsing Functions
###############################
def parse_comments(text):
    pattern = re.compile(
        r"^(?P<header>.*?,\s+on\s+\w{3}\s+(?P<date>\d{4}-\d{2}-\d{2})\s+at\s+\d{2}:\d{2}.*)"
        r"(?P<comment>(?:\n(?!.*?,\s+on\s+\w{3}\s+\d{4}-\d{2}-\d{2}\s+at\s+\d{2}:\d{2}).+)*)",
        re.MULTILINE)
    comments = []
    dates = []
    for match in pattern.finditer(text):
        comment_text = match.group("comment").strip()
        if comment_text:
            comments.append(comment_text)
            dates.append(match.group("date"))
    return comments, dates

def parse_html_file(file_text):
    soup = BeautifulSoup(file_text, 'html.parser')
    comment_divs = soup.find_all("div", class_="comment")
    if comment_divs and len(comment_divs) > 5:
        comments = [div.get_text(separator="\n", strip=True) for div in comment_divs]
        return comments, []
    body = soup.find("body")
    text_content = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
    return parse_comments(text_content)

def parse_csv_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    comment_col = None
    for col in df.columns:
        if col.lower() == "comment":
            comment_col = col
            break
    if comment_col is None:
        comment_col = df.columns[0]
    return df[comment_col].astype(str).tolist()

def parse_json_file(uploaded_file):
    json_data = json.load(uploaded_file)
    comments = []
    analysis_date = None
    discussion_type = None
    for item in json_data:
        if analysis_date is None and item.get("dataType") == "post":
            title = item.get("title", "")
            if "discussion thread for" in title.lower():
                if "weekend" in title.lower():
                    discussion_type = "weekend"
                elif "daily" in title.lower():
                    discussion_type = "daily"
                m = re.search(r"([A-Za-z]+ \d{1,2}, \d{4})", title)
                if m:
                    try:
                        analysis_date = datetime.datetime.strptime(m.group(1), "%B %d, %Y").date()
                    except Exception:
                        pass
        if item.get("dataType") == "comment":
            body = item.get("body", "")
            if body:
                comments.append(body)
    return comments, analysis_date, discussion_type

###############################
# Sentiment Analysis
###############################
def analyze_sentiment(comments):
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(text)['compound'] for text in comments]
    overall = sum(scores) / len(scores) if scores else 0
    return overall, scores

###############################
# Ticker Extraction & Top Stocks
###############################
def extract_ticker_mentions_with_alias(text, ticker_aliases):
    text_lower = text.lower()
    found = set()
    for ticker, aliases in ticker_aliases.items():
        if len(ticker) == 1 or ticker.upper() in common_ticker_blacklist:
            continue
        for alias in aliases:
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text_lower):
                found.add(ticker)
                break
    return list(found)

def get_top_stocks_and_sentiment(comments, sentiment_scores, ticker_aliases):
    ticker_counts = {}
    ticker_sentiments = {}
    for comment, score in zip(comments, sentiment_scores):
        tickers = extract_ticker_mentions_with_alias(comment, ticker_aliases)
        for ticker in tickers:
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            ticker_sentiments.setdefault(ticker, []).append(score)
    top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    results = {}
    for ticker, count in top_tickers:
        sentiments = ticker_sentiments[ticker]
        avg_sent = sum(sentiments) / len(sentiments)
        if avg_sent > 0.05:
            sentiment_class = "bullish"
        elif avg_sent < -0.05:
            sentiment_class = "bearish"
        else:
            sentiment_class = "neutral"
        results[ticker] = {
            "mentions": count,
            "average_sentiment": avg_sent,
            "sentiment_class": sentiment_class
        }
    return results

###############################
# Trading Data Retrieval (with caching)
###############################
@st.cache_data(show_spinner=False)
def get_next_day_open_close(ticker, base_date_str):
    base_date = datetime.datetime.strptime(base_date_str, "%Y-%m-%d").date()
    end_date = base_date + datetime.timedelta(days=10)
    stock = yf.Ticker(ticker)
    hist = stock.history(start=base_date.strftime("%Y-%m-%d"),
                         end=end_date.strftime("%Y-%m-%d"))
    if hist.empty or len(hist) < 2:
        raise ValueError(f"Not enough trading data for {ticker}.")
    df = hist.reset_index()
    df = df[df['Date'].dt.date >= base_date]
    if len(df) < 2:
        raise ValueError(f"Not enough trading data for {ticker} after {base_date}.")
    next_day_row = df.iloc[1]
    return next_day_row['Open'], next_day_row['Close'], next_day_row['Date'].date()

@st.cache_data(show_spinner=False)
def get_week_open_close(ticker, base_date_str):
    base_date = datetime.datetime.strptime(base_date_str, "%Y-%m-%d").date()
    end_date = base_date + datetime.timedelta(days=10)
    stock = yf.Ticker(ticker)
    hist = stock.history(start=base_date.strftime("%Y-%m-%d"),
                         end=end_date.strftime("%Y-%m-%d"))
    if hist.empty or len(hist) < 2:
        raise ValueError(f"Not enough trading data for {ticker} to compute weekly data.")
    df = hist.reset_index()
    df = df[df['Date'].dt.date >= base_date]
    if df.empty:
        raise ValueError(f"No trading data for {ticker} after {base_date}.")
    week_open = df.iloc[0]['Open']
    week_close = df.iloc[-1]['Close']
    last_day = df.iloc[-1]['Date'].date()
    return week_open, week_close, last_day

###############################
# Analysis Date Extraction
###############################
def extract_analysis_date(file_text):
    match = re.search(r"on\s+\w{3}\s+(\d{4}-\d{2}-\d{2})\s+at", file_text)
    if match:
        date_str = match.group(1)
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return None
    return None

###############################
# Market Sentiment Overview
###############################
def compute_market_sentiment(comments):
    sia = SentimentIntensityAnalyzer()
    criteria = {
        "Political": ["biden", "trump", "politics", "government"],
        "Economic": ["economy", "recession", "growth", "inflation"],
        "Market Mood": ["market", "stock market", "bullish", "bearish", "optimistic", "pessimistic"],
        "Volatility": ["crash", "dip", "rally", "correction", "volatility"]
    }
    results = []
    for crit, keywords in criteria.items():
        filtered = [c for c in comments if any(kw in c.lower() for kw in keywords)]
        count = len(filtered)
        if count > 0:
            scores = [sia.polarity_scores(c)['compound'] for c in filtered]
            avg_score = sum(scores) / count
        else:
            avg_score = None
        if avg_score is None:
            interp = "No Data"
        elif avg_score > 0.05:
            interp = "High"
        elif avg_score < -0.05:
            interp = "Low"
        else:
            interp = "Neutral"
        results.append({
            "Criteria": crit,
            "Count": count,
            "Average Sentiment": round(avg_score, 3) if avg_score is not None else "N/A",
            "Interpretation": interp
        })
    return pd.DataFrame(results)

###############################
# PDF Export Function
###############################
def df_to_pdf(df: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=8)
    col_width = pdf.epw / len(df.columns)
    for col in df.columns:
        pdf.cell(col_width, 10, col, border=1)
    pdf.ln(10)
    for _, row in df.iterrows():
        for item in row:
            pdf.cell(col_width, 10, str(item), border=1)
        pdf.ln(10)
    output = pdf.output(dest="S")
    return bytes(output)

###############################
# Streamlit Interface
###############################
st.title("Stock Sentiment & Price Correlation Analyzer")

# Sidebar: Optionally upload ticker list CSV.
uploaded_ticker_csv = st.sidebar.file_uploader("Upload ticker list CSV (optional)", type=["csv"], key="ticker_csv")
if uploaded_ticker_csv is not None:
    try:
        ticker_aliases = load_ticker_csv(uploaded_ticker_csv)
        st.sidebar.success("Ticker list loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading ticker CSV: {e}")
else:
    st.sidebar.info("Using default ticker aliases.")

# Choose the file type for comments.
file_type = st.selectbox("Select file type", options=["HTML", "CSV", "JSON"])
if file_type == "HTML":
    uploaded_file = st.file_uploader("Upload an HTML file", type=["html", "htm"])
elif file_type == "CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
else:
    uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])

if uploaded_file is not None:
    try:
        discussion_type = None
        if file_type == "HTML":
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            file_text = stringio.read()
            analysis_date = extract_analysis_date(file_text)
            if analysis_date is None:
                st.warning("No analysis date found in HTML. Using today's date.")
                analysis_date = datetime.date.today()
            elif analysis_date > datetime.date.today():
                st.warning(f"Analysis date ({analysis_date}) is in the future; using today's date.")
                analysis_date = datetime.date.today()
            st.write(f"**Analysis date:** {analysis_date}")
            comments, _ = parse_html_file(file_text)
        elif file_type == "CSV":
            comments = parse_csv_file(uploaded_file)
            analysis_date = st.date_input("Select Analysis Date", value=datetime.date.today())
            st.write(f"**Analysis date:** {analysis_date}")
        else:
            comments, analysis_date, discussion_type = parse_json_file(uploaded_file)
            if analysis_date is None:
                st.warning("No analysis date found in JSON. Using today's date.")
                analysis_date = datetime.date.today()
            elif analysis_date > datetime.date.today():
                st.warning(f"Analysis date ({analysis_date}) is in the future; using today's date.")
                analysis_date = datetime.date.today()
            st.write(f"**Analysis date:** {analysis_date}")
        
        if discussion_type is None:
            discussion_type = "weekend" if analysis_date.weekday() >= 5 else "daily"
        st.write(f"**Discussion type:** {discussion_type}")
        
        if discussion_type == "weekend" or analysis_date.weekday() >= 5:
            if analysis_date.weekday() == 5:
                next_trading_date = analysis_date + datetime.timedelta(days=2)
                prev_trading_date = analysis_date - datetime.timedelta(days=1)
            elif analysis_date.weekday() == 6:
                next_trading_date = analysis_date + datetime.timedelta(days=1)
                prev_trading_date = analysis_date - datetime.timedelta(days=2)
            else:
                next_trading_date = analysis_date
                prev_trading_date = analysis_date
        else:
            next_trading_date = analysis_date
            prev_trading_date = analysis_date
        
        st.write(f"**Next Trading Date (for next day data):** {next_trading_date}")
        st.write(f"**Previous Trading Date (for weekly data):** {prev_trading_date}")
        
        if not comments:
            st.error("No comments found. Check the file structure.")
        else:
            st.success(f"Parsed {len(comments)} comments.")
            
            market_sent_df = compute_market_sentiment(comments)
            st.subheader("Market Sentiment Overview")
            st.table(market_sent_df)
            
            overall_sent, sentiment_scores = analyze_sentiment(comments)
            st.write(f"**Overall average sentiment:** {overall_sent:.3f}")
            
            top_stocks = get_top_stocks_and_sentiment(comments, sentiment_scores, ticker_aliases)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            n_tickers = len(top_stocks)
            start_time = time.time()
            next_date_str = next_trading_date.strftime("%Y-%m-%d")
            prev_date_str = prev_trading_date.strftime("%Y-%m-%d")
            
            for i, (ticker, data) in enumerate(top_stocks.items(), start=1):
                try:
                    nd_open, nd_close, nd_date = get_next_day_open_close(ticker, next_date_str)
                    if data["sentiment_class"] == "bullish":
                        nd_corr = "Yes" if nd_close > nd_open else "No"
                    elif data["sentiment_class"] == "bearish":
                        nd_corr = "Yes" if nd_close < nd_open else "No"
                    else:
                        nd_corr = "N/A"
                except Exception as e:
                    nd_open, nd_close, nd_corr = "N/A", "N/A", f"Error: {e}"
                
                try:
                    wk_open, wk_close, wk_last_day = get_week_open_close(ticker, prev_date_str)
                    if data["sentiment_class"] == "bullish":
                        wk_corr = "Yes" if wk_close > wk_open else "No"
                    elif data["sentiment_class"] == "bearish":
                        wk_corr = "Yes" if wk_close < wk_open else "No"
                    else:
                        wk_corr = "N/A"
                except Exception as e:
                    wk_open, wk_close, wk_corr = "N/A", "N/A", f"Error: {e}"
                
                results.append({
                    "Ticker": ticker,
                    "Mentions": data["mentions"],
                    "Avg Sentiment": round(data["average_sentiment"], 3),
                    "Sentiment": data["sentiment_class"],
                    "Next Day Open": round(nd_open, 2) if isinstance(nd_open, (int, float)) else nd_open,
                    "Next Day Close": round(nd_close, 2) if isinstance(nd_close, (int, float)) else nd_close,
                    "Next Day Correlation": nd_corr,
                    "Week Open": round(wk_open, 2) if isinstance(wk_open, (int, float)) else wk_open,
                    "Week Close": round(wk_close, 2) if isinstance(wk_close, (int, float)) else wk_close,
                    "Week Correlation": wk_corr
                })
                
                progress = int((i / n_tickers) * 100)
                elapsed = time.time() - start_time
                eta = int((elapsed / i) * (n_tickers - i)) if i > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"Processing top stocks... {progress}% complete. ETA: {eta} seconds")
            
            status_text.text("Processing complete.")
            
            if results:
                df_results = pd.DataFrame(results)
                st.subheader("Top 10 Stocks Analysis")
                st.table(df_results)
                
                # Export Buttons
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("Export CSV", data=csv_data, file_name="top_stocks_analysis.csv", mime="text/csv")
                
                pdf_data = df_to_pdf(df_results)
                st.download_button("Export PDF", data=pdf_data, file_name="top_stocks_analysis.pdf", mime="application/pdf")
                
                # Existing Scatter Plot: Avg Sentiment vs Next Day Close
                st.subheader("Avg Sentiment vs. Next Day Close")
                scatter_fig = px.scatter(
                    df_results, x="Avg Sentiment", y="Next Day Close", color="Sentiment",
                    size="Mentions", hover_data=["Ticker"],
                    title="Avg Sentiment vs. Next Day Close"
                )
                x_min = df_results["Avg Sentiment"].min()
                x_max = df_results["Avg Sentiment"].max()
                y_min = df_results["Next Day Close"].min()
                y_max = df_results["Next Day Close"].max()
                median_close = df_results["Next Day Close"].median()
                x_margin = (x_max - x_min) * 0.1 if (x_max - x_min) != 0 else 0.1
                y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1
                scatter_fig.update_xaxes(range=[x_min - x_margin, x_max + x_margin])
                scatter_fig.update_yaxes(range=[y_min - y_margin, y_max + y_margin])
                st.plotly_chart(scatter_fig)
                
                # Mentions Bar Chart
                st.subheader("Mentions Bar Chart")
                bar_data = pd.DataFrame({
                    "Ticker": list(top_stocks.keys()),
                    "Mentions": [data["mentions"] for data in top_stocks.values()],
                    "Sentiment": [data["sentiment_class"] for data in top_stocks.values()]
                })
                bar_fig = px.bar(bar_data, x="Ticker", y="Mentions", color="Sentiment",
                                 title="Top Tickers by Mentions")
                st.plotly_chart(bar_fig)
                
                # Sentiment Distribution Histogram
                st.subheader("Sentiment Distribution Histogram")
                hist_data = pd.DataFrame({"Sentiment Score": sentiment_scores})
                hist_fig = px.histogram(hist_data, x="Sentiment Score", nbins=20,
                                        title="Distribution of Sentiment Scores")
                st.plotly_chart(hist_fig)
                
                # Price Change vs. Sentiment Bubble Chart
                st.subheader("Price Change vs. Sentiment Bubble Chart")
                bubble_data = []
                for ticker, data in top_stocks.items():
                    try:
                        nd_open, nd_close, _ = get_next_day_open_close(ticker, next_date_str)
                        pct_change = ((nd_close - nd_open) / nd_open) * 100 if nd_open != 0 else 0
                    except Exception as e:
                        pct_change = None
                    bubble_data.append({
                        "Ticker": ticker,
                        "Avg Sentiment": data["average_sentiment"],
                        "Mentions": data["mentions"],
                        "Pct Change": pct_change,
                        "Sentiment": data["sentiment_class"]
                    })
                bubble_df = pd.DataFrame(bubble_data)
                bubble_fig = px.scatter(bubble_df, x="Avg Sentiment", y="Pct Change", size="Mentions",
                                        color="Sentiment", hover_data=["Ticker"],
                                        title="Price Change vs. Sentiment Bubble Chart")
                st.plotly_chart(bubble_fig)
            else:
                st.error("No stock data computed.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to begin the analysis.")



# import datetime
# import re
# import io
# import json
# import time
# import yfinance as yf
# import streamlit as st
# import pandas as pd
# from bs4 import BeautifulSoup
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import plotly.express as px
# from fpdf import FPDF  # pip install fpdf

# # Download the VADER lexicon if needed.
# nltk.download('vader_lexicon')

# ###############################
# # Global Blacklist for Common Word Tickers
# ###############################
# common_ticker_blacklist = {"IT", "ALL", "NOW", "ON", "SO", "BRO", "WELL", "PM"}

# ###############################
# # Ticker Aliases Dictionary
# ###############################
# default_ticker_aliases = {
#     "AAPL": ["aapl", "apple", "apple inc"],
#     "MSFT": ["msft", "microsoft", "microsoft corporation"],
#     "NVDA": ["nvda", "nvidia", "nvidia corporation"],
#     "AMZN": ["amzn", "amazon", "amazon.com"],
#     "GOOGL": ["googl", "alphabet", "google"],
#     "META": ["meta", "meta platforms", "facebook"],
#     "TSLA": ["tsla", "tesla", "tesla inc"],
#     "SPY": ["spy", "s&p500", "s&p 500", "sp500"],
#     "QQQ": ["qqq", "nasdaq 100"],
#     # ... add more as needed.
# }
# ticker_aliases = default_ticker_aliases.copy()

# ###############################
# # Function to Load Ticker CSV
# ###############################
# def load_ticker_csv(file) -> dict:
#     df = pd.read_csv(file)
#     aliases = {}
#     for _, row in df.iterrows():
#         symbol = str(row["Symbol"]).strip()
#         name = str(row["Name"]).strip()
#         if symbol and len(symbol) > 1 and symbol.upper() not in common_ticker_blacklist:
#             aliases[symbol] = [symbol.lower(), name.lower()]
#     return aliases

# ###############################
# # Parsing Functions
# ###############################
# def parse_comments(text):
#     pattern = re.compile(
#         r"^(?P<header>.*?,\s+on\s+\w{3}\s+(?P<date>\d{4}-\d{2}-\d{2})\s+at\s+\d{2}:\d{2}.*)"
#         r"(?P<comment>(?:\n(?!.*?,\s+on\s+\w{3}\s+\d{4}-\d{2}-\d{2}\s+at\s+\d{2}:\d{2}).+)*)",
#         re.MULTILINE)
#     comments = []
#     dates = []
#     for match in pattern.finditer(text):
#         comment_text = match.group("comment").strip()
#         if comment_text:
#             comments.append(comment_text)
#             dates.append(match.group("date"))
#     return comments, dates

# def parse_html_file(file_text):
#     soup = BeautifulSoup(file_text, 'html.parser')
#     comment_divs = soup.find_all("div", class_="comment")
#     if comment_divs and len(comment_divs) > 5:
#         comments = [div.get_text(separator="\n", strip=True) for div in comment_divs]
#         return comments, []
#     body = soup.find("body")
#     text_content = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
#     return parse_comments(text_content)

# def parse_csv_file(uploaded_file):
#     df = pd.read_csv(uploaded_file)
#     comment_col = None
#     for col in df.columns:
#         if col.lower() == "comment":
#             comment_col = col
#             break
#     if comment_col is None:
#         comment_col = df.columns[0]
#     return df[comment_col].astype(str).tolist()

# def parse_json_file(uploaded_file):
#     json_data = json.load(uploaded_file)
#     comments = []
#     analysis_date = None
#     discussion_type = None
#     for item in json_data:
#         if analysis_date is None and item.get("dataType") == "post":
#             title = item.get("title", "")
#             if "discussion thread for" in title.lower():
#                 if "weekend" in title.lower():
#                     discussion_type = "weekend"
#                 elif "daily" in title.lower():
#                     discussion_type = "daily"
#                 m = re.search(r"([A-Za-z]+ \d{1,2}, \d{4})", title)
#                 if m:
#                     try:
#                         analysis_date = datetime.datetime.strptime(m.group(1), "%B %d, %Y").date()
#                     except Exception:
#                         pass
#         if item.get("dataType") == "comment":
#             body = item.get("body", "")
#             if body:
#                 comments.append(body)
#     return comments, analysis_date, discussion_type

# ###############################
# # Sentiment Analysis
# ###############################
# def analyze_sentiment(comments):
#     sia = SentimentIntensityAnalyzer()
#     scores = [sia.polarity_scores(text)['compound'] for text in comments]
#     overall = sum(scores) / len(scores) if scores else 0
#     return overall, scores

# ###############################
# # Ticker Extraction & Top Stocks
# ###############################
# def extract_ticker_mentions_with_alias(text, ticker_aliases):
#     text_lower = text.lower()
#     found = set()
#     for ticker, aliases in ticker_aliases.items():
#         if len(ticker) == 1 or ticker.upper() in common_ticker_blacklist:
#             continue
#         for alias in aliases:
#             pattern = r"\b" + re.escape(alias) + r"\b"
#             if re.search(pattern, text_lower):
#                 found.add(ticker)
#                 break
#     return list(found)

# def get_top_stocks_and_sentiment(comments, sentiment_scores, ticker_aliases):
#     ticker_counts = {}
#     ticker_sentiments = {}
#     for comment, score in zip(comments, sentiment_scores):
#         tickers = extract_ticker_mentions_with_alias(comment, ticker_aliases)
#         for ticker in tickers:
#             ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
#             ticker_sentiments.setdefault(ticker, []).append(score)
#     top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
#     results = {}
#     for ticker, count in top_tickers:
#         sentiments = ticker_sentiments[ticker]
#         avg_sent = sum(sentiments) / len(sentiments)
#         if avg_sent > 0.05:
#             sentiment_class = "bullish"
#         elif avg_sent < -0.05:
#             sentiment_class = "bearish"
#         else:
#             sentiment_class = "neutral"
#         results[ticker] = {
#             "mentions": count,
#             "average_sentiment": avg_sent,
#             "sentiment_class": sentiment_class
#         }
#     return results

# ###############################
# # Trading Data Retrieval (with caching)
# ###############################
# @st.cache_data(show_spinner=False)
# def get_next_day_open_close(ticker, base_date_str):
#     base_date = datetime.datetime.strptime(base_date_str, "%Y-%m-%d").date()
#     end_date = base_date + datetime.timedelta(days=10)
#     stock = yf.Ticker(ticker)
#     hist = stock.history(start=base_date.strftime("%Y-%m-%d"),
#                          end=end_date.strftime("%Y-%m-%d"))
#     if hist.empty or len(hist) < 2:
#         raise ValueError(f"Not enough trading data for {ticker}.")
#     df = hist.reset_index()
#     df = df[df['Date'].dt.date >= base_date]
#     if len(df) < 2:
#         raise ValueError(f"Not enough trading data for {ticker} after {base_date}.")
#     next_day_row = df.iloc[1]
#     return next_day_row['Open'], next_day_row['Close'], next_day_row['Date'].date()

# @st.cache_data(show_spinner=False)
# def get_week_open_close(ticker, base_date_str):
#     base_date = datetime.datetime.strptime(base_date_str, "%Y-%m-%d").date()
#     end_date = base_date + datetime.timedelta(days=10)
#     stock = yf.Ticker(ticker)
#     hist = stock.history(start=base_date.strftime("%Y-%m-%d"),
#                          end=end_date.strftime("%Y-%m-%d"))
#     if hist.empty or len(hist) < 2:
#         raise ValueError(f"Not enough trading data for {ticker} to compute weekly data.")
#     df = hist.reset_index()
#     df = df[df['Date'].dt.date >= base_date]
#     if df.empty:
#         raise ValueError(f"No trading data for {ticker} after {base_date}.")
#     week_open = df.iloc[0]['Open']
#     week_close = df.iloc[-1]['Close']
#     last_day = df.iloc[-1]['Date'].date()
#     return week_open, week_close, last_day

# ###############################
# # Analysis Date Extraction
# ###############################
# def extract_analysis_date(file_text):
#     match = re.search(r"on\s+\w{3}\s+(\d{4}-\d{2}-\d{2})\s+at", file_text)
#     if match:
#         date_str = match.group(1)
#         try:
#             return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
#         except Exception:
#             return None
#     return None

# ###############################
# # Market Sentiment Overview
# ###############################
# def compute_market_sentiment(comments):
#     sia = SentimentIntensityAnalyzer()
#     criteria = {
#         "Political": ["biden", "trump", "politics", "government"],
#         "Economic": ["economy", "recession", "growth", "inflation"],
#         "Market Mood": ["market", "stock market", "bullish", "bearish", "optimistic", "pessimistic"],
#         "Volatility": ["crash", "dip", "rally", "correction", "volatility"]
#     }
#     results = []
#     for crit, keywords in criteria.items():
#         filtered = [c for c in comments if any(kw in c.lower() for kw in keywords)]
#         count = len(filtered)
#         if count > 0:
#             scores = [sia.polarity_scores(c)['compound'] for c in filtered]
#             avg_score = sum(scores) / count
#         else:
#             avg_score = None
#         if avg_score is None:
#             interp = "No Data"
#         elif avg_score > 0.05:
#             interp = "High"
#         elif avg_score < -0.05:
#             interp = "Low"
#         else:
#             interp = "Neutral"
#         results.append({
#             "Criteria": crit,
#             "Count": count,
#             "Average Sentiment": round(avg_score, 3) if avg_score is not None else "N/A",
#             "Interpretation": interp
#         })
#     return pd.DataFrame(results)

# ###############################
# # PDF Export Function
# ###############################
# def df_to_pdf(df: pd.DataFrame) -> bytes:
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=8)
#     col_width = pdf.epw / len(df.columns)
#     # Write header
#     for col in df.columns:
#         pdf.cell(col_width, 10, col, border=1)
#     pdf.ln(10)
#     # Write data rows
#     for _, row in df.iterrows():
#         for item in row:
#             pdf.cell(col_width, 10, str(item), border=1)
#         pdf.ln(10)
#     output = pdf.output(dest="S")
#     # Ensure output is of type bytes
#     return bytes(output)

# ###############################
# # Streamlit Interface
# ###############################
# st.title("Stock Sentiment & Price Correlation Analyzer")

# # Sidebar: Optionally upload ticker list CSV.
# uploaded_ticker_csv = st.sidebar.file_uploader("Upload ticker list CSV (optional)", type=["csv"], key="ticker_csv")
# if uploaded_ticker_csv is not None:
#     try:
#         ticker_aliases = load_ticker_csv(uploaded_ticker_csv)
#         st.sidebar.success("Ticker list loaded successfully.")
#     except Exception as e:
#         st.sidebar.error(f"Error loading ticker CSV: {e}")
# else:
#     st.sidebar.info("Using default ticker aliases.")

# # Let the user choose the file type for comments.
# file_type = st.selectbox("Select file type", options=["HTML", "CSV", "JSON"])
# if file_type == "HTML":
#     uploaded_file = st.file_uploader("Upload an HTML file", type=["html", "htm"])
# elif file_type == "CSV":
#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
# else:
#     uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])

# if uploaded_file is not None:
#     try:
#         discussion_type = None
#         if file_type == "HTML":
#             stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
#             file_text = stringio.read()
#             analysis_date = extract_analysis_date(file_text)
#             if analysis_date is None:
#                 st.warning("No analysis date found in HTML. Using today's date.")
#                 analysis_date = datetime.date.today()
#             elif analysis_date > datetime.date.today():
#                 st.warning(f"Analysis date ({analysis_date}) is in the future; using today's date.")
#                 analysis_date = datetime.date.today()
#             st.write(f"**Analysis date:** {analysis_date}")
#             comments, _ = parse_html_file(file_text)
#         elif file_type == "CSV":
#             comments = parse_csv_file(uploaded_file)
#             analysis_date = st.date_input("Select Analysis Date", value=datetime.date.today())
#             st.write(f"**Analysis date:** {analysis_date}")
#         else:
#             comments, analysis_date, discussion_type = parse_json_file(uploaded_file)
#             if analysis_date is None:
#                 st.warning("No analysis date found in JSON. Using today's date.")
#                 analysis_date = datetime.date.today()
#             elif analysis_date > datetime.date.today():
#                 st.warning(f"Analysis date ({analysis_date}) is in the future; using today's date.")
#                 analysis_date = datetime.date.today()
#             st.write(f"**Analysis date:** {analysis_date}")
        
#         if discussion_type is None:
#             discussion_type = "weekend" if analysis_date.weekday() >= 5 else "daily"
#         st.write(f"**Discussion type:** {discussion_type}")
        
#         if discussion_type == "weekend" or analysis_date.weekday() >= 5:
#             if analysis_date.weekday() == 5:
#                 next_trading_date = analysis_date + datetime.timedelta(days=2)
#                 prev_trading_date = analysis_date - datetime.timedelta(days=1)
#             elif analysis_date.weekday() == 6:
#                 next_trading_date = analysis_date + datetime.timedelta(days=1)
#                 prev_trading_date = analysis_date - datetime.timedelta(days=2)
#             else:
#                 next_trading_date = analysis_date
#                 prev_trading_date = analysis_date
#         else:
#             next_trading_date = analysis_date
#             prev_trading_date = analysis_date
        
#         st.write(f"**Next Trading Date (for next day data):** {next_trading_date}")
#         st.write(f"**Previous Trading Date (for weekly data):** {prev_trading_date}")
        
#         if not comments:
#             st.error("No comments found. Check the file structure.")
#         else:
#             st.success(f"Parsed {len(comments)} comments.")
            
#             market_sent_df = compute_market_sentiment(comments)
#             st.subheader("Market Sentiment Overview")
#             st.table(market_sent_df)
            
#             overall_sent, sentiment_scores = analyze_sentiment(comments)
#             st.write(f"**Overall average sentiment:** {overall_sent:.3f}")
            
#             top_stocks = get_top_stocks_and_sentiment(comments, sentiment_scores, ticker_aliases)
            
#             progress_bar = st.progress(0)
#             status_text = st.empty()
#             results = []
#             n_tickers = len(top_stocks)
#             start_time = time.time()
#             next_date_str = next_trading_date.strftime("%Y-%m-%d")
#             prev_date_str = prev_trading_date.strftime("%Y-%m-%d")
            
#             for i, (ticker, data) in enumerate(top_stocks.items(), start=1):
#                 try:
#                     nd_open, nd_close, nd_date = get_next_day_open_close(ticker, next_date_str)
#                     if data["sentiment_class"] == "bullish":
#                         nd_corr = "Yes" if nd_close > nd_open else "No"
#                     elif data["sentiment_class"] == "bearish":
#                         nd_corr = "Yes" if nd_close < nd_open else "No"
#                     else:
#                         nd_corr = "N/A"
#                 except Exception as e:
#                     nd_open, nd_close, nd_corr = "N/A", "N/A", f"Error: {e}"
                
#                 try:
#                     wk_open, wk_close, wk_last_day = get_week_open_close(ticker, prev_date_str)
#                     if data["sentiment_class"] == "bullish":
#                         wk_corr = "Yes" if wk_close > wk_open else "No"
#                     elif data["sentiment_class"] == "bearish":
#                         wk_corr = "Yes" if wk_close < wk_open else "No"
#                     else:
#                         wk_corr = "N/A"
#                 except Exception as e:
#                     wk_open, wk_close, wk_corr = "N/A", "N/A", f"Error: {e}"
                
#                 results.append({
#                     "Ticker": ticker,
#                     "Mentions": data["mentions"],
#                     "Avg Sentiment": round(data["average_sentiment"], 3),
#                     "Sentiment": data["sentiment_class"],
#                     "Next Day Open": round(nd_open, 2) if isinstance(nd_open, (int, float)) else nd_open,
#                     "Next Day Close": round(nd_close, 2) if isinstance(nd_close, (int, float)) else nd_close,
#                     "Next Day Correlation": nd_corr,
#                     "Week Open": round(wk_open, 2) if isinstance(wk_open, (int, float)) else wk_open,
#                     "Week Close": round(wk_close, 2) if isinstance(wk_close, (int, float)) else wk_close,
#                     "Week Correlation": wk_corr
#                 })
                
#                 progress = int((i / n_tickers) * 100)
#                 elapsed = time.time() - start_time
#                 eta = int((elapsed / i) * (n_tickers - i)) if i > 0 else 0
#                 progress_bar.progress(progress)
#                 status_text.text(f"Processing top stocks... {progress}% complete. ETA: {eta} seconds")
            
#             status_text.text("Processing complete.")
            
#             if results:
#                 df_results = pd.DataFrame(results)
#                 st.subheader("Top 10 Stocks Analysis")
#                 st.table(df_results)
                
#                 csv_data = df_results.to_csv(index=False).encode('utf-8')
#                 st.download_button("Export CSV", data=csv_data, file_name="top_stocks_analysis.csv", mime="text/csv")
                
#                 pdf_data = df_to_pdf(df_results)
#                 st.download_button("Export PDF", data=pdf_data, file_name="top_stocks_analysis.pdf", mime="application/pdf")
                
#                 st.subheader("Avg Sentiment vs. Next Day Close")
#                 fig = px.scatter(
#                     df_results, x="Avg Sentiment", y="Next Day Close", color="Sentiment",
#                     size="Mentions", hover_data=["Ticker"],
#                     title="Avg Sentiment vs. Next Day Close"
#                 )
                
#                 x_min = df_results["Avg Sentiment"].min()
#                 x_max = df_results["Avg Sentiment"].max()
#                 y_min = df_results["Next Day Close"].min()
#                 y_max = df_results["Next Day Close"].max()
#                 median_close = df_results["Next Day Close"].median()
                
#                 x_margin = (x_max - x_min) * 0.1 if (x_max - x_min) != 0 else 0.1
#                 y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1
#                 x_min_adj = x_min - x_margin
#                 x_max_adj = x_max + x_margin
#                 y_min_adj = y_min - y_margin
#                 y_max_adj = y_max + y_margin
                
#                 fig.update_xaxes(range=[x_min_adj, x_max_adj])
#                 fig.update_yaxes(range=[y_min_adj, y_max_adj])
                
#                 fig.add_shape(type="line", x0=0, x1=0, y0=y_min_adj, y1=y_max_adj,
#                               line=dict(color="black", dash="dash"))
#                 fig.add_shape(type="line", x0=x_min_adj, x1=x_max_adj, y0=median_close, y1=median_close,
#                               line=dict(color="black", dash="dash"))
                
#                 fig.add_shape(type="rect", x0=0, x1=x_max_adj, y0=median_close, y1=y_max_adj,
#                               fillcolor="green", opacity=0.2, layer="below", line_width=0)
#                 fig.add_shape(type="rect", x0=x_min_adj, x1=0, y0=median_close, y1=y_max_adj,
#                               fillcolor="red", opacity=0.2, layer="below", line_width=0)
#                 fig.add_shape(type="rect", x0=x_min_adj, x1=0, y0=y_min_adj, y1=median_close,
#                               fillcolor="red", opacity=0.2, layer="below", line_width=0)
#                 fig.add_shape(type="rect", x0=0, x1=x_max_adj, y0=y_min_adj, y1=median_close,
#                               fillcolor="green", opacity=0.2, layer="below", line_width=0)
                
#                 fig.add_annotation(x=(0+x_max_adj)/2, y=(median_close+y_max_adj)/2,
#                                    text="Strong Bullish", showarrow=False,
#                                    font=dict(color="green", size=12))
#                 fig.add_annotation(x=(x_min_adj+0)/2, y=(median_close+y_max_adj)/2,
#                                    text="Weak Bearish", showarrow=False,
#                                    font=dict(color="red", size=12))
#                 fig.add_annotation(x=(x_min_adj+0)/2, y=(y_min_adj+median_close)/2,
#                                    text="Strong Bearish", showarrow=False,
#                                    font=dict(color="red", size=12))
#                 fig.add_annotation(x=(0+x_max_adj)/2, y=(y_min_adj+median_close)/2,
#                                    text="Weak Bullish", showarrow=False,
#                                    font=dict(color="green", size=12))
                
#                 st.plotly_chart(fig)
                
#                 st.write("""
#                 **Quadrant Explanation:**
#                 - **Quadrant I (Top Right):** Positive sentiment & high next day close → Strong Bullish.
#                 - **Quadrant II (Top Left):** Negative sentiment but high next day close → Weak Bearish.
#                 - **Quadrant III (Bottom Left):** Negative sentiment & low next day close → Strong Bearish.
#                 - **Quadrant IV (Bottom Right):** Positive sentiment but low next day close → Weak Bullish.
#                 """)
#             else:
#                 st.error("No stock data computed.")
#     except Exception as e:
#         st.error(f"Error processing file: {e}")
# else:
#     st.info("Please upload a file to begin the analysis.")
