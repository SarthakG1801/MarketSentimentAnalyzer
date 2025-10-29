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
import requests
import pandas_market_calendars as mcal
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tweepy

def fetch_latest_tweets(query="#stocks OR $SPY OR $AAPL", max_results=100):
    """
    Fetch latest tweets based on a given query.
    Returns: list of tweet texts and the fetch date.
    """
    try:
        # Replace these with your actual credentials
        BEARER_TOKEN = st.secrets.get("TWITTER_BEARER_TOKEN", None)
        if not BEARER_TOKEN:
            st.error("⚠️ Twitter API bearer token not found. Please set TWITTER_BEARER_TOKEN in Streamlit secrets.")
            return [], None

        client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

        tweets = client.search_recent_tweets(
            query=query + " -is:retweet lang:en",
            tweet_fields=["created_at", "text"],
            max_results=max_results
        )

        # Safely extract tweet texts if results exist
        tweet_texts = []
        if tweets and hasattr(tweets, "data") and tweets.data:
            tweet_texts = [getattr(t, "text", "") for t in tweets.data if hasattr(t, "text")]

        fetch_date = datetime.date.today()
        return tweet_texts, fetch_date

    except Exception as e:
        st.error(f"❌ Error fetching tweets: {e}")
        return [], None


def fetch_latest_daily_discussion():
    """
    Fetch the latest discussion threads from multiple subreddits.
    Returns combined comments, latest post date, and subreddit source(s).
    """
    import requests
    import datetime
    import streamlit as st

    subreddits = ["wallstreetbets", "stocks", "investing"]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketMoodRadar/1.0)"}

    all_comments = []
    fetched_from = []  # ✅ Track successful subreddits
    latest_date = None

    try:
        for sub in subreddits:
            reddit_url = f"https://www.reddit.com/r/{sub}/hot.json?limit=10"
            res = requests.get(reddit_url, headers=headers, timeout=10)
            res.raise_for_status()
            posts = res.json().get("data", {}).get("children", [])

            daily_thread = None
            for post in posts:
                title = post["data"]["title"]
                if "discussion" in title.lower():
                    daily_thread = post["data"]
                    break

            if not daily_thread:
                st.warning(f"⚠️ No discussion thread found in r/{sub}, skipping...")
                continue

            post_date = datetime.datetime.utcfromtimestamp(daily_thread["created_utc"]).date()
            post_url = f"https://www.reddit.com{daily_thread['permalink']}.json"

            comments_res = requests.get(post_url, headers=headers, timeout=10)
            comments_res.raise_for_status()
            comments_json = comments_res.json()
            comments_data = comments_json[1]["data"]["children"]

            comments = [
                c["data"]["body"]
                for c in comments_data
                if c["kind"] == "t1" and "body" in c["data"]
            ]

            if comments:
                all_comments.extend(comments)
                fetched_from.append(sub)  # ✅ Record successful subreddit
                st.success(f"✅ Fetched {len(comments)} comments from r/{sub} ({post_date})")

            if latest_date is None or post_date > latest_date:
                latest_date = post_date

        if not all_comments:
            st.warning("⚠️ No comments found in any subreddit. Trying Pushshift fallback...")
            return fetch_from_pushshift_fallback()

        # ✅ Return all successful subreddit names as the source
        return all_comments, latest_date, ", ".join(fetched_from)

    except Exception as e:
        st.error(f"Reddit multi-fetch failed: {e}")
        return [], None, None

    """
    Fetch the latest 'Daily Discussion Thread' or popular posts from multiple subreddits
    using Reddit's public JSON API. Falls back to Pushshift if Reddit fails.
    Returns: combined comments, latest post date, and discussion type.
    """
    import requests
    import datetime

    subreddits = ["wallstreetbets", "stocks", "investing"]  # ✅ Add any others you want
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketMoodRadar/1.0)"}

    all_comments = []
    latest_date = None

    try:
        for sub in subreddits:
            reddit_url = f"https://www.reddit.com/r/{sub}/hot.json?limit=10"
            res = requests.get(reddit_url, headers=headers, timeout=10)
            res.raise_for_status()
            posts = res.json().get("data", {}).get("children", [])

            daily_thread = None
            for post in posts:
                title = post["data"]["title"]
                # Match any "Daily Discussion Thread" or "Discussion"
                if "discussion" in title.lower():
                    daily_thread = post["data"]
                    break

            if not daily_thread:
                st.warning(f"⚠️ No discussion thread found in r/{sub}, skipping...")
                continue

            post_date = datetime.datetime.utcfromtimestamp(
                daily_thread["created_utc"]
            ).date()
            post_url = f"https://www.reddit.com{daily_thread['permalink']}.json"

            comments_res = requests.get(post_url, headers=headers, timeout=10)
            comments_res.raise_for_status()
            comments_json = comments_res.json()
            comments_data = comments_json[1]["data"]["children"]

            comments = [
                c["data"]["body"]
                for c in comments_data
                if c["kind"] == "t1" and "body" in c["data"]
            ]

            all_comments.extend(comments)

            if latest_date is None or post_date > latest_date:
                latest_date = post_date

        if not all_comments:
            st.warning("No comments found from any subreddit. Trying Pushshift fallback...")
            return fetch_from_pushshift_fallback()

        return all_comments, latest_date, "multi-subreddit"

    except Exception as e:
        st.error(f"Reddit multi-fetch failed: {e}")
        return [], None, None

    """
    Fetch the latest 'Daily Discussion Thread' from r/wallstreetbets using Reddit's public JSON API.
    Falls back to Pushshift if Reddit fails.
    Returns: comments, post_date, discussion_type
    """
    import requests
    import datetime
    import re

    try:
        # Step 1: Fetch subreddit hot posts from Reddit
        reddit_url = "https://www.reddit.com/r/wallstreetbets/hot.json?limit=10"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketMoodRadar/1.0)"}
        res = requests.get(reddit_url, headers=headers, timeout=10)
        res.raise_for_status()
        posts = res.json()["data"]["children"]

        daily_thread = None
        for post in posts:
            title = post["data"]["title"]
            if "Daily Discussion Thread" in title:
                daily_thread = post["data"]
                break

        if not daily_thread:
            st.warning("Could not find today's discussion thread in Reddit's live feed. Trying Pushshift fallback...")
            return fetch_from_pushshift_fallback()

        post_id = daily_thread["id"]
        post_date = datetime.datetime.utcfromtimestamp(daily_thread["created_utc"]).date()
        post_url = f"https://www.reddit.com{daily_thread['permalink']}.json"

        # Step 2: Fetch comments
        comments_res = requests.get(post_url, headers=headers, timeout=10)
        comments_res.raise_for_status()
        comments_json = comments_res.json()

        # Comments are in the second element of the list
        comments_data = comments_json[1]["data"]["children"]
        comments = [
            c["data"]["body"]
            for c in comments_data
            if c["kind"] == "t1" and "body" in c["data"]
        ]

        return comments, post_date, "daily"

    except Exception as e:
        st.error(f"Reddit fetch failed: {e}")
        return [], None, None


def fetch_from_pushshift_fallback():
    """Fallback to Pushshift API if Reddit JSON fails."""
    url_post = "https://api.pullpush.io/reddit/search/submission/"
    params_post = {
        "subreddit": "wallstreetbets",
        "title": "Daily Discussion Thread",
        "sort": "desc",
        "sort_type": "created_utc",
        "size": 1,
    }
    try:
        post_response = requests.get(url_post, params=params_post, timeout=10)
        post_data = post_response.json().get("data", [])
        if not post_data:
            return [], None, None

        post = post_data[0]
        post_id = post["id"]
        post_date = datetime.datetime.utcfromtimestamp(post["created_utc"]).date()

        url_comments = "https://api.pullpush.io/reddit/comment/search/"
        params_comments = {
            "link_id": post_id,
            "subreddit": "wallstreetbets",
            "size": 1000,
            "sort": "asc",
        }
        comments_response = requests.get(url_comments, params=params_comments, timeout=10)
        comments_data = comments_response.json().get("data", [])
        comments = [c["body"] for c in comments_data if "body" in c]

        return comments, post_date, "daily"
    except Exception:
        return [], None, None

    """
    Fetches the latest 'Daily Discussion Thread' post and its comments from WallStreetBets using Pushshift API.
    Returns a list of comments and the post date.
    """
    print("Fetching latest Daily Discussion Thread from r/wallstreetbets...")

    # Step 1: Find the latest Daily Discussion Thread
    url_post = "https://api.pullpush.io/reddit/search/submission/"
    params_post = {
        "subreddit": "wallstreetbets",
        "title": "Daily Discussion Thread",
        "sort": "desc",
        "sort_type": "created_utc",
        "size": 1
    }

    post_response = requests.get(url_post, params=params_post)
    post_data = post_response.json()["data"]

    if not post_data:
        st.error("No Daily Discussion Thread found.")
        return [], None, None

    post = post_data[0]
    post_id = post["id"]
    post_date = datetime.datetime.utcfromtimestamp(post["created_utc"]).date()
    post_title = post.get("title", "Daily Discussion Thread")

    # Step 2: Fetch comments for that post
    url_comments = "https://api.pullpush.io/reddit/comment/search/"
    params_comments = {
        "link_id": post_id,
        "subreddit": "wallstreetbets",
        "size": 1000,  # adjust if needed
        "sort": "asc"
    }

    comments_response = requests.get(url_comments, params=params_comments)
    comments_data = comments_response.json()["data"]

    comments = [c["body"] for c in comments_data if "body" in c]

    return comments, post_date, "daily"


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
@st.cache_resource
def load_finbert_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def analyze_sentiment(comments):
    tokenizer, model,device = load_finbert_model()
    sentiments = []
    model.eval()

    for text in comments:
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}  # ✅ Move inputs to GPU if available
            with torch.no_grad():
                outputs = model(**inputs)

                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                label = torch.argmax(probs).item()
                # FinBERT labels: 0=negative, 1=neutral, 2=positive
                score = probs[0, 2].item() - probs[0, 0].item()  # positive minus negative
                sentiments.append(score)
        except Exception:
            sentiments.append(0.0)

    overall = sum(sentiments) / len(sentiments) if sentiments else 0
    return overall, sentiments


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
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=base_date - datetime.timedelta(days=5),
                             end_date=base_date + datetime.timedelta(days=10))
    valid_days = [d.date() for d in schedule.index]

    # If base_date not in valid_days, shift to last valid day
    if base_date not in valid_days:
        base_date = max([d for d in valid_days if d < base_date])

    next_days = [d for d in valid_days if d > base_date]
    if not next_days:
        raise ValueError(f"No next trading day found for {ticker} after {base_date}")
    next_day = next_days[0]

    stock = yf.Ticker(ticker)
    hist = stock.history(start=base_date.strftime("%Y-%m-%d"),
                         end=(next_day + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
    if hist.empty:
        raise ValueError(f"No data for {ticker} between {base_date} and {next_day}")
    nd_open = hist.iloc[-1]["Open"]
    nd_close = hist.iloc[-1]["Close"]
    return nd_open, nd_close, next_day


@st.cache_data(show_spinner=False)
def get_week_open_close(ticker, base_date_str):
    base_date = datetime.datetime.strptime(base_date_str, "%Y-%m-%d").date()
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=base_date - datetime.timedelta(days=7),
                             end_date=base_date + datetime.timedelta(days=7))
    valid_days = [d.date() for d in schedule.index]

    # Use the most recent valid trading day before base_date
    trading_days = [d for d in valid_days if d <= base_date]
    if not trading_days:
        raise ValueError(f"No trading data for {ticker} near {base_date}")

    start_day = trading_days[0]
    end_day = trading_days[-1]

    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_day.strftime("%Y-%m-%d"),
                         end=(end_day + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
    if hist.empty:
        raise ValueError(f"No trading data for {ticker} between {start_day} and {end_day}")
    week_open = hist.iloc[0]["Open"]
    week_close = hist.iloc[-1]["Close"]
    return week_open, week_close, end_day

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


st.sidebar.subheader("Data Source")
data_source = st.sidebar.radio(
    "Select source of discussion data:",
    ["Upload File", "Fetch Latest from Reddit", "Fetch Latest Tweets"]
)

st.sidebar.info("🧠 Using **FinBERT (Transformer-based)** model for sentiment analysis.")

if data_source == "Upload File":
    uploaded_file = st.file_uploader("Upload comments file (HTML / CSV / JSON)", type=["html", "csv", "json"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".json"):
            comments, analysis_date, discussion_type = parse_json_file(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            comments = parse_csv_file(uploaded_file)
            analysis_date, discussion_type = None, None
        else:
            file_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            comments, _ = parse_html_file(file_text)
            analysis_date, discussion_type = extract_analysis_date(file_text), None

elif data_source == "Fetch Latest Tweets":
    query = st.sidebar.text_input("Enter Twitter search query:", "#stocks OR $SPY OR $AAPL")
    max_results = st.sidebar.slider("Number of tweets to fetch", 10, 100, 50)

    with st.spinner("Fetching latest tweets..."):
        comments, analysis_date = fetch_latest_tweets(query=query, max_results=max_results)

    if not comments:
        st.error("❌ Could not fetch any tweets.")
    else:
        st.success(f"✅ Fetched {len(comments)} tweets for query: '{query}' ({analysis_date})")

        # Reuse your existing pipeline
        overall_sent, sentiment_scores = analyze_sentiment(comments)
        st.write(f"**Overall average sentiment:** {overall_sent:.3f}")

        top_stocks = get_top_stocks_and_sentiment(comments, sentiment_scores, ticker_aliases)
        if not top_stocks:
            st.warning("No stock mentions found in tweets.")
        else:
            st.subheader("Top Stocks Mentioned in Tweets")
            st.table(pd.DataFrame(top_stocks).T)

else:
    with st.spinner("Fetching latest WallStreetBets daily discussion..."):
        comments, analysis_date, discussion_type = fetch_latest_daily_discussion()

    if not comments:
        st.error("❌ Could not fetch Reddit comments.")
    else:
        if analysis_date is None:
            analysis_date = datetime.date.today()
        if analysis_date >= datetime.date.today():
            analysis_date -= datetime.timedelta(days=1)
        st.success(f"✅ Fetched {len(comments)} comments from r/wallstreetbets Daily Discussion ({analysis_date})")

        # Proceed directly with analysis pipeline
        discussion_type = "daily"
        next_trading_date = analysis_date
        prev_trading_date = analysis_date

        st.write(f"**Next Trading Date (for next day data):** {next_trading_date}")
        st.write(f"**Previous Trading Date (for weekly data):** {prev_trading_date}")

        market_sent_df = compute_market_sentiment(comments)
        st.subheader("Market Sentiment Overview")
        st.table(market_sent_df)

        overall_sent, sentiment_scores = analyze_sentiment(comments)
        st.write(f"**Overall average sentiment:** {overall_sent:.3f}")

        top_stocks = get_top_stocks_and_sentiment(comments, sentiment_scores, ticker_aliases)

        if not top_stocks:
            st.warning("No tickers mentioned frequently enough for analysis.")
        else:
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

            df_results = pd.DataFrame(results)
            st.subheader("Top 10 Stocks Analysis")
            st.table(df_results)

            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("Export CSV", data=csv_data, file_name="top_stocks_analysis.csv", mime="text/csv")

            # Quadrant Scatter Plot
            st.subheader("Avg Sentiment vs. Next Day Close (Quadrant View)")
            quad_fig = px.scatter(
                df_results, x="Avg Sentiment", y="Next Day Close", color="Sentiment",
                size="Mentions", hover_data=["Ticker"],
                title="Avg Sentiment vs. Next Day Close with Quadrants"
            )
            st.plotly_chart(quad_fig)

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
                except Exception:
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
                
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("Export CSV", data=csv_data, file_name="top_stocks_analysis.csv", mime="text/csv")
                
                # Quadrant Scatter Plot: Avg Sentiment vs. Next Day Close with Quadrants
                st.subheader("Avg Sentiment vs. Next Day Close (Quadrant View)")
                quad_fig = px.scatter(
                    df_results, x="Avg Sentiment", y="Next Day Close", color="Sentiment",
                    size="Mentions", hover_data=["Ticker"],
                    title="Avg Sentiment vs. Next Day Close with Quadrants"
                )
                # Calculate plot ranges
                x_min = df_results["Avg Sentiment"].min()
                x_max = df_results["Avg Sentiment"].max()
                y_min = df_results["Next Day Close"].min()
                y_max = df_results["Next Day Close"].max()
                median_close = df_results["Next Day Close"].median()
                x_margin = (x_max - x_min) * 0.1 if (x_max - x_min) != 0 else 0.1
                y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1
                x_min_adj = x_min - x_margin
                x_max_adj = x_max + x_margin
                y_min_adj = y_min - y_margin
                y_max_adj = y_max + y_margin
                
                quad_fig.update_xaxes(range=[x_min_adj, x_max_adj])
                quad_fig.update_yaxes(range=[y_min_adj, y_max_adj])
                
                # Add quadrant lines and rectangles
                quad_fig.add_shape(type="line", x0=0, x1=0, y0=y_min_adj, y1=y_max_adj,
                                   line=dict(color="black", dash="dash"))
                quad_fig.add_shape(type="line", x0=x_min_adj, x1=x_max_adj, y0=median_close, y1=median_close,
                                   line=dict(color="black", dash="dash"))
                quad_fig.add_shape(type="rect", x0=0, x1=x_max_adj, y0=median_close, y1=y_max_adj,
                                   fillcolor="green", opacity=0.2, layer="below", line_width=0)
                quad_fig.add_shape(type="rect", x0=x_min_adj, x1=0, y0=median_close, y1=y_max_adj,
                                   fillcolor="red", opacity=0.2, layer="below", line_width=0)
                quad_fig.add_shape(type="rect", x0=x_min_adj, x1=0, y0=y_min_adj, y1=median_close,
                                   fillcolor="red", opacity=0.2, layer="below", line_width=0)
                quad_fig.add_shape(type="rect", x0=0, x1=x_max_adj, y0=y_min_adj, y1=median_close,
                                   fillcolor="green", opacity=0.2, layer="below", line_width=0)
                
                quad_fig.add_annotation(x=(0+x_max_adj)/2, y=(median_close+y_max_adj)/2,
                                        text="Strong Bullish", showarrow=False,
                                        font=dict(color="green", size=12))
                quad_fig.add_annotation(x=(x_min_adj+0)/2, y=(median_close+y_max_adj)/2,
                                        text="Weak Bearish", showarrow=False,
                                        font=dict(color="red", size=12))
                quad_fig.add_annotation(x=(x_min_adj+0)/2, y=(y_min_adj+median_close)/2,
                                        text="Strong Bearish", showarrow=False,
                                        font=dict(color="red", size=12))
                quad_fig.add_annotation(x=(0+x_max_adj)/2, y=(y_min_adj+median_close)/2,
                                        text="Weak Bullish", showarrow=False,
                                        font=dict(color="green", size=12))
                st.plotly_chart(quad_fig)
                
                # Additional Visualizations
                st.subheader("Mentions Bar Chart")
                bar_data = pd.DataFrame({
                    "Ticker": list(top_stocks.keys()),
                    "Mentions": [data["mentions"] for data in top_stocks.values()],
                    "Sentiment": [data["sentiment_class"] for data in top_stocks.values()]
                })
                bar_fig = px.bar(bar_data, x="Ticker", y="Mentions", color="Sentiment",
                                 title="Top Tickers by Mentions")
                st.plotly_chart(bar_fig)
                
                st.subheader("Sentiment Distribution Histogram")
                hist_data = pd.DataFrame({"Sentiment Score": sentiment_scores})
                hist_fig = px.histogram(hist_data, x="Sentiment Score", nbins=20,
                                        title="Distribution of Sentiment Scores")
                st.plotly_chart(hist_fig)
                
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