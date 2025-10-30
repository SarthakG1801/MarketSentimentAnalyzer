📈 MarketPulse

MarketPulse is an AI-powered Streamlit dashboard that analyzes stock market sentiment from Reddit and Twitter in real time.
It combines FinBERT-based deep learning sentiment analysis, financial data from yfinance, and interactive data visualizations to reveal how public mood correlates with stock movements.

🚀 Features
🗣️ Multi-Source Sentiment Analysis

Fetches live discussions from multiple subreddits (r/wallstreetbets, r/stocks, r/investing, etc.).

Retrieves recent tweets using the Twitter API (or via snscrape fallback if the API is unavailable).

Merges insights from social chatter into a single, unified market sentiment view.

🤖 Deep Learning Model

Uses FinBERT (a finance-tuned BERT model) via Hugging Face Transformers.

Classifies sentiment as Positive, Negative, or Neutral.

Runs entirely on CPU or GPU (if available).

💰 Stock Price Integration

Fetches live stock data using yfinance.

Automatically adjusts trading dates for weekends and holidays using pandas_market_calendars.

Correlates sentiment with next-day or weekly price movements.

📊 Interactive Visualizations

Sentiment Distribution: Histogram of overall sentiment scores.

Top Mentions Chart: Frequency of most-discussed stocks.

Quadrant Scatter Plot: Sentiment vs. next-day performance.

Bubble Chart: Sentiment vs. price change, scaled by mention volume.

🧠 Smart Data Processing

Filters out invalid tickers and common words.

Supports input from Reddit JSON, tweets, or uploaded text files.

Includes Pushshift fallback for older Reddit threads.

🏗️ Project Structure
MarketPulse/
│
├── app.py                              # Streamlit main application
├── requirements.txt                    # All dependencies
├── Top 1000 Companies Ranked by Market Cap.csv
├── Top 10000 Companies Ranked by Market Cap.csv
├── Daily Discussion Thread for 1.21.25.json  # Example Reddit data
└── README.md

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/SarthakG1801/MarketPulse.git
cd MarketPulse

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

3️⃣ Install Dependencies
pip install -r requirements.txt

🔑 API Setup

Create a .streamlit/secrets.toml file in your project root and add:

TWITTER_BEARER_TOKEN = "your_twitter_api_bearer_token"


If you’re fetching Reddit data via API (optional), you can also add:

REDDIT_CLIENT_ID = "your_id"
REDDIT_CLIENT_SECRET = "your_secret"

▶️ Run the Dashboard
streamlit run app.py


Then open your browser at http://localhost:8501
.

🧩 Usage

Choose a data source:

Fetch latest discussions from multiple subreddits.

Fetch tweets related to popular stock tickers.

Or upload a saved Reddit JSON file.

Select your model:

FinBERT for deep-learning sentiment analysis.

(VADER fallback optional.)

Analyze:

View aggregate sentiment across tickers.

Compare stock price trends and mood shifts.

Explore visualizations interactively.

Export Results:

Download analyzed data as CSV for further study.

🧠 Tech Stack
Component	Technology
Frontend	Streamlit
Sentiment Analysis	FinBERT (Hugging Face Transformers)
Data Sources	Reddit JSON API, Pushshift, Twitter API
Financial Data	yfinance
Visualization	Plotly Express
Scheduler	pandas_market_calendars
💡 Example Insights
Stock	Avg Sentiment	Mentions	Price Change (Next Day)
AAPL	Positive (0.62)	310	+1.4%
TSLA	Negative (-0.33)	250	-0.8%
NVDA	Neutral (0.05)	120	+0.2%
🧰 Future Improvements

Add live news headlines sentiment feed.

Implement Llama 3 or DeBERTa models for enhanced accuracy.

Store past analysis results in a local SQLite or vector DB.