# MarketMoodRadar 🎯

**MarketMoodRadar** is an interactive Streamlit tool that analyzes social media sentiment and correlates it with stock price data. Developed for an AP Research class, this project extracts comments from HTML, CSV, or JSON files, identifies stock tickers (with custom filtering to ignore common words), and retrieves trading data via yfinance. Visualizations include scatter plots with quadrant overlays, bar charts, histograms, and bubble charts—all with real-time progress feedback and CSV export.

👉 **Live Demo:** [tommy-marketmood.streamlit.app/](https://tommy-marketmood.streamlit.app/)

## Features 🚀

- **Multi-format Input:**  
  Accepts comments in **HTML**, **CSV**, and **JSON** formats.

- **Ticker Extraction:**  
  Uses a customizable ticker alias dictionary (loaded via CSV or default sample) to extract stock tickers from comments. Built-in filtering ignores common words like "PM", "NOW", "IT", etc.

- **Sentiment Analysis:**  
  Utilizes the VADER lexicon from NLTK to compute sentiment scores, classifying comments as:
  - **Bullish:** score > 0.05  
  - **Bearish:** score < -0.05  
  - **Neutral:** otherwise

- **Trading Data Integration:**  
  Retrieves next day and weekly trading data from yfinance with caching for improved performance.

- **Visualizations:**  
  - **Quadrant Scatter Plot:** Avg Sentiment vs. Next Day Close with quadrant overlays and annotations.  
  - **Mentions Bar Chart:** Top tickers by mention count, color-coded by sentiment.  
  - **Sentiment Distribution Histogram:** Distribution of individual sentiment scores.  
  - **Price Change vs. Sentiment Bubble Chart:** Bubble chart showing the percentage price change vs. average sentiment.

- **Progress Feedback:**  
  A progress bar with estimated time remaining (ETA) is displayed during processing.

- **Export Options:**  
  Export the analysis table as **CSV**.

## Installation 🛠️

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/marketmoodradar.git
   cd marketmoodradar
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**

   ```bash
   pip install streamlit yfinance beautifulsoup4 nltk plotly pandas
   ```

   Or, if you have a `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

## Usage 💡

1. **Run the Application:**

   ```bash
   python -m streamlit run app.py
   ```

2. **Upload Files:**
   - **Ticker CSV (Optional):**  
     Upload your ticker CSV (with columns like "Rank", "Name", "Symbol", etc.) via the sidebar.
   - **Comments File:**  
     Upload your comments file (HTML, CSV, or JSON). The app automatically extracts the analysis date and adjusts trading dates (weekday vs. weekend).

3. **View Results:**
   - Review the **Market Sentiment Overview** and **Top 10 Stocks Analysis** tables.
   - Explore interactive visualizations:
     - Quadrant Scatter Plot
     - Mentions Bar Chart
     - Sentiment Histogram
     - Price Change vs. Sentiment Bubble Chart
   - Watch the progress bar with ETA during processing.

4. **Export Data:**
   - Download the analysis table as **CSV** using the export button.

## Customization ⚙️

- **Thresholds:**  
  Modify the sentiment thresholds (currently 0.05 and -0.05) directly in the code if needed.
- **Blacklist:**  
  Update the `common_ticker_blacklist` to add or remove ticker symbols that are common words.
- **Visualizations:**  
  Customize charts using Plotly Express parameters.

## Contributing 🤝

Contributions, suggestions, and bug reports are welcome!  
Please open an issue or submit a pull request or email me at tommyacollege@gmail.com

## License 📄

This project is licensed under the [MIT License](LICENSE).
