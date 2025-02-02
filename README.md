# Stock Sentiment & Price Correlation Analyzer

A Streamlit-based analytical tool that extracts and analyzes sentiment from social media comments (HTML, CSV, or JSON formats), identifies stock tickers using a custom ticker CSV (with built-in filtering to ignore common words), retrieves trading data via yfinance, and visualizes the correlation between sentiment and price movement. The app also features a progress bar with ETA during processing and provides options to export the analysis as CSV and PDF.

## Features

- **Multi-format Input:**  
  Accepts comments in HTML, CSV, and JSON formats.
  
- **Ticker Extraction:**  
  Uses a customizable ticker alias dictionary (loaded via CSV or default sample) to extract stock tickers from comments. Built-in filtering prevents matching common words (e.g. "PM", "NOW", "IT", etc.).
  
- **Sentiment Analysis:**  
  Uses the VADER sentiment lexicon (from NLTK) to calculate the compound sentiment score for each comment. Classifies sentiment as:
  - Bullish: score > 0.05
  - Bearish: score < -0.05
  - Neutral: otherwise

- **Trading Data Integration:**  
  Retrieves next day and weekly trading data for each identified ticker from yfinance. Uses caching to improve performance.

- **Visualizations:**  
  - **Scatter Plot:** Average sentiment vs. next day close (with quadrant overlays and annotations).  
  - **Mentions Bar Chart:** Bar chart of top tickers by mention count, colored by sentiment.  
  - **Sentiment Distribution Histogram:** Histogram showing distribution of individual sentiment scores.  
  - **Price Change vs. Sentiment Bubble Chart:** Bubble chart where the x-axis is average sentiment, the y-axis is percentage price change (from open to close), bubble size indicates mention count, and bubbles are colored by sentiment.

- **Progress Feedback:**  
  Displays a progress bar with an estimated time of arrival (ETA) during processing of the top stocks.

- **Export Options:**  
  Download the final analysis table as a CSV or PDF file.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/stock-sentiment-analyzer.git
   cd stock-sentiment-analyzer
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the Required Packages:**

   You can install the necessary libraries via pip:

   ```bash
   pip install streamlit yfinance beautifulsoup4 nltk plotly fpdf pandas
   ```

   Alternatively, if you provide a `requirements.txt` file, run:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application:**

   ```bash
   python -m streamlit run app.py
   ```

2. **Upload Files:**
   - Use the sidebar to optionally upload your ticker CSV file (with columns such as "Rank", "Name", "Symbol", etc.) to build a comprehensive ticker alias dictionary.
   - Then upload your comments file in HTML, CSV, or JSON format.
   - The app extracts an analysis date and determines if the discussion was during a weekend or on a weekday to adjust trading dates.

3. **View Results:**
   - The Market Sentiment Overview table is displayed at the top.
   - The Top 10 Stocks Analysis table is shown along with several graphs:
     - Avg Sentiment vs. Next Day Close scatter plot.
     - Mentions Bar Chart.
     - Sentiment Distribution Histogram.
     - Price Change vs. Sentiment Bubble Chart.
   - A progress bar with ETA is shown during processing.

4. **Export Data:**
   - Use the provided download buttons to export the analysis table as CSV or PDF.

## Customization

- **Thresholds:**  
  The app uses VADER’s compound score thresholds (0.05 and -0.05) to classify sentiment as bullish, bearish, or neutral. You can modify these thresholds in the code if needed.

- **Blacklist:**  
  Update the `common_ticker_blacklist` set in the code to add or remove any ticker symbols that should be ignored during extraction.

- **Visualizations:**  
  The charts are created with Plotly Express. Feel free to customize colors, sizes, or add additional charts as needed.

## Contributing

Contributions, suggestions, and bug reports are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

This repository is for an AP Research class.
