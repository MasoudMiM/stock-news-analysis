# Stock Analysis and Recommendation System

## Overview

The Stock Analysis and Recommendation System is an application designed to pull financial data, pull stock performance data, scrape the web for related news, and then generate actionable recommendations for investors using Large Language Model. This project leverages various APIs, databases, and machine learning techniques to provide insights into stock market trends and individual stock performance.

## Features

- **Financial Data Retrieval**: Automatically fetches financial data for stocks from various sources, including income statements, balance sheets, and cash flow statements.
- **News Aggregation**: Gathers the latest news articles' first few lines related to specified stocks using DuckDuckGo.
- **Performance Analysis**: Analyzes stock performance over specified time periods.
- **Conversational Reporting**: Formats analysis results into a conversational style for easy understanding and presentation.
- **Model Integration**: Utilizes the Ollama model for advanced analysis and recommendations based on historical data and current market conditions.
- **MongoDB Storage**: Stores stock data, web scraping results, and recommendations in a MongoDB database for easy retrieval and management.
- **Email Notifications**: Sends detailed reports and recommendations via email using the Gmail API.
- **Logging and Error Handling**: Implements robust logging to track application performance and errors, ensuring reliability and maintainability.

## Tools/Libraries Used

- **Programming Language and Data handling**: Python \& Pandas data manipulation and analysis
- **Database**: MongoDB for storing stock data, web scraping results, and recommendations.
- **APIs**:
  - **Google Gmail API**: For sending email notifications.
  - **OpenBB API**: For fetching financial reports and stock data.
  - **DuckDuckGo**: For news aggregation related to stocks.
- **Machine Learning**: Sentence Transformers for embedding and similarity search using ChromaDB.
- **Logging**: Python's built-in logging library for tracking application behavior.

## Database Structure

### MongoDB/ChromaDB Collections

1. **stock_data**: Stores stock performance data.
   - Fields: `type`, `data`, `timestamp`

2. **web_scraping_data**: Contains the results of web scraping activities, including news articles and comments.
   - Fields: `data`, `timestamp`

3. **recommendations**: Stores generated recommendations based on analysis.
   - Fields: `recommendations`, `timestamp`

4. **financial_reports**: Contains financial reports fetched from the OpenBB API.
   - Fields: `ticker`, `income_statement`, `balance_sheet`, `cash_flow`, `timestamp`

5. **stock_recommendations** (ChromaDB): Stores embeddings and metadata for past recommendations to facilitate similarity searches.
   - Fields: `documents`, `metadatas`, `embeddings`, `ids`

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- MongoDB installed and running
- Access to the Google Cloud Console for Gmail API credentials
- OpenBB API access

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MasoudMiM/stock-news-analysis.git
   cd stock-news-analysis
   ```

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Google Gmail API**:
   - Create a project in the Google Cloud Console.
   - Enable the Gmail API and create credentials.
   - Download the `client_secret.json` file and place it in the project directory.

4. **Set Up MongoDB**:
   - Ensure MongoDB is running locally or configure the connection string for a remote MongoDB instance.

5. **Run the Application**:
   ```bash
   python llm_stock_analysis.py
   ```

## Usage

- Modify the user input parameters in the code to specify stock tickers, time periods, and email settings.
- The application will automatically fetch data, analyze stocks, and send a report to the specified email address.

## Contributing

Contributions are welcome. If you have suggestions for improvements or new features, please open an issue or submit a pull request.


## Contact

For any inquiries or feedback, please reach out to [masoumi.masoud@gmail.com](mailto:masoumi.masoud@gmail.com).

## TODO

&#9633; Get related web search data from social mmedia platforms (Reddit, Lemmy, Gaurdian, Twitter, etc) and news websites and platforms (Yahoo finance, Open BB, etc) and not just DuckDuckGo search engine.

&#9633; Add web scraping so it reads the whole article and not just the first few lines.

&#9633; Add the company names to extracted text to make sure there is always an association between the text and the company.

&#9633; Add extra filtering to include only really relevant news in the analysis.

&#9633; Optimize the prompts.


