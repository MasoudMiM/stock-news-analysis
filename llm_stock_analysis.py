import json, markdown
import os, requests, re, logging, traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import glob, base64
import pandas as pd
from pymongo import MongoClient

import chromadb
from sentence_transformers import SentenceTransformer

from newssearch import gather_data, DuckDuckGoNewsSource
from openbb import obb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import concurrent.futures
import time


# ----- USER INPUTS
INPUT_STOCK_DATA = "stocks.txt"
TIME_PERIOD = 'month'  # Options: 'hour', 'day', 'week', 'month', 'year', 'all'
SEARCH_LIMIT = 10 # number of search results per item (stock)
INPUT_GMAIL = "masoumi.masoud@gmail.com"
# ChromaDB setup
chroma_client = chromadb.PersistentClient(path='databases/chroma_db')
# Creating collection in ChromaDB
collection = chroma_client.get_or_create_collection("stock_recommendations")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Make sure ollama is running and listening to this port
ollama_url = "http://127.0.0.1:11434"
model_names = ["llama3.3:latest"] # you can choose any LLM as long as it is running via Ollama
outputs_folder = 'outputs'
os.makedirs(outputs_folder, exist_ok=True)
# MongoDB setup
client = MongoClient('mongodb://127.0.0.1:27017/') 
db = client['stock_database']
stock_collection = db['stock_data']
web_scraping_collection = db['web_scraping_data']
recommendations_collection = db['recommendations']
financial_reports_collection = db['financial_reports']
# --------

# function to generate conversational format based on the stop price data
def conversational_format(df):
    formatted_data = ""
    grouped = df.groupby('Ticker')
    for ticker, group in grouped:
        formatted_data += f"Let's talk about {group['Name'].iloc[0]} (Ticker: {ticker}).\n"
        formatted_data += "Here's how the stock prices moved:\n"
        prev_price = None
        for _, row in group.iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            price = row['Close_Price']
            if prev_price is not None:
                if price > prev_price:
                    formatted_data += f"On {date}, the stock price increased to ${price:.2f}.\n"
                elif price < prev_price:
                    formatted_data += f"On {date}, the stock price dropped to ${price:.2f}.\n"
                else:
                    formatted_data += f"On {date}, the stock price stayed the same at ${price:.2f}.\n"
            else:
                formatted_data += f"On {date}, the stock price was ${price:.2f}.\n"
            prev_price = price
        formatted_data += "\n"
    return formatted_data

# function to retrieve the most similar previous recommendations using ChromaDB
def get_most_similar_memory(current_prompt, top_n=2, min_score=0.7):
    current_embedding = embedding_model.encode(current_prompt)
    
    logger.info(f"Performing similarity search in ChromaDB based on embeddings...")

    results = collection.query(
        query_embeddings=[current_embedding],
        n_results=top_n,
        where={"performance_score": {"$gte": min_score}}
    )

    if results['documents'] and results['documents'][0]:
        similar_recommendations = "\n".join([doc[0] for doc in results['documents']])
    else:
        similar_recommendations = ""

    return similar_recommendations


# my function to save recommendations to ChromaDB
def save_recommendation(text, model_name, performance_score=None):
    embedding = embedding_model.encode(text)
    
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metadata = {"model_name": model_name, "date": time_stamp}
    if performance_score is not None:
        metadata["performance_score"] = performance_score
    
    collection.add(
        documents=[text],
        metadatas=[metadata],
        embeddings=[embedding.tolist()],  # Convert numpy array to list
        ids=[f"recommendation_{time_stamp}_{model_name}"]
    )
    logger.info(f"Stored the recommendation and its metadata in the collection.")
    logger.info(f"Current collection size: {collection.count()}") # DEBUG

# setting up the llm logger
def setup_logger():
    log_folder = 'logs'
    os.makedirs(log_folder, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f'llm_{current_time}.log')
    
    logger = logging.Logger('llm_logger')
    logger.setLevel(logging.INFO)
    
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    return service


def send_email(service, report, recipient_email):
    try:
        logger.info(f"Attempting to send email to {recipient_email}")
        message = MIMEMultipart()
        message['to'] = recipient_email
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message['subject'] = f"Stock Recommendations - {current_time}"

        # adding report content
        for title, content in report:
            html_content = markdown.markdown(content)
            message.attach(MIMEText(f"<h2>{title}</h2>{html_content}", 'html'))
            logger.info(f"Added HTML content: {title}")

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        logger.info("Message encoded")
        
        message = service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        logger.info(f"Message Id: {message['id']}")
        logger.info("Email sent successfully")
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(' +', ' ', text) 
    return text

def read_stock_data(folder="data"):
    

    datat = pd.read_csv(f'{folder}/{INPUT_STOCK_DATA}')

    datat['Date'] = pd.to_datetime(datat['Date'])
    conversational_stocks = conversational_format(datat)

    return conversational_stocks

from datetime import datetime, date

def convert_dates(obj):
    if isinstance(obj, date):
        return datetime.combine(obj, datetime.min.time())
    elif isinstance(obj, dict):
        return {key: convert_dates(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_dates(item) for item in obj]
    return obj

def save_stock_data_to_mongodb(top_n_data, timestamp):
    converted_top_n_data = convert_dates(top_n_data)
    
    stock_collection.insert_many([
        {'type': 'n_performers', 'data': converted_top_n_data, 'timestamp': timestamp}
    ])
    logger.info(f"Stock data saved to MongoDB at {timestamp}")

def save_web_scraping_data_to_mongodb(web_scraping_data, timestamp):
    web_scraping_collection.insert_one({'data': web_scraping_data, 'timestamp': timestamp})
    logger.info(f"Web scraping data saved to MongoDB at {timestamp}")

def save_recommendations_to_mongodb(recommendations, timestamp):
    recommendations_collection.insert_one({
        'recommendations': recommendations,
        'timestamp': timestamp
    })
    logger.info(f"Recommendations saved to MongoDB at {timestamp}")

def read_latest_stock_data_from_mongodb(timestamp):
    top_n_data = stock_collection.find_one({'type': 'n_performers', 'timestamp': timestamp})['data']
    return top_n_data

def read_latest_web_scraping_data_from_mongodb(timestamp):
    return web_scraping_collection.find_one({'timestamp': timestamp})['data']

def fetch_financial_reports(tickers):
    reports = {}
    for ticker in tickers:
        try:
            income_statement = obb.equity.fundamental.income(ticker, limit=4, period="quarter")
            balance_sheet = obb.equity.fundamental.balance(ticker, limit=4, period="quarter")
            cash_flow = obb.equity.fundamental.cash(ticker, limit=4, period="quarter")
            report= {
                'income_statement': income_statement.to_dict() if income_statement is not None else None,
                'balance_sheet': balance_sheet.to_dict() if balance_sheet is not None else None,
                'cash_flow': cash_flow.to_dict() if cash_flow is not None else None
            }
            
            reports[ticker] = convert_dates(report)
            logger.info(f"Successfully fetched financial reports for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching financial reports for {ticker}: {str(e)}")
    return reports


def prepare_financial_summary(financial_reports):
    financial_summary = "Here are summaries of the latest financial reports for the stocks:\n\n"
    for ticker, report in financial_reports.items():
        financial_summary += f"Financial summary for {ticker}:\n"
        
        if report['income_statement']:
            income_stmt = report['income_statement']
            financial_summary += f"Total Revenue: {income_stmt.get('total_revenue', ['Unknown'])[-1]}\n"
            financial_summary += f"Operating Income: {income_stmt.get('operating_income', ['Unknown'])[-1]}\n"
            financial_summary += f"Net Income: {income_stmt.get('net_income', ['Unknown'])[-1]}\n"
            financial_summary += f"EPS (Diluted): {income_stmt.get('diluted_earnings_per_share', ['Unknown'])[-1]}\n"
        else:
            financial_summary += "Income statement data not available\n"
        
        if report['balance_sheet']:
            balance_sheet = report['balance_sheet']
            financial_summary += f"Total Assets: {balance_sheet.get('total_assets', ['Unknown'])[-1]}\n"
            financial_summary += f"Total Liabilities: {balance_sheet.get('total_liabilities_net_minority_interest', ['Unknown'])[-1]}\n"
            financial_summary += f"Total Equity: {balance_sheet.get('total_equity_non_controlling_interests', ['Unknown'])[-1]}\n"
        else:
            financial_summary += "Balance sheet data not available\n"
        
        if report['cash_flow']:
            cash_flow = report['cash_flow']
            financial_summary += f"Investing Cash Flow: {cash_flow.get('investing_cash_flow', ['Unknown'])[-1]}\n"
            financial_summary += f"Operating Cash Flow: {cash_flow.get('operating_cash_flow', ['Unknown'])[-1]}\n"
            financial_summary += f"Financing Cash Flow: {cash_flow.get('financing_cash_flow', ['Unknown'])[-1]}\n"
            financial_summary += f"Free Cash Flow: {cash_flow.get('free_cash_flow', ['Unknown'])[-1]}\n"
        else:
            financial_summary += "Cash flow data not available\n"
        
        financial_summary += "\n"
    
    return financial_summary


def chunk_text(text, max_chunk_size=15000):
    chunks = []
    sentences = text.split(". ")
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
    
    return chunks


def run_ollama_model(model_name, prompt, companies_to_verify=None):
    chunks = chunk_text(prompt)
    full_response = ""
    context_window = []
    
    # first, let's check which companies are mentioned in each chunk
    chunk_companies = []
    for chunk in chunks:
        companies_in_chunk = [company for company in companies_to_verify 
                            if company.lower() in chunk.lower()] if companies_to_verify else []
        chunk_companies.append(companies_in_chunk)
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            current_prompt = f"""Analyze this information carefully and completely. Make sure to consider all companies and data points mentioned:
            
            {chunk}"""
        else:
            # include a summary of previous chunks
            context = "\n".join(context_window[-2:])
            current_prompt = f"""Previous context and analysis:
            {context}
            
            Previous analysis summary:
            {full_response}
            
            Continue analyzing with this additional information:
            {chunk}
            
            Companies mentioned in this section:
            {', '.join(chunk_companies[i])}"""
        
        try:
            payload = {
                "model": model_name,
                "prompt": current_prompt,
                "stream": False
            }
            
            response = requests.post(f"{ollama_url}/api/generate", json=payload)
            
            if response.status_code == 200:
                responses = []
                for line in response.text.strip().splitlines():
                    try:
                        json_response = json.loads(line)
                        responses.append(json_response)
                    except json.JSONDecodeError:
                        continue
                
                chunk_response = "".join(response["response"] for response in responses)
                
                # only verify companies that should be in this chunk
                if companies_to_verify and chunk_companies[i]:
                    if not verify_analysis_completeness(chunk_response, chunk_companies[i]):
                        logger.warning(f"Chunk {i+1} missing analysis for some mentioned companies. Requesting clarification...")
                        clarification_prompt = f"""Your response is missing analysis for some companies that were mentioned in this section.
                        Previous response: {chunk_response}
                        Provide analysis for these companies that were mentioned: {', '.join(chunk_companies[i])}
                        
                        Original data:
                        {chunk}"""
                        
                        try:
                            clarification_payload = {
                                "model": model_name,
                                "prompt": clarification_prompt,
                                "stream": False
                            }
                            clarification_response = requests.post(f"{ollama_url}/api/generate", json=clarification_payload)
                            if clarification_response.status_code == 200:
                                clarification_text = "".join(json.loads(line)["response"] 
                                                           for line in clarification_response.text.strip().splitlines())
                                chunk_response += "\n" + clarification_text
                        except Exception as e:
                            logger.error(f"Error getting clarification: {str(e)}")
                
                full_response += chunk_response
                context_window.append(chunk)
                logger.info(f"Processed chunk {i+1}/{len(chunks)} - Size: {len(chunk)} characters")
                
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {str(e)}")
    
    # final verification only after all chunks are processed
    if companies_to_verify:
        if not verify_analysis_completeness(full_response, companies_to_verify):
            logger.warning("Final analysis is incomplete. Requesting final review...")
            missing_companies = [company for company in companies_to_verify 
                               if company.lower() not in full_response.lower()]
            
            final_review_prompt = f"""Your analysis is missing information for these companies:
            {', '.join(missing_companies)}
            
            Provide a complete analysis for EACH of these missing companies using all available data.
            
            Previous analysis:
            {full_response}"""
            
            try:
                final_review_payload = {
                    "model": model_name,
                    "prompt": final_review_prompt,
                    "stream": False
                }
                final_review_response = requests.post(f"{ollama_url}/api/generate", json=final_review_payload)
                if final_review_response.status_code == 200:
                    final_review_text = "".join(json.loads(line)["response"] 
                                              for line in final_review_response.text.strip().splitlines())
                    full_response += "\n\nFinal Review:\n" + final_review_text
            except Exception as e:
                logger.error(f"Error getting final review: {str(e)}")
    
    return model_name, full_response


def run_model_analysis(model_name, n_data, web_scraping_data, financial_summary, analysis_timestamp):
    logger.info(f"Started the conversation with model: {model_name}")
    conversation_history = []
    
    # getting a list of companies for verification
    companies = list(set(re.findall(r"Let's talk about (.*?) \(Ticker:", n_data)))
    
    # 1- analyzing stocks
    logger.info(f"--- Analyzing the stocks using model: {model_name}")
    companies_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(set(companies))])
    prompt_top = f"""IMPORTANT: Analyze ALL of the following stocks for the last 30 days.
    You must address EACH company listed below in your analysis.
    
    Companies to analyze:
    {companies_list}
    
    Detailed stock data:
    {n_data}
    
    Requirements:
    1. Provide specific analysis for EACH company listed
    2. Include price movements and trends
    3. Do not skip any company
    4. Make sure to include any data in your analysis
    
    Format your response with clear sections for each company."""
    
    logger.info(f"Initial Input Prompt: {prompt_top}.")
                
    response_top = run_ollama_model(model_name, prompt_top, companies_to_verify=companies)
    
    # verifying complete analysis
    if not verify_analysis_completeness(response_top[1], companies):
        logger.warning("Final stocks analysis is incomplete. Requesting full analysis...")
        missing_companies = [company for company in companies 
                           if company.lower() not in response_top[1].lower()]
        
        my_prompt = f"""Your analysis is missing information for these companies:
        {', '.join(missing_companies)}
        
        Please provide a complete analysis for EACH of these missing companies using the data:
        {n_data}"""
        
        additional_response = run_ollama_model(model_name, my_prompt, companies_to_verify=missing_companies)
        response_top = (model_name, response_top[1] + "\n\nAdditional Analysis:\n" + additional_response[1])
    
    conversation_history.append(("Human", prompt_top))
    conversation_history.append(("AI", response_top[1]))

    # 2 - analyzing web scraping data
    logger.info(f"--- Analyzing the web scraping results using model: {model_name}")
    prompt_web = f"Analyze this web scraping data about the stocks we discussed in our Previous conversation: {web_scraping_data}..."
    logger.info(f"----- prompt_web: {prompt_web}")  
    response_web = run_ollama_model(model_name, prompt_web + "\n\nPrevious conversation:\n" + "\n".join([f"{role}: {content}" for role, content in conversation_history]))
    conversation_history.append(("Human", prompt_web))
    conversation_history.append(("AI", response_web[1]))

    # 3 - analyzing financial reports
    logger.info(f"--- Analyzing the financial reports using model: {model_name}")
    prompt_financial = f"Analyze these financial reports for the stocks: {financial_summary}..." 
    logger.info(f"----- prompt_financial: {prompt_financial}") 
    response_financial = run_ollama_model(model_name, prompt_financial + "\n\nPrevious conversation:\n" + "\n".join([f"{role}: {content}" for role, content in conversation_history]))
    conversation_history.append(("Human", prompt_financial))
    conversation_history.append(("AI", response_financial[1]))

    # 4 - retrieving similar past recommendations
    logger.info(f"--- Retrieving similar past recommendations using model: {model_name}")
    current_context = "\n".join([f"{role}: {content}" for role, content in conversation_history])
    similar_recommendations = get_most_similar_memory(current_context)
    prompt_similar = f"Consider these similar past recommendations (if there is any): {similar_recommendations}\n\n. If there are any, how do they relate to our current analysis? If there is none, ignore this request."
    logger.info(f"----- prompt_similar: {prompt_similar}")
    response_similar = run_ollama_model(model_name, prompt_similar + "\n\nPrevious conversation:\n" + current_context)
    conversation_history.append(("Human", prompt_similar))
    conversation_history.append(("AI", response_similar[1]))

    # 5 - final recommendation (updating the existing step 5)
    logger.info(f"--- Coming up with the final recommendation using model: {model_name}")
    prompt_final = "Based on our previous conversations, including the analysis of similar past recommendations, provide a comprehensive swing trader approach advice for investing $100 in the stock market today. How should the $100 be split between the stocks for possible maximum profit?"
    logger.info(f"----- prompt_final: {prompt_final}") 
    response_final = run_ollama_model(model_name, prompt_final + "\n\nPrevious conversation:\n" + "\n".join([f"{role}: {content}" for role, content in conversation_history]))

    combined_response = "\n\n".join([
        f"Top Stocks Analysis:\n{response_top[1]}",
        f"Web Scraping Analysis:\n{response_web[1]}",
        f"Financial Reports Analysis:\n{response_financial[1]}",
        f"Similar Past Recommendations Analysis:\n{response_similar[1]}",
        f"Final Recommendation:\n{response_final[1]}"
    ])

    logger.info(f"----- combined_response: {combined_response}")
    # saving the new recommendation to ChromaDB
    logger.info(f"--- Saving the recommendation in ChromaDB for model: {model_name}")
    save_recommendation(combined_response, model_name)

    return {
        "model": model_name,
        "conversation_history": conversation_history,
        "initial_recommendation": response_top[1],
        "final_recommendation": response_final[1],
        "analysis_timestamp": analysis_timestamp
    }


def generate_recommendations(n_data, analysis_timestamp):
    if n_data:
        # reading latest stock data from MongoDB
        n_data = read_latest_stock_data_from_mongodb(analysis_timestamp)
        
        # getting a list of all stock tickers
        data = pd.read_csv(f'data/{INPUT_STOCK_DATA}')
        
        all_tickers = data.loc[:,"Ticker"].drop_duplicates()
        all_names = data.loc[:, "Name"].drop_duplicates()
    
        # getting the names of the stocks
        potential_stocks_names = [str(name) for name in all_names if (str(name) in n_data)]
        logger.info(f" The stocks the web will be searched for: {potential_stocks_names}")
        
        # TODO: 
        # - Add duckducgo searching specific sources with options such as "site:finance.yahoo.com" 
        # - Add OpenBB news results
        sources = [DuckDuckGoNewsSource()]
        keywords = potential_stocks_names
        search_limit = SEARCH_LIMIT
        time_filter = TIME_PERIOD


        logger.info("Starting data gathering process")
        internet_data = gather_data(sources, keywords, search_limit, 0, time_filter, outputs_folder)
        logger.info("Data gathering process completed")

        all_texts = []

        for post_key, post_value in internet_data.items():
            body = post_value['content'].get('body', '')
            all_texts.append(body.replace('\n', ', '))
            

        final_text = ', '.join(all_texts)
        final_text = clean_text(final_text)

        save_web_scraping_data_to_mongodb(final_text, analysis_timestamp)

        all_tickers_analysis = [str(tik) for tik in all_tickers if str(tik) in n_data] 
        
        web_scraping_data = read_latest_web_scraping_data_from_mongodb(analysis_timestamp)
        
        financial_reports = fetch_financial_reports(all_tickers_analysis)
        financial_summary = prepare_financial_summary(financial_reports)
        
        results = []
        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for model_name in model_names:
                future = executor.submit(
                    run_model_analysis,
                    model_name,
                    n_data,
                    web_scraping_data,
                    financial_summary,
                    analysis_timestamp
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Model {result['model']} completed analysis")
                except Exception as e:
                    logger.error(f"An error occurred during model analysis: {str(e)}")

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"All models completed in {total_time:.2f} seconds")

        return results
    else:
        return []


def generate_and_send_report(recipient_email):
    # Generate a single timestamp for this analysis cycle
    analysis_timestamp = datetime.now()

    # reading stock data
    data_txt = read_stock_data()
    
    # saving stock data to MongoDB
    logger.info(f"Saving the web scraping data to MongoDB database")
    save_stock_data_to_mongodb(data_txt, analysis_timestamp)
    
    # generating the recommendations
    recommendations = generate_recommendations(data_txt, analysis_timestamp)
    
    # saving recommendations to MongoDB
    logger.info(f"Saving the recommendation to MongoDB database")
    save_recommendations_to_mongodb(recommendations, analysis_timestamp)

    # a formatted version of recommendations for email
    formatted_recommendations = [
        (f"Comprehensive Analysis and Recommendation from {rec['model']}", 
         f"Initial Recommendation:\n{rec['initial_recommendation']}\n\n"
         f"Final Recommendation:\n{rec['final_recommendation']}\n\n"
         f"Analysis performed at: {rec['analysis_timestamp']}")
        for rec in recommendations
    ]

    # Gmail service and send the email
    service = get_gmail_service()
    send_email(service, formatted_recommendations, recipient_email)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    recomm_file = f'{outputs_folder}/recommendations_{current_time}.json' 
    logger.info(f"The recommendation file was saved at {recomm_file}")


def verify_analysis_completeness(response, company_list):
    missing_companies = []
    for company in company_list:
        if company.lower() not in response.lower():
            missing_companies.append(company)
    
    if missing_companies:
        logger.warning(f"Analysis missing for companies: {missing_companies}")
        return False
    return True

if __name__ == "__main__":
    recipient_email = f"{INPUT_GMAIL}"  
    generate_and_send_report(recipient_email)