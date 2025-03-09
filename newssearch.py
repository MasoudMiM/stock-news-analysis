import json
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import os, logging, time
from pathlib import Path
from duckduckgo_search import DDGS 
from duckduckgo_search.exceptions import RatelimitException


def setup_logger():
    log_folder = Path('logs')
    log_folder.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_folder / f'news_search_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# Base class for all data sources
class DataSource(ABC):
    @abstractmethod
    def search(self, keyword, limit, time_filter):
        pass

    @abstractmethod
    def get_content_and_comments(self, article_id, comment_limit):
        pass

    def limit_comments(self, comments, limit):
        return comments[:limit]

# 
class DuckDuckGoNewsSource(DataSource):
    def __init__(self, max_results=20, delay=5):
        self.max_results = max_results
        self.delay = delay  
        self.last_search_results = []

    def search(self, keyword, limit, time_filter):
        max_retries = 3
        keyword = f'"{keyword}"'
        for attempt in range(max_retries):
            try:
                results = self._sync_search(keyword, limit, time_filter)
                self.last_search_results = results
                time.sleep(self.delay)  
                return results
            except RatelimitException:
                logger.warning(f"Rate limit hit for DuckDuckGo. Attempt {attempt + 1} of {max_retries}. Waiting for {self.delay * 2} seconds.")
                time.sleep(self.delay * 2)  
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                break  
        logger.error(f"Failed to search DuckDuckGo after {max_retries} attempts.")
        return []

    def _sync_search(self, keyword, limit, time_filter):
        with DDGS() as ddgs:
            results = ddgs.news(
                keywords=keyword,
                region='wt-wt',
                safesearch='off',
                timelimit=self._get_time_filter(time_filter),
                max_results=min(limit, self.max_results)
            )

        time.sleep(2)  
        
        search_results = []
        for result in results:
            search_results.append({
                'title': result['title'],
                'href': result['url'],
                'published': result['date'],
                'body': result['body']
            })
        
        logger.info(f"Found {len(search_results)} DuckDuckGo news results for keyword: {keyword}")
        return search_results

    def get_content_and_comments(self, url, comment_limit):
        content = next((item for item in self.last_search_results if item['href'] == url), None)
        if content:
            content_dict = {
                'id': url,
                'timestamp': content['published'],
                'title': content['title'],
                'url': content['href'],
                'body': content['body']
            }
        else:
            content_dict = {
                'id': url,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'title': 'DuckDuckGo News Result',
                'url': url,
                'body': 'Content not found. Visit the URL to read the full article.'
            }
        return content_dict, []

    def _get_time_filter(self, time_filter):
        if time_filter == 'day':
            return 'd'
        elif time_filter == 'week':
            return 'w'
        elif time_filter == 'month':
            return 'm'
        else:
            return 'y' 
        
    
# main function to handle all sources
def gather_data(sources, keywords, search_limit, comment_limit, time_filter, outputs_folder):
    final_data = {}
    all_results = []

    for source in sources:
        source_name = source.__class__.__name__  # Get the class name as the source identifier
        for keyword in keywords:
            results = source.search(keyword, search_limit, time_filter)
            
            # Add source to search results
            for result in results:
                result['source'] = source_name
            all_results.extend(results)

            for result in results:
                try:
                    content, comments = source.get_content_and_comments(result['href'], comment_limit)
                    if content or comments:
                        content_timestamp = content['timestamp'] if content else "Unknown"
                        content_title = content['title'] if content else result['title']
                        
                        comments_data = {comment['timestamp']: comment['body'] for comment in comments} if comments else {}

                        final_data[f"{content_timestamp} : {content_title}"] = {
                            'url': result['href'],
                            'source': source_name,
                            'content': content,
                            'comments': comments_data
                        }
                        
                        logger.info(f"Retrieved {'content' if content else ''} from {source_name}")
                    else:
                        logger.warning(f"No content found for URL: {result['href']} from {source_name}")
                except KeyError as e:
                    logger.error(f"KeyError processing URL {result['href']} from {source_name}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error processing URL {result['href']} from {source_name}: {str(e)}")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_search_file_name =f'{outputs_folder}/search_output_{current_time}.json'
    output_contents_file_name = f'{outputs_folder}/content_output_{current_time}.json'

    with open(output_search_file_name, 'w') as f:
        json.dump(all_results, f, indent=4)

    with open(output_contents_file_name, 'w') as f:
        json.dump(final_data, f, indent=4)

    logger.info(f"Processed {len(all_results)} search results")
    logger.info(f"Extracted content for {len(final_data)} items")
    
    return final_data
