import datasets
import kagglehub
import logging
import os
import pandas as pd
import requests
import time
import unicodedata

from bs4 import BeautifulSoup
from urllib.parse import urljoin


class DataCollectionAgent:


    def __init__(self, ):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        self.delay = 1
        self._setup_logging()


    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

   
    def fetch_page(self, url, timeout=10):
        """
        Загрузка страницы с обработкой ошибок
        """
        try:
            time.sleep(self.delay) 
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            if response.encoding is None:
                response.encoding = 'utf-8'

            self.logger.info(f"Успешно загружено: {url} ({len(response.content)} байт)")
            return response

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка загрузки {url}: {e}")
            return None


    def parse_html(self, html_content, parser='html.parser'):
        """
        Парсинг HTML контента
        """
        return BeautifulSoup(html_content, parser)


    def parse_site(self, url, parser='html.parser', timeout=10):
        response = self.fetch_page(url, timeout)
        if response:
            return self.parse_html(response.content, parser)
        return None


    def extract_links(self, soup, base_url):
        """
        Извлечение всех ссылок со страницы
        """
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)

            full_url = urljoin(base_url, href)

            links.append({
                'url': full_url,
                'text': text,
            })

        return links


    def extract_text(self, soup):
        """
        Извлечение текста из HTML
        """
        ocur = soup.find_all('section')[0]
        return unicodedata.normalize('NFKD', ocur.get_text(strip=False)).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace(r' +', ' ').strip()


    def save_to_csv(self, data, filename):
        """
        Сохранение данных в CSV
        """
        if data:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            self.logger.info(f"Сохранено {len(data)} записей в {filename}")
            return True
        return False


    def merge(self, dataframes: list[pd.DataFrame]):
        """
        Объединение нескольких DataFrame в один
        """
        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
            self.logger.info(f"Объединено {len(dataframes)} DataFrame в один с {len(merged_df)} записями")
            return merged_df
        return pd.DataFrame()


    def scrape(self, url: str, name: str) -> pd.DataFrame:
        
        response = self.fetch_page(url)

        possible_parts = ['часть', 'раздел', 'подраздел', 'глава', 'статья', '§']

        if response:
            soup = self.parse_html(response.content)
            links = self.extract_links(soup, url)

        metadata = {
            'Название нормативно-правового акта': name,
            'Часть': '',
            'Раздел': '',
            'Подраздел': '',
            'Глава': '',
            'Параграф': '',
            'Статья': '',
        }
        proper_links = []

        for link in links:
            if len(link['text']) != 0:
                prefix = link['text'].split()[0].lower()
                if prefix in possible_parts:
                    if prefix == 'часть':
                        metadata['Часть'] = link['text']
                    elif prefix == 'раздел':
                        metadata['Раздел'] = link['text']
                    elif prefix == 'подраздел':
                        metadata['Подраздел'] = link['text']
                    elif prefix == 'глава':
                        metadata['Глава'] = link['text']
                    elif prefix == '§':
                        metadata['Параграф'] = link['text']
                    elif prefix == 'статья':
                        metadata['Статья'] = link['text']
                        try:
                            response = self.fetch_page(link['url'])
                        except Exception as e:
                            print(link['url'], e)
                            response = self.fetch_page(link['url']).rstrip(':')
                        if response:
                            soup = self.parse_html(response.content)
                            content = self.extract_text(soup)
                            link['content'] = content
                        link['metadata'] = metadata.copy()
                        proper_links.append(link)         

        return pd.DataFrame(proper_links)[["metadata", "content"]]


    def load_dataset(self, dataset_name: str, file_path: str = None, source: str = 'kaggle') -> pd.DataFrame:
        
        if source == 'hf':
            df = pd.DataFrame(datasets.load_dataset(dataset_name, split="train"))
            df['content'] = df['Текст'].astype(str)
            df['metadata'] = df['Название документа'].apply(lambda x: {'Название нормативно-правового акта': x})
            return df
        elif source == 'kaggle':
            if file_path is None:
                file_path = os.path.join("data", "corpus.csv")  # Update this path as needed
            df = kagglehub.dataset_load(
                kagglehub.KaggleDatasetAdapter.PANDAS,
                dataset_name,
                file_path,
            )
            df['content'] = df['Текст'].astype(str)
            df['metadata'] = df['Название документа'].apply(lambda _: {'Название нормативно-правового акта': 'Разное'})
            return df[['content', 'metadata']]
        else:
            raise ValueError(f"Unsupported source: {source}")


    def run(self, sources: dict[str, str]):

        dfs = []

        for source in sources:

            if source.get("type") == "scrape":
                df = self.scrape(source.get("url"), source.get("name"))
            elif source.get("type") == "hf_dataset":
                df = self.load_dataset(
                    dataset_name=source.get("dataset_name"), 
                    source="hf", 
                )
            elif source.get("type") == "kaggle_dataset":
                df = self.load_dataset(
                    dataset_name=source.get("dataset_name"), 
                    file_path=source.get("file_path"), 
                    source="kaggle", 
                )

            dfs.append(df)

        return self.merge(dfs)
