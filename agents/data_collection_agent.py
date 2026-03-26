import datasets
import kagglehub
import logging
import os
import pandas as pd
import requests
import time
import unicodedata
import yaml

from bs4 import BeautifulSoup
from datetime import datetime, timezone
from urllib.parse import urljoin


class DataCollectionAgent:

    def __init__(self, config: str = None):
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

        self.config = {}
        if config:
            with open(config, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _to_standard(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Приводит DataFrame к унифицированной схеме: text, label, source, collected_at."""
        result = pd.DataFrame()
        result['text'] = df['content'].astype(str)
        if 'label' in df.columns:
            result['label'] = df['label']
        elif 'metadata' in df.columns:
            result['label'] = df['metadata'].apply(
                lambda x: (eval(x) if isinstance(x, str) else x).get(
                    'Название нормативно-правового акта', 'unknown'
                )
            )
        else:
            result['label'] = 'unknown'
        result['source'] = source
        result['collected_at'] = self._now()
        return result

    def fetch_page(self, url, timeout=10):
        """Загрузка страницы с обработкой ошибок."""
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
        """Парсинг HTML контента."""
        return BeautifulSoup(html_content, parser)

    def parse_site(self, url, parser='html.parser', timeout=10):
        response = self.fetch_page(url, timeout)
        if response:
            return self.parse_html(response.content, parser)
        return None

    def extract_links(self, soup, base_url):
        """Извлечение всех ссылок со страницы."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            full_url = urljoin(base_url, href)
            links.append({'url': full_url, 'text': text})
        return links

    def extract_text(self, soup):
        """Извлечение текста из HTML."""
        ocur = soup.find_all('section')[0]
        return unicodedata.normalize('NFKD', ocur.get_text(strip=False)).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace(r' +', ' ').strip()

    def save_to_csv(self, data, filename):
        """Сохранение данных в CSV."""
        if data:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            self.logger.info(f"Сохранено {len(data)} записей в {filename}")
            return True
        return False

    def merge(self, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        """Объединение нескольких DataFrame в один."""
        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
            self.logger.info(f"Объединено {len(dataframes)} DataFrame в один с {len(merged_df)} записями")
            return merged_df
        return pd.DataFrame()

    def scrape(self, url: str, name: str) -> pd.DataFrame:
        """Skill: scrape(url, selector) → DataFrame со стандартными колонками."""
        response = self.fetch_page(url)

        possible_parts = ['часть', 'раздел', 'подраздел', 'глава', 'статья', '§']

        if response:
            soup = self.parse_html(response.content)
            links = self.extract_links(soup, url)
        else:
            return pd.DataFrame(columns=['text', 'label', 'source', 'collected_at'])

        metadata = {
            'Название нормативно-правового акта': name,
            'Часть': '',
            'Раздел': '',
            'Подраздел': '',
            'Глава': '',
            'Параграф': '',
            'Статья': '',
        }
        rows = []

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
                            art_response = self.fetch_page(link['url'])
                        except Exception as e:
                            self.logger.error(f"{link['url']}: {e}")
                            art_response = None
                        if art_response:
                            art_soup = self.parse_html(art_response.content)
                            content = self.extract_text(art_soup)
                            rows.append({
                                'text': content,
                                'label': name,
                                'source': url,
                                'collected_at': self._now(),
                            })

        return pd.DataFrame(rows, columns=['text', 'label', 'source', 'collected_at'])

    def fetch_api(self, endpoint: str, params: dict = None) -> pd.DataFrame:
        """Skill: fetch_api(endpoint, params) → DataFrame со стандартными колонками.

        Ожидает JSON-ответ: список объектов или {'results': [...]} / {'data': [...]}.
        Каждый объект должен содержать поле 'text' (или 'content') и опционально 'label'.
        """
        try:
            time.sleep(self.delay)
            response = self.session.get(endpoint, params=params or {}, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            self.logger.error(f"Ошибка API {endpoint}: {e}")
            return pd.DataFrame(columns=['text', 'label', 'source', 'collected_at'])

        if isinstance(data, dict):
            data = data.get('results', data.get('data', [data]))

        rows = []
        for item in data:
            text = item.get('text', item.get('content', ''))
            label = item.get('label', item.get('category', 'unknown'))
            rows.append({
                'text': str(text),
                'label': str(label),
                'source': endpoint,
                'collected_at': self._now(),
            })

        self.logger.info(f"Получено {len(rows)} записей из API {endpoint}")
        return pd.DataFrame(rows, columns=['text', 'label', 'source', 'collected_at'])

    def load_dataset(self, dataset_name: str, file_path: str = None, source: str = 'kaggle') -> pd.DataFrame:
        """Skill: load_dataset(name, source='hf'|'kaggle') → DataFrame со стандартными колонками."""
        if source == 'hf':
            raw = pd.DataFrame(datasets.load_dataset(dataset_name, split='train'))
            raw['content'] = raw['Текст'].astype(str)
            raw['metadata'] = raw['Название документа'].apply(
                lambda x: {'Название нормативно-правового акта': x}
            )
            return self._to_standard(raw, source=f'hf:{dataset_name}')
        elif source == 'kaggle':
            if file_path is None:
                file_path = os.path.join('data', 'corpus.csv')
            raw = kagglehub.dataset_load(
                kagglehub.KaggleDatasetAdapter.PANDAS,
                dataset_name,
                file_path,
            )
            raw['content'] = raw['Текст'].astype(str)
            raw['metadata'] = raw['Название документа'].apply(
                lambda _: {'Название нормативно-правового акта': 'Разное'}
            )
            return self._to_standard(raw, source=f'kaggle:{dataset_name}')
        else:
            raise ValueError(f"Unsupported source: {source}")

    def run(self, sources: list[dict] = None) -> pd.DataFrame:
        """Запуск агента. sources можно передать явно или взять из config.yaml."""
        if sources is None:
            sources = self.config.get('sources', [])

        dfs = []
        for source in sources:
            stype = source.get('type')
            if stype == 'scrape':
                df = self.scrape(source['url'], source['name'])
            elif stype == 'hf_dataset':
                df = self.load_dataset(dataset_name=source['dataset_name'], source='hf')
            elif stype == 'kaggle_dataset':
                df = self.load_dataset(
                    dataset_name=source['dataset_name'],
                    file_path=source.get('file_path'),
                    source='kaggle',
                )
            elif stype == 'api':
                df = self.fetch_api(source['endpoint'], source.get('params'))
            else:
                self.logger.warning(f"Неизвестный тип источника: {stype}")
                continue
            dfs.append(df)

        return self.merge(dfs)
