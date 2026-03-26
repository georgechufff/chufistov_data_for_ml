# annotation_agent_gpt.py - Обновленная версия с OpenAI GPT

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import warnings
import time
from datetime import datetime
import os
import glob

# OpenAI для LLM
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings('ignore')

@dataclass
class QualityMetrics:
    """Метрики качества разметки"""
    kappa: Optional[float]
    label_distribution: Dict[str, int]
    confidence_mean: float
    confidence_std: float
    low_confidence_count: int
    low_confidence_threshold: float
    total_samples: int
    
    def to_dict(self):
        return asdict(self)

@dataclass
class AnnotationSpec:
    """Спецификация разметки"""
    task: str
    description: str
    classes: Dict[str, str]
    examples: List[Dict]
    edge_cases: List[str]
    guidelines: List[str]
    
    def to_markdown(self) -> str:
        """Конвертирует спецификацию в Markdown"""
        md = f"""# Спецификация аннотации данных

## Задача: {self.task}

### Описание
{self.description}

### Классы разметки
"""
        for class_name, description in self.classes.items():
            md += f"\n#### {class_name}\n{description}\n"
        
        md += "\n### Примеры\n"
        for i, example in enumerate(self.examples, 1):
            md += f"\n#### Пример {i}\n"
            md += f"- **Текст**: {example.get('text', 'N/A')}\n"
            md += f"- **Метка**: {example.get('label', 'N/A')}\n"
            if 'explanation' in example:
                md += f"- **Пояснение**: {example['explanation']}\n"
        
        md += "\n### Граничные случаи\n"
        for case in self.edge_cases:
            md += f"- {case}\n"
        
        md += "\n### Рекомендации по разметке\n"
        for guideline in self.guidelines:
            md += f"- {guideline}\n"
        
        md += f"\n---\n*Создано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return md

class AnnotationAgent:
    """
    LLM-агент для автоматической разметки текстовых данных на основе GPT.
    Поддерживает классификацию типа документа (кодексы РФ).
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini",
                 confidence_threshold: float = 0.7):
        """
        Инициализация агента с OpenAI GPT
        
        Parameters:
        -----------
        api_key : str
            API ключ OpenAI
        model_name : str
            Название модели GPT (gpt-4o-mini, gpt-4, gpt-3.5-turbo)
        confidence_threshold : float
            Порог уверенности для флагов low confidence
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.quality_report = None
        self.annotation_history = []
        self.data_info = {}
        
        # Инициализируем OpenAI клиент
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            print(f"✅ OpenAI клиент инициализирован (модель: {model_name})")
        else:
            raise ValueError("API ключ OpenAI не предоставлен. Укажите api_key или установите OPENAI_API_KEY")
    
    def load_csv_from_folder(self, folder_path: str = "data/raw", 
                            file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        Загружает CSV файлы из папки data/raw.
        
        Parameters:
        -----------
        folder_path : str
            Путь к папке с данными
        file_pattern : str
            Паттерн файлов для загрузки
            
        Returns:
        --------
        pd.DataFrame : Объединенный датафрейм
        """
        print(f"📂 Загружаю CSV файлы из папки: {folder_path}")
        
        all_files = glob.glob(os.path.join(folder_path, file_pattern))
        
        if not all_files:
            raise FileNotFoundError(f"CSV файлы не найдены в {folder_path}")
        
        dfs = []
        for file in all_files:
            print(f"  • Загружаю: {os.path.basename(file)}")
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"    ✓ Загружено {len(df)} записей, {len(df.columns)} колонок")
            except Exception as e:
                print(f"    ✗ Ошибка при загрузке {file}: {e}")
        
        if not dfs:
            raise ValueError("Не удалось загрузить ни одного CSV файла")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        self.data_info = {
            'source_folder': folder_path,
            'files_loaded': len(all_files),
            'total_records': len(combined_df),
            'columns': list(combined_df.columns)
        }
        
        print(f"\n✅ Всего загружено: {len(combined_df)} записей, {len(combined_df.columns)} колонок")
        print(f"📋 Доступные колонки: {list(combined_df.columns)}")
        
        return combined_df
    
    def _classify_with_gpt(self, text: str, candidate_labels: List[str]) -> Tuple[str, float]:
        """
        Классифицирует текст с помощью GPT.
        
        Parameters:
        -----------
        text : str
            Текст для классификации
        candidate_labels : List[str]
            Список возможных меток
            
        Returns:
        --------
        Tuple[str, float] : (метка, уверенность)
        """
        # Описания типов документов
        doc_descriptions = {
            "civil_code_rf": "Гражданский кодекс РФ - статьи о сделках, договорах, обязательствах, праве собственности",
            "labor_code_rf": "Трудовой кодекс РФ - статьи о трудовых договорах, рабочем времени, отпусках, увольнении",
            "tax_code_rf": "Налоговый кодекс РФ - статьи о налогах, сборах, налоговой базе, ставках",
            "administrative_code_rf": "КоАП РФ - статьи об административных правонарушениях, штрафах",
            "criminal_code_rf": "Уголовный кодекс РФ - статьи о преступлениях, наказаниях",
            "civil_procedure_code_rf": "ГПК РФ - статьи о гражданском судопроизводстве, исковых заявлениях",
            "criminal_procedure_code_rf": "УПК РФ - статьи об уголовном судопроизводстве, следственных действиях",
            "land_code_rf": "Земельный кодекс РФ - статьи о земельных участках, кадастре",
            "housing_code_rf": "Жилищный кодекс РФ - статьи о жилых помещениях, коммунальных услугах",
            "arbitration_code_rf": "АПК РФ - статьи об арбитражном судопроизводстве",
            "family_code_rf": "Семейный кодекс РФ - статьи о браке, родительских правах, алиментах",
            "budget_code_rf": "Бюджетный кодекс РФ - статьи о бюджете, доходах, расходах",
            "forest_code_rf": "Лесной кодекс РФ - статьи о лесных участках, лесопользовании",
            "urban_code_rf": "Градостроительный кодекс РФ - статьи о градостроительной деятельности"
        }
        
        # Формируем промпт для GPT с более строгим форматом
        labels_str = "\n".join([f"- {label}: {doc_descriptions.get(label, '')[:100]}" for label in candidate_labels])
        
        prompt = f"""Ты - эксперт по классификации документов российского законодательства.
        Определи, к какому кодексу относится следующий текст.

        Возможные варианты:
        {labels_str}

        Текст:
        {text[:2000]}

        Ответь ТОЛЬКО в формате JSON. Никакого дополнительного текста.
        Вот пример правильного ответа:
        {{"label": "civil_code_rf", "confidence": 0.95, "reasoning": "текст содержит ссылки на договор и обязательства"}}

        Теперь твой ответ (только JSON):""".replace("\t", " ").replace("\n", " ").replace(" +", " ")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Ты - эксперт по классификации юридических документов. Отвечай только в формате JSON без пояснений."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            
            # Пробуем найти JSON в ответе
            import re
            json_match = re.search(r'\{[^{}]*\}', content)
            
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                label = result.get("label", candidate_labels[0])
                confidence = float(result.get("confidence", 0.5))
                
                # Проверяем, что метка существует
                if label not in candidate_labels:
                    # Ищем ближайшую метку
                    for candidate in candidate_labels:
                        if candidate in label or label in candidate:
                            label = candidate
                            break
                    else:
                        label = candidate_labels[0]
                        confidence = 0.3
                
                return label, min(max(confidence, 0.0), 1.0)
            else:
                # Если JSON не найден, пробуем извлечь метку из текста
                for label in candidate_labels:
                    if label in content.lower():
                        return label, 0.5
                
                # Если ничего не найдено, возвращаем первый вариант с низкой уверенностью
                print(f"  ⚠️ Не удалось распарсить ответ: {content[:100]}")
                return candidate_labels[0], 0.3
                
        except json.JSONDecodeError as e:
            print(f"  ⚠️ Ошибка парсинга JSON: {e}")
            print(f"  Ответ: {content[:200] if 'content' in locals() else 'N/A'}")
            return candidate_labels[0], 0.3
            
        except Exception as e:
            print(f"  ⚠️ Ошибка при классификации: {e}")
            return candidate_labels[0], 0.0
    
    def auto_label(self, df: pd.DataFrame, text_column: str = None, 
               candidate_labels: List[str] = None,
               multi_label: bool = False,
               task_type: str = 'document_type',
               batch_size: int = 10,
               retry_count: int = 2) -> pd.DataFrame:
        """
        Skill 1: Автоматическая разметка текстовых данных с помощью GPT.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Датафрейм с данными
        text_column : str
            Название колонки с текстом
        candidate_labels : List[str]
            Список возможных меток
        multi_label : bool
            Мульти-лейбл (не поддерживается в этой версии)
        task_type : str
            Тип задачи
        batch_size : int
            Размер батча для обработки (не используется в текущей версии, оставлено для совместимости)
        retry_count : int
            Количество повторных попыток при ошибке
            
        Returns:
        --------
        pd.DataFrame : Датафрейм с добавленными метками
        """
        print(f"\n🏷️ Автоматическая разметка с помощью {self.model_name}")
        print("-" * 50)
        
        # Автоматическое определение текстовой колонки
        if text_column is None:
            text_column = self._find_text_column(df)
            if text_column:
                print(f"  • Использую текстовую колонку: '{text_column}'")
            else:
                print("❌ Не найдена текстовая колонка в данных")
                return df
        
        if candidate_labels is None:
            candidate_labels = self._get_default_document_types()
            print(f"  • Использую {len(candidate_labels)} типов документов")
        
        df_labeled = df.copy()
        predictions = []
        confidences = []
        
        print(f"  • Размечаю {len(df)} примеров с помощью GPT...")
        
        for idx, row in df.iterrows():
            text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            
            if not text.strip():
                predictions.append(candidate_labels[0])
                confidences.append(0.0)
                continue
            
            # Повторные попытки при ошибке
            for attempt in range(retry_count + 1):
                try:
                    label, confidence = self._classify_with_gpt(text, candidate_labels)
                    predictions.append(label)
                    confidences.append(confidence)
                    break
                except Exception as e:
                    if attempt == retry_count:
                        print(f"  ✗ Ошибка после {retry_count} попыток для строки {idx}: {e}")
                        predictions.append(candidate_labels[0])
                        confidences.append(0.0)
                    else:
                        import time
                        time.sleep(1)  # Ждем перед повторной попыткой
            
            # Прогресс-бар
            if (idx + 1) % 10 == 0:
                print(f"    • Обработано {idx + 1}/{len(df)} примеров...")
        
        # Добавляем колонки с предсказаниями
        label_column = f'auto_label_{task_type}'
        confidence_column = f'confidence_{task_type}'
        low_conf_column = f'low_confidence_{task_type}'
        
        df_labeled[label_column] = predictions
        df_labeled[confidence_column] = confidences
        df_labeled[low_conf_column] = df_labeled[confidence_column] < self.confidence_threshold
        
        # Статистика
        print(f"\n✅ Разметка завершена")
        print(f"  • Всего размечено: {len(df_labeled)}")
        print(f"  • Распределение меток:")
        label_counts = pd.Series(predictions).value_counts()
        
        # Добавляем человекочитаемые названия для статистики
        doc_names = {
            "civil_code_rf": "ГК РФ",
            "labor_code_rf": "ТК РФ",
            "tax_code_rf": "НК РФ",
            "administrative_code_rf": "КоАП РФ",
            "criminal_code_rf": "УК РФ",
            "civil_procedure_code_rf": "ГПК РФ",
            "criminal_procedure_code_rf": "УПК РФ",
            "land_code_rf": "ЗК РФ",
            "housing_code_rf": "ЖК РФ",
            "arbitration_code_rf": "АПК РФ",
            "family_code_rf": "СК РФ",
            "budget_code_rf": "БК РФ",
            "forest_code_rf": "ЛК РФ",
            "urban_code_rf": "ГрК РФ"
        }
        
        for label, count in label_counts.head(10).items():
            short_name = doc_names.get(label, label)
            print(f"    - {short_name}: {count} ({count/len(df_labeled)*100:.1f}%)")
        
        print(f"  • Средняя уверенность: {df_labeled[confidence_column].mean():.3f}")
        print(f"  • Низкая уверенность (<{self.confidence_threshold}): {df_labeled[low_conf_column].sum()} примеров")
        
        return df_labeled
    
    def predict_document_type(self, df: pd.DataFrame, text_column: str = None,
                            document_types: List[str] = None) -> pd.DataFrame:
        """
        Специализированный метод для предсказания типа документа (Кодексы РФ).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Датафрейм с данными
        text_column : str
            Название колонки с текстом
        document_types : List[str]
            Список возможных типов документов
            
        Returns:
        --------
        pd.DataFrame : Датафрейм с добавленными метками
        """
        print(f"\n📄 Предсказание типа документа (Кодексы РФ) с помощью {self.model_name}")
        print("-" * 50)
        
        if document_types is None:
            document_types = self._get_default_document_types()
        
        df_labeled = self.auto_label(
            df, 
            text_column=text_column,
            candidate_labels=document_types,
            multi_label=False,
            task_type='document_type'
        )
        
        # Добавляем человекочитаемые названия
        doc_names = {
            "civil_code_rf": "Гражданский кодекс РФ (ГК РФ)",
            "labor_code_rf": "Трудовой кодекс РФ (ТК РФ)",
            "tax_code_rf": "Налоговый кодекс РФ (НК РФ)",
            "administrative_code_rf": "КоАП РФ",
            "criminal_code_rf": "Уголовный кодекс РФ (УК РФ)",
            "civil_procedure_code_rf": "ГПК РФ",
            "criminal_procedure_code_rf": "УПК РФ",
            "land_code_rf": "Земельный кодекс РФ (ЗК РФ)",
            "housing_code_rf": "Жилищный кодекс РФ (ЖК РФ)",
            "arbitration_code_rf": "АПК РФ",
            "family_code_rf": "Семейный кодекс РФ (СК РФ)",
            "budget_code_rf": "Бюджетный кодекс РФ (БК РФ)",
            "forest_code_rf": "Лесной кодекс РФ (ЛК РФ)",
            "urban_code_rf": "Градостроительный кодекс РФ (ГрК РФ)"
        }
        
        df_labeled['document_type_name'] = df_labeled['auto_label_document_type'].map(doc_names)
        
        return df_labeled
    
    def _get_default_document_types(self) -> List[str]:
        """Возвращает стандартные типы документов (Кодексы РФ)"""
        return [
            "civil_code_rf",              # Гражданский кодекс РФ
            "labor_code_rf",              # Трудовой кодекс РФ
            "tax_code_rf",                # Налоговый кодекс РФ
            "administrative_code_rf",     # КоАП РФ
            "criminal_code_rf",           # Уголовный кодекс РФ
            "civil_procedure_code_rf",    # ГПК РФ
            "criminal_procedure_code_rf", # УПК РФ
            "land_code_rf",               # Земельный кодекс РФ
            "housing_code_rf",            # Жилищный кодекс РФ
            "arbitration_code_rf",        # АПК РФ
            "family_code_rf",             # Семейный кодекс РФ
            "budget_code_rf",             # Бюджетный кодекс РФ
            "forest_code_rf",             # Лесной кодекс РФ
            "urban_code_rf"               # Градостроительный кодекс РФ
        ]
    
    def generate_spec(self, df: pd.DataFrame, task: str = 'document_type_classification',
                     class_descriptions: Dict[str, str] = None,
                     task_type: str = 'document_type') -> AnnotationSpec:
        """Генерирует спецификацию разметки"""
        print(f"\n📝 Генерация спецификации разметки")
        print("-" * 50)
        
        if class_descriptions is None:
            class_descriptions = self._get_document_type_descriptions()
        
        # Генерируем примеры из данных
        examples = []
        text_column = self._find_text_column(df)
        label_column = f'auto_label_{task_type}'
        
        if label_column in df.columns and text_column:
            for label in list(class_descriptions.keys())[:5]:
                label_examples = df[df[label_column] == label].head(2)
                for _, row in label_examples.iterrows():
                    text = str(row[text_column])[:200]
                    if text:
                        examples.append({
                            'text': text,
                            'label': label,
                            'explanation': f"Пример текста из {label}"
                        })
        
        spec = AnnotationSpec(
            task=task,
            description="Определение типа кодекса РФ по тексту статьи или фрагмента документа",
            classes=class_descriptions,
            examples=examples,
            edge_cases=self._get_edge_cases(task_type),
            guidelines=self._get_guidelines(task_type)
        )
        
        spec_md = spec.to_markdown()
        with open("annotation_spec.md", "w", encoding='utf-8') as f:
            f.write(spec_md)
        
        print(f"✅ Спецификация сохранена в annotation_spec.md")
        
        return spec
    
    def _get_document_type_descriptions(self) -> Dict[str, str]:
        """Возвращает описания для типов документов (Кодексы РФ)"""
        return {
            "civil_code_rf": "Гражданский кодекс РФ - регулирует гражданско-правовые отношения, сделки, договоры, обязательства, право собственности",
            "labor_code_rf": "Трудовой кодекс РФ - регулирует трудовые отношения, трудовые договоры, рабочее время, отпуска, увольнение",
            "tax_code_rf": "Налоговый кодекс РФ - регулирует налоговые отношения, виды налогов, налоговую базу, ставки, порядок уплаты",
            "administrative_code_rf": "КоАП РФ - регулирует административные правонарушения и ответственность, штрафы, предупреждения",
            "criminal_code_rf": "Уголовный кодекс РФ - регулирует преступления и уголовную ответственность, виды наказаний",
            "civil_procedure_code_rf": "ГПК РФ - регулирует порядок гражданского судопроизводства, исковые заявления, судебные разбирательства",
            "criminal_procedure_code_rf": "УПК РФ - регулирует порядок уголовного судопроизводства, следственные действия, меры пресечения",
            "land_code_rf": "Земельный кодекс РФ - регулирует земельные отношения, земельные участки, кадастр, категории земель",
            "housing_code_rf": "Жилищный кодекс РФ - регулирует жилищные отношения, права собственников, коммунальные услуги",
            "arbitration_code_rf": "АПК РФ - регулирует судопроизводство в арбитражных судах, экономические споры",
            "family_code_rf": "Семейный кодекс РФ - регулирует семейные отношения, брак, права родителей, алименты",
            "budget_code_rf": "Бюджетный кодекс РФ - регулирует бюджетные отношения, доходы, расходы, бюджетный процесс",
            "forest_code_rf": "Лесной кодекс РФ - регулирует лесные отношения, лесопользование, аренду лесных участков",
            "urban_code_rf": "Градостроительный кодекс РФ - регулирует градостроительную деятельность, строительство, планировку территорий"
        }
    
    def _get_edge_cases(self, task_type: str) -> List[str]:
        """Возвращает граничные случаи"""
        return [
            "Текст может содержать ссылки на несколько кодексов одновременно",
            "Статья может быть изменена или дополнена (с указанием редакции)",
            "Текст может содержать только номер статьи без полного содержания",
            "Фрагмент может быть слишком коротким для точной классификации",
            "Текст может содержать судебную практику или комментарии, а не сам кодекс"
        ]
    
    def _get_guidelines(self, task_type: str) -> List[str]:
        """Возвращает рекомендации по разметке"""
        return [
            "Обращайте внимание на номер статьи (ст. X) - это ключевой признак",
            "Учитывайте специфическую терминологию, характерную для каждого кодекса",
            "При наличии ссылок на другие кодексы выбирайте основной предмет регулирования",
            "Для коротких фрагментов используйте контекст, если он доступен",
            "Если текст содержит несколько кодексов, выбирайте тот, который упоминается первым или является основным"
        ]
    
    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """Находит колонку с текстом, приоритет - content"""
        potential_text_cols = ['content', 'text', 'text_content', 'article_text', 
                            'document_text', 'message', 'review', 'comment', 
                            'description', 'body', 'sentence', 'title']
        
        for col in potential_text_cols:
            if col in df.columns:
                return col
        
        # Если не нашли, берем первую строковую колонку
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(string_cols) > 0:
            return string_cols[0]
        
        return None
    
    def check_quality(self, df_labeled: pd.DataFrame, 
                     ground_truth_column: Optional[str] = None,
                     task_type: str = 'document_type') -> QualityMetrics:
        """Оценка качества разметки"""
        print(f"\n📊 Оценка качества разметки")
        print("-" * 50)
        
        label_column = f'auto_label_{task_type}'
        
        if label_column in df_labeled.columns:
            label_dist = df_labeled[label_column].value_counts().to_dict()
        else:
            label_dist = {}
        
        confidence_column = f'confidence_{task_type}'
        if confidence_column in df_labeled.columns:
            confidence_mean = df_labeled[confidence_column].mean()
            confidence_std = df_labeled[confidence_column].std()
            low_conf_column = f'low_confidence_{task_type}'
            low_confidence_count = df_labeled[low_conf_column].sum() if low_conf_column in df_labeled.columns else 0
        else:
            confidence_mean = 0.0
            confidence_std = 0.0
            low_confidence_count = 0
        
        kappa = None
        if ground_truth_column and ground_truth_column in df_labeled.columns and label_column in df_labeled.columns:
            try:
                valid_mask = df_labeled[ground_truth_column].notna()
                y_true = df_labeled[valid_mask][ground_truth_column].values
                y_pred = df_labeled[valid_mask][label_column].values
                
                if len(y_true) > 0:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    all_labels = list(set(y_true) | set(y_pred))
                    le.fit(all_labels)
                    kappa = cohen_kappa_score(le.transform(y_true), le.transform(y_pred))
                    print(f"  • Cohen's κ: {kappa:.3f}")
            except Exception as e:
                print(f"  ⚠️ Не удалось рассчитать kappa: {e}")
        
        metrics = QualityMetrics(
            kappa=kappa,
            label_distribution=label_dist,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            low_confidence_count=low_confidence_count,
            low_confidence_threshold=self.confidence_threshold,
            total_samples=len(df_labeled)
        )
        
        print(f"  • Распределение меток:")
        for label, count in list(label_dist.items())[:10]:
            print(f"    - {label}: {count} ({count/len(df_labeled)*100:.1f}%)")
        print(f"  • Средняя уверенность: {confidence_mean:.3f}")
        print(f"  • Низкая уверенность: {low_confidence_count} примеров")
        
        return metrics
    
    def export_to_labelstudio(self, df_labeled: pd.DataFrame, 
                             output_path: str = "labelstudio_import.json",
                             text_column: str = None,
                             label_column: str = None) -> None:
        """Экспорт в LabelStudio"""
        print(f"\n📤 Экспорт в LabelStudio")
        print("-" * 50)
        
        if text_column is None:
            text_column = self._find_text_column(df_labeled)
        
        if label_column is None:
            label_column = 'auto_label_document_type'
        
        if text_column is None:
            print("❌ Не найдена текстовая колонка")
            return
        
        labelstudio_data = []
        
        for idx, row in df_labeled.iterrows():
            label_list = [{"text": str(row[label_column])}] if label_column in row and pd.notna(row[label_column]) else []
            
            item = {
                "id": idx,
                "data": {"text": str(row[text_column]) if pd.notna(row[text_column]) else ""},
                "annotations": [{
                    "result": [{
                        "value": {"choices": label_list},
                        "from_name": "document_type",
                        "to_name": "text",
                        "type": "choices"
                    }],
                    "was_cancelled": False,
                    "ground_truth": False,
                    "created_at": datetime.now().isoformat()
                }]
            }
            
            labelstudio_data.append(item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labelstudio_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Экспортировано {len(labelstudio_data)} записей в {output_path}")
    
    def export_low_confidence_for_review(self, df_labeled: pd.DataFrame,
                                        output_path: str = "low_confidence_review.csv",
                                        text_column: str = None,
                                        task_type: str = 'document_type') -> pd.DataFrame:
        """Экспорт примеров с низкой уверенностью"""
        print(f"\n📋 Экспорт примеров с низкой уверенностью")
        print("-" * 50)
        
        low_conf_column = f'low_confidence_{task_type}'
        
        if low_conf_column not in df_labeled.columns:
            print("⚠️ Нет данных о низкой уверенности")
            return df_labeled
        
        if text_column is None:
            text_column = self._find_text_column(df_labeled)
        
        low_conf_df = df_labeled[df_labeled[low_conf_column]].copy()
        
        if len(low_conf_df) == 0:
            print("✅ Нет примеров с низкой уверенностью")
            return low_conf_df
        
        low_conf_df['manual_label'] = ""
        low_conf_df['notes'] = ""
        low_conf_df['reviewed'] = False
        
        columns_to_keep = [text_column, 'auto_label_document_type', 'confidence_document_type', 
                          'manual_label', 'notes', 'reviewed']
        existing_cols = [col for col in columns_to_keep if col in low_conf_df.columns]
        low_conf_df = low_conf_df[existing_cols]
        low_conf_df = low_conf_df.rename(columns={'auto_label_document_type': 'auto_label'})
        
        low_conf_df.to_csv(output_path, index=False)
        
        print(f"✅ Экспортировано {len(low_conf_df)} примеров в {output_path}")
        
        return low_conf_df
    
    def run(self,):

        """
        # AnnotationAgent с GPT-4o-mini - Определение типа кодекса РФ
        ## Автоматическая разметка статей российского законодательства
        """

        print("="*80)
        print("AnnotationAgent - Определение типа кодекса РФ с помощью LLM")
        print("="*80)

        print("\n" + "="*80)
        print("ЧАСТЬ 1: Загрузка данных")
        print("="*80)

        # Загружаем данные из папки data/raw
        try:
            df = self.load_csv_from_folder("data/raw", file_pattern="*.csv").iloc[:20]
            print(f"\n✅ Загружено {len(df)} записей")
            print(f"📋 Колонки: {list(df.columns)}")
            
            # Проверяем наличие текстовой колонки content
            if 'content' in df.columns:
                text_col = 'content'
                print(f"📝 Текстовая колонка: '{text_col}'")
            else:
                # Если нет content, ищем другие варианты
                text_col = self._find_text_column(df)
                if text_col:
                    print(f"📝 Использую текстовую колонку: '{text_col}'")
                    print(f"⚠️ Внимание: ожидалась колонка 'content', но найдена '{text_col}'")
                else:
                    print("❌ Не найдена текстовая колонка в данных")
                    print("   Ожидалась колонка 'content'")
                    df = None
                
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            print("Пожалуйста, убедитесь, что в папке data/raw есть CSV файлы с колонкой 'content'")
            df = None

        if df is not None:
            print("\n📝 Примеры данных (первые 3 записи):")
            for i in range(min(3, len(df))):
                content_preview = str(df.iloc[i].get('content', df.iloc[i].get(text_col, '')))[:150]
                print(f"\n  {i+1}. {content_preview}...")
                
                # Также показываем другие колонки, если есть
                for col in df.columns:
                    if col != text_col and col != 'content':
                        print(f"     {col}: {df.iloc[i].get(col, 'N/A')}")

        if df is not None:
            print("\n" + "="*80)
            print("ЧАСТЬ 2: Определение типа кодекса РФ")
            print("="*80)
            
            print("⚠️ Обработка может занять несколько минут в зависимости от количества данных...")
            start_time = time.time()
            
            # Определяем тип документа, явно указываем колонку content
            df_labeled = self.predict_document_type(
                df, 
                text_column='content'  # Явно указываем колонку content
            )
            
            elapsed_time = time.time() - start_time
            print(f"\n⏱️ Время обработки: {elapsed_time:.2f} сек")
            
            print("\n📊 Результаты классификации:")
            print(f"  • Всего документов: {len(df_labeled)}")
            print(f"  • Распределение типов:")
            
            # Добавляем человекочитаемые названия
            doc_names = {
                "civil_code_rf": "Гражданский кодекс РФ (ГК РФ)",
                "labor_code_rf": "Трудовой кодекс РФ (ТК РФ)",
                "tax_code_rf": "Налоговый кодекс РФ (НК РФ)",
                "administrative_code_rf": "КоАП РФ",
                "criminal_code_rf": "Уголовный кодекс РФ (УК РФ)",
                "civil_procedure_code_rf": "ГПК РФ",
                "criminal_procedure_code_rf": "УПК РФ",
                "land_code_rf": "Земельный кодекс РФ (ЗК РФ)",
                "housing_code_rf": "Жилищный кодекс РФ (ЖК РФ)",
                "arbitration_code_rf": "АПК РФ",
                "family_code_rf": "Семейный кодекс РФ (СК РФ)",
                "budget_code_rf": "Бюджетный кодекс РФ (БК РФ)",
                "forest_code_rf": "Лесной кодекс РФ (ЛК РФ)",
                "urban_code_rf": "Градостроительный кодекс РФ (ГрК РФ)"
            }
            
            for doc_type, count in df_labeled['auto_label_document_type'].value_counts().items():
                type_name = doc_names.get(doc_type, doc_type)
                print(f"    - {type_name}: {count} ({count/len(df_labeled)*100:.1f}%)")
            
            print(f"\n  • Средняя уверенность: {df_labeled['confidence_document_type'].mean():.3f}")
            print(f"  • Низкая уверенность (<0.7): {df_labeled['low_confidence_document_type'].sum()} примеров")
            
            # Показываем примеры
            print("\n📝 Примеры классификации:")
            for i in range(min(5, len(df_labeled))):
                row = df_labeled.iloc[i]
                content_preview = str(row['content'])[:100] + "..." if len(str(row['content'])) > 100 else str(row['content'])
                type_name = doc_names.get(row['auto_label_document_type'], row['auto_label_document_type'])
                print(f"\n  {i+1}. Текст: {content_preview}")
                print(f"     Тип: {type_name}")
                print(f"     Уверенность: {row['confidence_document_type']:.3f}")
                
                # Показываем другие колонки, если есть
                for col in df_labeled.columns:
                    if col not in ['content', 'auto_label_document_type', 'confidence_document_type', 
                                'low_confidence_document_type', 'document_type_name']:
                        if pd.notna(row[col]):
                            print(f"     {col}: {row[col]}")

        if df is not None:
            print("\n" + "="*80)
            print("ЧАСТЬ 3: Генерация спецификации разметки")
            print("="*80)
            
            spec = self.generate_spec(
                df_labeled,
                task="document_type_classification",
                task_type='document_type'
            )
            
            with open(os.path.join("data", "markdown", "annotation_spec.md"), "w", encoding='utf-8') as f:
                f.write(spec.to_markdown())
            
            print("\n✅ Спецификация сохранена в annotation_spec.md")

        if df is not None:
            print("\n" + "="*80)
            print("ЧАСТЬ 4: Оценка качества")
            print("="*80)
            
            # Если в данных есть колонка с истинными метками
            ground_truth = None
            possible_truth_cols = ['true_document_type', 'true_label', 'actual_type', 'ground_truth', 'document_type']
            for col in possible_truth_cols:
                if col in df_labeled.columns:
                    ground_truth = col
                    print(f"✅ Найдены истинные метки в колонке '{col}'")
                    break
            
            metrics = self.check_quality(
                df_labeled,
                ground_truth_column=ground_truth,
                task_type='document_type'
            )
            
            print(f"\n📊 Метрики качества:")
            if metrics.kappa is not None:
                print(f"  • Cohen's κ: {metrics.kappa:.3f}")
            else:
                print(f"  • Cohen's κ: N/A (нет истинных меток)")
            print(f"  • Средняя уверенность: {metrics.confidence_mean:.3f}")
            print(f"  • Низкая уверенность: {metrics.low_confidence_count}")

        if df is not None:
            print("\n" + "="*80)
            print("ЧАСТЬ 5: Экспорт в LabelStudio")
            print("="*80)
            
            self.export_to_labelstudio(
                df_labeled,
                output_path="labelstudio_import.json",
                text_column='content',  # Явно указываем колонку content
                label_column='auto_label_document_type'
            )

        if df is not None:
            print("\n" + "="*80)
            print("БОНУС: Экспорт примеров с низкой уверенностью")
            print("="*80)
            
            low_conf_df = self.export_low_confidence_for_review(
                df_labeled,
                output_path="low_confidence_review.csv",
                text_column='content',  # Явно указываем колонку content
                task_type='document_type'
            )
            
            if len(low_conf_df) > 0:
                print(f"\n📋 Экспортировано {len(low_conf_df)} примеров для ручной проверки")
                print("\n📝 Примеры, требующие внимания:")
                for i in range(min(3, len(low_conf_df))):
                    row = low_conf_df.iloc[i]
                    content_preview = str(row['content'])[:100] + "..." if len(str(row['content'])) > 100 else str(row['content'])
                    print(f"\n  {i+1}. Текст: {content_preview}")
                    type_name = doc_names.get(row['auto_label'], row['auto_label'])
                    print(f"     Предсказанный тип: {type_name} (уверенность: {row['confidence_document_type']:.3f})")
            else:
                print("✅ Нет примеров с низкой уверенностью")

        if df is not None:
            print("\n" + "="*80)
            print("Сохранение результатов")
            print("="*80)
            
            # Создаем папку для результатов
            os.makedirs("data/clean", exist_ok=True)
            
            # Сохраняем размеченные данные
            output_file = "data/clean/labeled_documents.csv"
            df_labeled.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✅ Размеченные данные сохранены в {output_file}")
            
            # Сохраняем краткий отчет
            report_data = []
            for doc_type in df_labeled['auto_label_document_type'].unique():
                type_name = doc_names.get(doc_type, doc_type)
                count = len(df_labeled[df_labeled['auto_label_document_type'] == doc_type])
                avg_conf = df_labeled[df_labeled['auto_label_document_type'] == doc_type]['confidence_document_type'].mean()
                report_data.append({
                    'Тип документа': type_name,
                    'Количество': count,
                    'Процент': f"{count/len(df_labeled)*100:.1f}%",
                    'Средняя уверенность': f"{avg_conf:.3f}"
                })
            
            report_df = pd.DataFrame(report_data)
            report_df = report_df.sort_values('Количество', ascending=False)
            report_df.to_csv("data/clean/classification_report.csv", index=False, encoding='utf-8-sig')
            print(f"✅ Отчет сохранен в data/clean/classification_report.csv")
            
            # Показываем отчет
            print("\n📊 ОТЧЕТ ПО КЛАССИФИКАЦИИ:")
            print(report_df.to_string(index=False))

        if df is not None:
            print("\n" + "="*80)
            print("Дополнительный анализ данных")
            print("="*80)
            
            # Статистика по длине текста
            if 'content' in df_labeled.columns:
                df_labeled['content_length'] = df_labeled['content'].astype(str).str.len()
                print(f"\n📏 Статистика длины текста (символы):")
                print(f"  • Минимальная: {df_labeled['content_length'].min()}")
                print(f"  • Максимальная: {df_labeled['content_length'].max()}")
                print(f"  • Средняя: {df_labeled['content_length'].mean():.0f}")
                print(f"  • Медианная: {df_labeled['content_length'].median():.0f}")
            
            # Анализ уверенности по типам
            print(f"\n📊 Уверенность классификации по типам кодексов:")
            type_confidence = df_labeled.groupby('auto_label_document_type')['confidence_document_type'].agg(['mean', 'std', 'count'])
            type_confidence = type_confidence.sort_values('mean', ascending=False)
            
            for doc_type, row in type_confidence.iterrows():
                type_name = doc_names.get(doc_type, doc_type)
                print(f"  • {type_name}:")
                print(f"    - Средняя уверенность: {row['mean']:.3f} (±{row['std']:.3f})")
                print(f"    - Количество: {int(row['count'])}")

        print("\n" + "="*80)
        print("✅ ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("="*80)

        if df is not None:
            kappa_str = f"{metrics.kappa:.3f}" if metrics.kappa is not None else "N/A"
            
            print(f"""
        📊 ИТОГОВАЯ СТАТИСТИКА:

        • Модель: {self.model_name}
        • Источник данных: data/raw/*.csv
        • Текстовая колонка: content
        • Всего документов: {len(df_labeled)}
        • Типов кодексов: {len(df_labeled['auto_label_document_type'].unique())}
        • Cohen's κ: {kappa_str}
        • Средняя уверенность: {metrics.confidence_mean:.3f}
        • Низкая уверенность (<{self.confidence_threshold}): {metrics.low_confidence_count}

        📁 СОЗДАННЫЕ ФАЙЛЫ:
        • annotation_spec.md - спецификация разметки
        • labelstudio_import.json - для импорта в LabelStudio
        • low_confidence_review.csv - примеры для ручной проверки
        • data/clean/labeled_documents.csv - размеченные данные
        • data/clean/classification_report.csv - отчет по классификации

        💡 ПРИМЕЧАНИЕ:
        • Все операции выполнены с использованием колонки 'content'
        • Если в ваших данных используется другое название текстовой колонки,
            укажите его в параметре text_column при вызове методов
        """)

        # Проверка структуры финального датасета
        if df is not None:
            print("\n📋 СТРУКТУРА ФИНАЛЬНОГО ДАТАСЕТА:")
            print(f"  • Всего колонок: {len(df_labeled.columns)}")
            print(f"  • Колонки: {list(df_labeled.columns)}")
            print("\n  • Первые 3 строки финального датасета:")
            display_df = df_labeled[['content', 'auto_label_document_type', 'confidence_document_type', 
                                    'low_confidence_document_type', 'document_type_name']].head(3)
            print(display_df.to_string())