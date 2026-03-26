# data_quality_agent_text.py (исправленная версия)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings
import os
import glob
from pathlib import Path
import re
from collections import Counter
import json

warnings.filterwarnings('ignore')

# Для бонусной части с LLM (OpenAI)
from openai import OpenAI

@dataclass
class QualityReport:
    """Датакласс для отчета о качестве текстовых данных"""
    missing: Dict[str, Dict]
    duplicates: Dict[str, Union[int, float]]
    outliers: Dict[str, Dict]  # Для текста это могут быть аномалии длины, редкие слова и т.д.
    imbalance: Optional[Dict] = None
    text_stats: Optional[Dict] = None  # Специфичная статистика для текста
    
    def __str__(self):
        """Красивое строковое представление отчета"""
        lines = ["📊 Отчет о качестве текстовых данных:"]
        
        if self.missing:
            lines.append("\n  Пропущенные значения:")
            for col, stats in self.missing.items():
                lines.append(f"    • {col}: {stats['count']} ({stats['percentage']}%)")
        
        lines.append(f"\n  Дубликаты: {self.duplicates['count']} ({self.duplicates['percentage']}%)")
        
        if self.outliers:
            lines.append("\n  Текстовые аномалии:")
            for col, stats in self.outliers.items():
                # Проверяем тип аномалии
                if isinstance(stats, dict) and 'type' in stats:
                    if stats['type'] == 'text_length_anomaly':
                        lines.append(f"    • {col}: {stats['count']} аномалий по длине ({stats['percentage']}%)")
                    elif stats['type'] == 'special_elements':
                        lines.append(f"    • {col}: URL={stats.get('has_url', 0)}, Email={stats.get('has_email', 0)}, "
                                   f"Спецсимволы={stats.get('has_special_chars', 0)}")
                elif isinstance(stats, dict) and 'has_url' in stats:
                    # Это специальные элементы
                    lines.append(f"    • {col}: URL={stats.get('has_url', 0)}, Email={stats.get('has_email', 0)}, "
                               f"Спецсимволы={stats.get('has_special_chars', 0)}")
                else:
                    lines.append(f"    • {col}: {stats}")
        
        if self.imbalance:
            lines.append(f"\n  Дисбаланс классов: {'ДА' if self.imbalance['is_imbalanced'] else 'НЕТ'}")
            lines.append(f"    • Доминирующий класс: {self.imbalance['dominant_class']} ({self.imbalance['dominant_percentage']}%)")
        
        if self.text_stats:
            lines.append("\n  Статистика текстовых полей:")
            for col, stats in self.text_stats.items():
                lines.append(f"    • {col}: средняя длина = {stats['avg_length']:.1f}, "
                           f"макс = {stats['max_length']}, мин = {stats['min_length']}, "
                           f"пустых = {stats['empty_count']} ({stats['empty_pct']}%)")
        
        return "\n".join(lines)

@dataclass
class ComparisonReport:
    """Датакласс для сравнения до/после очистки"""
    before: Dict
    after: Dict
    changes: Dict
    summary: str
    
    def display_table(self) -> pd.DataFrame:
        """Отображает таблицу сравнения"""
        comparison_df = pd.DataFrame({
            'Метрика': list(self.before.keys()),
            'До очистки': list(self.before.values()),
            'После очистки': list(self.after.values()),
            'Изменение': list(self.changes.values())
        })
        return comparison_df

class DataQualityAgent:
    """
    Агент для автоматического выявления и устранения проблем качества ТЕКСТОВЫХ данных.
    Специализирован на обработке текстовых полей.
    """
    
    def __init__(self, random_state: int = 42, use_llm: bool = False, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Инициализация агента
        
        Parameters:
        -----------
        random_state : int
            Seed для воспроизводимости
        use_llm : bool
            Использовать ли LLM для рекомендаций
        api_key : str
            API ключ OpenAI
        model : str
            Модель OpenAI для использования
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.quality_report = None
        self.fix_history = []
        self.use_llm = use_llm
        self.model = model
        
        if use_llm and api_key:
            self.llm_client = OpenAI(api_key=api_key)
            print("✅ OpenAI клиент инициализирован")
        elif use_llm:
            # Пробуем взять из переменной окружения
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = OpenAI(api_key=api_key)
                self.use_llm = True
                print("✅ OpenAI клиент инициализирован из переменной окружения")
            else:
                self.use_llm = False
                print("⚠️  LLM отключен: не предоставлен API ключ")
    
    def load_data_from_folder(self, folder_path: str, file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        Загружает данные из папки data/raw.
        
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
        print(f"📂 Загружаю данные из папки: {folder_path}")
        
        all_files = glob.glob(os.path.join(folder_path, file_pattern))
        
        if not all_files:
            raise FileNotFoundError(f"Файлы не найдены в {folder_path} по паттерну {file_pattern}")
        
        dfs = []
        for file in all_files:
            print(f"  • Загружаю: {os.path.basename(file)}")
            try:
                # Пробуем разные форматы
                if file.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.endswith('.json'):
                    df = pd.read_json(file)
                elif file.endswith('.parquet'):
                    df = pd.read_parquet(file)
                elif file.endswith('.txt'):
                    # Для текстовых файлов создаем датафрейм с одной колонкой
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    df = pd.DataFrame({'text': lines})
                else:
                    print(f"    ⚠️ Неподдерживаемый формат: {file}")
                    continue
                
                dfs.append(df)
                print(f"    ✓ Загружено {len(df)} записей")
            except Exception as e:
                print(f"    ✗ Ошибка при загрузке {file}: {e}")
        
        if not dfs:
            raise ValueError("Не удалось загрузить ни одного файла")
        
        # Объединяем все датафреймы
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\n✅ Всего загружено: {len(combined_df)} записей, {len(combined_df.columns)} колонок")
        
        return combined_df
    
    def detect_issues(self, df: pd.DataFrame, target_col: Optional[str] = None, 
                     text_cols: Optional[List[str]] = None) -> QualityReport:
        """
        Skill 1: Детектирует проблемы качества ТЕКСТОВЫХ данных.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный датафрейм
        target_col : str, optional
            Название целевой колонки для анализа дисбаланса классов
        text_cols : List[str], optional
            Список текстовых колонок для анализа. Если None, определяет автоматически.
            
        Returns:
        --------
        QualityReport : Отчет о найденных проблемах
        """
        print("🔍 Детектив текстовых данных начинает расследование...")
        
        # Определяем текстовые колонки
        if text_cols is None:
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # 1. Анализ пропущенных значений
        print("  📊 Анализирую пропущенные значения...")
        missing_stats = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            if missing_count > 0:
                missing_stats[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
        
        # 2. Анализ дубликатов
        print("  🔍 Ищу дубликаты...")
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df)) * 100
        duplicates_stats = {
            'count': int(duplicate_count),
            'percentage': round(duplicate_pct, 2)
        }
        
        # 3. Анализ текстовых аномалий (аналог выбросов для текста)
        print("  📝 Анализирую текстовые аномалии...")
        outliers_stats = {}
        text_stats = {}
        
        for col in text_cols:
            # Вычисляем статистику длины текста
            text_lengths = df[col].astype(str).str.len()
            
            # Статистика по тексту
            text_stats[col] = {
                'avg_length': round(text_lengths.mean(), 2),
                'std_length': round(text_lengths.std(), 2),
                'min_length': int(text_lengths.min()),
                'max_length': int(text_lengths.max()),
                'empty_count': int((text_lengths == 0).sum()),
                'empty_pct': round(((text_lengths == 0).sum() / len(df)) * 100, 2)
            }
            
            # Аномалии длины текста (IQR метод для длины)
            Q1 = text_lengths.quantile(0.25)
            Q3 = text_lengths.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Тексты с аномальной длиной
            anomaly_mask = (text_lengths < lower_bound) | (text_lengths > upper_bound)
            anomaly_count = anomaly_mask.sum()
            anomaly_pct = (anomaly_count / len(df)) * 100
            
            if anomaly_count > 0:
                outliers_stats[col] = {
                    'count': int(anomaly_count),
                    'percentage': round(anomaly_pct, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'type': 'text_length_anomaly'
                }
            
            # Дополнительный анализ: редкие символы, эмодзи, URL и т.д.
            # Создаем временные колонки для анализа
            has_url = df[col].astype(str).str.contains(r'https?://|www\.', case=False, na=False).sum()
            has_email = df[col].astype(str).str.contains(r'[\w\.-]+@[\w\.-]+\.\w+', na=False).sum()
            has_special_chars = df[col].astype(str).str.contains(r'[^\w\s]', na=False).sum()
            
            # Добавляем информацию о специальных элементах
            special_key = f'{col}_special_elements'
            outliers_stats[special_key] = {
                'has_url': int(has_url),
                'has_email': int(has_email),
                'has_special_chars': int(has_special_chars),
                'type': 'special_elements'
            }
        
        # 4. Анализ дисбаланса классов (если указана целевая колонка)
        imbalance_stats = None
        if target_col and target_col in df.columns:
            print("  ⚖️  Проверяю дисбаланс классов...")
            value_counts = df[target_col].value_counts()
            value_pcts = (value_counts / len(df)) * 100
            
            is_imbalanced = any(pct < 10 or pct > 90 for pct in value_pcts)
            
            imbalance_stats = {
                'is_imbalanced': is_imbalanced,
                'class_counts': {str(k): int(v) for k, v in value_counts.to_dict().items()},
                'class_percentages': {str(k): round(v, 2) for k, v in value_pcts.to_dict().items()},
                'dominant_class': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'dominant_percentage': round(value_pcts.iloc[0], 2) if len(value_counts) > 0 else 0,
                'class_count': len(value_counts)
            }
        
        # Создаем отчет
        self.quality_report = QualityReport(
            missing=missing_stats,
            duplicates=duplicates_stats,
            outliers=outliers_stats,
            imbalance=imbalance_stats,
            text_stats=text_stats
        )
        
        print("✅ Расследование завершено! Проблемы найдены.")
        print(self.quality_report)
        return self.quality_report
    
    def fix(self, df: pd.DataFrame, strategy: Dict[str, str], 
            text_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Skill 2: Исправляет проблемы качества ТЕКСТОВЫХ данных согласно стратегии.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный датафрейм
        strategy : Dict
            Стратегия очистки для каждого типа проблем.
            Пример: {
                'missing': 'fill_empty' | 'fill_placeholder' | 'drop',
                'duplicates': 'drop' | 'keep_first',
                'outliers': 'truncate' | 'remove' | 'flag',
                'text_cleaning': 'basic' | 'advanced'  # Дополнительная очистка текста
            }
            
        text_cols : List[str], optional
            Список текстовых колонок для обработки
            
        Returns:
        --------
        pd.DataFrame : Очищенный датафрейм
        """
        print("🩺 Хирург текстовых данных приступает к операции...")
        df_clean = df.copy()
        
        # Определяем текстовые колонки
        if text_cols is None:
            text_cols = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Сохраняем статистику до очистки
        before_stats = self._get_quality_stats(df_clean)
        
        # 1. Обработка пропущенных значений
        if 'missing' in strategy:
            missing_strategy = strategy['missing']
            print(f"  🧹 Обрабатываю пропуски методом: {missing_strategy}")
            
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if missing_strategy == 'fill_empty':
                        df_clean[col].fillna('', inplace=True)
                    elif missing_strategy == 'fill_placeholder':
                        df_clean[col].fillna('[MISSING]', inplace=True)
                    elif missing_strategy == 'fill_mode':
                        mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else ''
                        df_clean[col].fillna(mode_value, inplace=True)
                    elif missing_strategy == 'drop':
                        df_clean.dropna(subset=[col], inplace=True)
        
        # 2. Обработка дубликатов
        if 'duplicates' in strategy:
            dup_strategy = strategy['duplicates']
            print(f"  🔄 Обрабатываю дубликаты методом: {dup_strategy}")
            
            if dup_strategy == 'drop':
                df_clean.drop_duplicates(inplace=True)
            elif dup_strategy == 'keep_first':
                df_clean.drop_duplicates(keep='first', inplace=True)
            elif dup_strategy == 'keep_last':
                df_clean.drop_duplicates(keep='last', inplace=True)
        
        # 3. Обработка текстовых аномалий
        if 'outliers' in strategy:
            outlier_strategy = strategy['outliers']
            print(f"  📝 Обрабатываю текстовые аномалии методом: {outlier_strategy}")
            
            for col in text_cols:
                text_lengths = df_clean[col].astype(str).str.len()
                Q1 = text_lengths.quantile(0.25)
                Q3 = text_lengths.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if outlier_strategy == 'truncate':
                    # Обрезаем слишком длинные тексты
                    df_clean[col] = df_clean[col].astype(str).str[:int(upper_bound)]
                elif outlier_strategy == 'remove':
                    # Удаляем строки с аномальной длиной
                    mask = (text_lengths >= lower_bound) & (text_lengths <= upper_bound)
                    df_clean = df_clean[mask]
                elif outlier_strategy == 'flag':
                    # Создаем флаг аномалии
                    df_clean[f'{col}_anomaly_flag'] = (text_lengths < lower_bound) | (text_lengths > upper_bound)
        
        # 4. Дополнительная очистка текста
        if 'text_cleaning' in strategy:
            cleaning_level = strategy['text_cleaning']
            print(f"  ✨ Выполняю очистку текста: {cleaning_level}")
            
            for col in text_cols:
                if cleaning_level == 'basic':
                    # Базовая очистка
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                    df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
                elif cleaning_level == 'advanced':
                    # Расширенная очистка
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                    df_clean[col] = df_clean[col].str.lower()
                    df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
                    df_clean[col] = df_clean[col].str.replace(r'[^\w\s]', '', regex=True)  # Удаляем пунктуацию
                    df_clean[col] = df_clean[col].str.replace(r'https?://\S+|www\.\S+', '', regex=True)  # Удаляем URL
                    df_clean[col] = df_clean[col].str.replace(r'[\w\.-]+@[\w\.-]+\.\w+', '', regex=True)  # Удаляем email
        
        # Сохраняем статистику после очистки
        after_stats = self._get_quality_stats(df_clean)
        
        # Сохраняем в историю
        self.fix_history.append({
            'strategy': strategy,
            'before': before_stats,
            'after': after_stats
        })
        
        print("✅ Операция завершена успешно!")
        return df_clean
    
    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> ComparisonReport:
        """
        Skill 3: Сравнивает состояние данных до и после очистки.
        
        Parameters:
        -----------
        df_before : pd.DataFrame
            Датафрейм до очистки
        df_after : pd.DataFrame
            Датафрейм после очистки
            
        Returns:
        --------
        ComparisonReport : Отчет о сравнении
        """
        print("📊 Сравниваю результаты...")
        
        before_stats = self._get_quality_stats(df_before)
        after_stats = self._get_quality_stats(df_after)
        
        # Вычисляем изменения
        changes = {}
        for key in before_stats.keys():
            if key in after_stats:
                if isinstance(before_stats[key], (int, float)):
                    changes[key] = after_stats[key] - before_stats[key]
                else:
                    changes[key] = f"{before_stats[key]} → {after_stats[key]}"
        
        # Формируем текстовое резюме
        summary = self._generate_comparison_summary(before_stats, after_stats, changes)
        
        report = ComparisonReport(
            before=before_stats,
            after=after_stats,
            changes=changes,
            summary=summary
        )
        
        print(summary)
        return report
    
    def _get_quality_stats(self, df: pd.DataFrame) -> Dict:
        """Вспомогательный метод для сбора статистики качества"""
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_total': int(df.isnull().sum().sum()),
            'missing_pct': round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
            'duplicates': int(df.duplicated().sum()),
            'duplicates_pct': round((df.duplicated().sum() / len(df)) * 100, 2),
            'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        # Добавляем информацию о текстовых полях
        text_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(text_cols) > 0:
            avg_text_length = 0
            count = 0
            for col in text_cols:
                avg_text_length += df[col].astype(str).str.len().mean()
                count += 1
            if count > 0:
                stats['avg_text_length'] = round(avg_text_length / count, 2)
        
        return stats
    
    def _generate_comparison_summary(self, before: Dict, after: Dict, changes: Dict) -> str:
        """Генерирует текстовое резюме сравнения"""
        summary = f"\n📈 Итоги очистки текстовых данных:\n"
        summary += f"  • Строк: {before['rows']} → {after['rows']} ({changes.get('rows', 0):+d})\n"
        summary += f"  • Пропусков: {before['missing_total']} → {after['missing_total']} ({changes.get('missing_total', 0):+d})\n"
        summary += f"  • Дубликатов: {before['duplicates']} → {after['duplicates']} ({changes.get('duplicates', 0):+d})\n"
        summary += f"  • Память: {before['memory_mb']} MB → {after['memory_mb']} MB"
        
        if 'avg_text_length' in before and 'avg_text_length' in after:
            summary += f"\n  • Средняя длина текста: {before['avg_text_length']} → {after['avg_text_length']}"
        
        return summary
    
    def visualize_issues(self, df: pd.DataFrame, report: QualityReport, text_cols: Optional[List[str]] = None):
        """
        Визуализирует найденные проблемы качества текстовых данных.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный датафрейм
        report : QualityReport
            Отчет о качестве данных
        text_cols : List[str], optional
            Список текстовых колонок для визуализации
        """
        if text_cols is None:
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Анализ качества текстовых данных', fontsize=16)
        
        # 1. Пропущенные значения
        ax1 = axes[0, 0]
        if report.missing:
            missing_df = pd.DataFrame(report.missing).T
            missing_df['percentage'].plot(kind='barh', ax=ax1, color='salmon')
            ax1.set_title('Пропущенные значения (%)')
            ax1.set_xlabel('Процент пропусков')
            ax1.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Порог 5%')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'Нет пропущенных значений', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Пропущенные значения')
        
        # 2. Дубликаты
        ax2 = axes[0, 1]
        if report.duplicates['count'] > 0:
            dup_data = {
                'Уникальные': len(df) - report.duplicates['count'],
                'Дубликаты': report.duplicates['count']
            }
            colors = ['lightgreen', 'lightcoral']
            wedges, texts, autotexts = ax2.pie(
                dup_data.values(), 
                labels=dup_data.keys(), 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax2.set_title(f'Дубликаты: {report.duplicates["count"]} шт.')
        else:
            ax2.text(0.5, 0.5, 'Нет дубликатов', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Дубликаты')
        
        # 3. Распределение длины текста
        ax3 = axes[0, 2]
        if text_cols:
            for col in text_cols[:3]:  # Показываем первые 3 текстовые колонки
                lengths = df[col].astype(str).str.len()
                ax3.hist(lengths, bins=50, alpha=0.5, label=col)
            ax3.set_title('Распределение длины текста')
            ax3.set_xlabel('Длина текста')
            ax3.set_ylabel('Частота')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Нет текстовых колонок', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Длина текста')
        
        # 4. Текстовые аномалии (выбросы по длине)
        ax4 = axes[1, 0]
        # Фильтруем только аномалии длины текста
        length_anomalies = {k: v for k, v in report.outliers.items() 
                           if isinstance(v, dict) and v.get('type') == 'text_length_anomaly'}
        if length_anomalies:
            anomalies_df = pd.DataFrame(length_anomalies).T
            anomalies_df['percentage'].plot(kind='bar', ax=ax4, color='orange', edgecolor='black')
            ax4.set_title('Текстовые аномалии по длине (%)')
            ax4.set_ylabel('Процент аномалий')
            ax4.tick_params(axis='x', rotation=45)
            ax4.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Порог 5%')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Нет аномалий длины текста', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Текстовые аномалии')
        
        # 5. Специальные символы и ссылки
        ax5 = axes[1, 1]
        special_stats = {k: v for k, v in report.outliers.items() 
                        if isinstance(v, dict) and v.get('type') == 'special_elements'}
        if special_stats:
            special_data = []
            for col, stats in special_stats.items():
                col_name = col.replace('_special_elements', '')
                special_data.append({
                    'колонка': col_name, 
                    'URL': stats.get('has_url', 0), 
                    'Email': stats.get('has_email', 0),
                    'Спецсимволы': stats.get('has_special_chars', 0)
                })
            if special_data:
                special_df = pd.DataFrame(special_data)
                special_df.set_index('колонка', inplace=True)
                special_df.plot(kind='bar', ax=ax5)
                ax5.set_title('Специальные элементы в тексте')
                ax5.set_ylabel('Количество')
                ax5.tick_params(axis='x', rotation=45)
                ax5.legend(loc='upper right')
        else:
            ax5.text(0.5, 0.5, 'Нет данных о специальных символах', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Специальные элементы')
        
        # 6. Дисбаланс классов
        ax6 = axes[1, 2]
        if report.imbalance:
            class_pcts = report.imbalance['class_percentages']
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_pcts)))
            bars = ax6.bar(range(len(class_pcts)), list(class_pcts.values()), color=colors, edgecolor='black')
            ax6.set_title('Распределение классов')
            ax6.set_ylabel('Процент')
            ax6.set_xticks(range(len(class_pcts)))
            ax6.set_xticklabels([str(k)[:15] for k in class_pcts.keys()], rotation=45, ha='right')
            
            # Добавляем линии порогов
            ax6.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Порог 10%')
            ax6.axhline(y=90, color='darkred', linestyle='--', alpha=0.7, label='Порог 90%')
            
            # Добавляем значения
            for i, (bar, (_, pct)) in enumerate(zip(bars, class_pcts.items())):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            if report.imbalance['is_imbalanced']:
                ax6.set_facecolor('#fff3f3')
                ax6.text(0.5, 0.95, '⚠️ Обнаружен дисбаланс классов', 
                        transform=ax6.transAxes, ha='center', color='red', fontweight='bold')
            
            ax6.legend(loc='upper right')
        else:
            ax6.text(0.5, 0.5, 'Нет данных для анализа дисбаланса', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Дисбаланс классов')
        
        plt.tight_layout()
        plt.show()
    
    def get_llm_recommendation(self, task_description: str) -> str:
        """
        Использует OpenAI API для получения рекомендаций по очистке текстовых данных.
        
        Parameters:
        -----------
        task_description : str
            Описание ML-задачи
            
        Returns:
        --------
        str : Рекомендация от LLM
        """
        if not self.use_llm:
            return "❌ LLM отключен. Укажите use_llm=True и API ключ при инициализации агента."
        
        if not self.quality_report:
            return "❌ Отчет о качестве не создан. Сначала выполните detect_issues()."
        
        try:
            # Формируем промпт для OpenAI
            system_prompt = """Ты - эксперт по качеству текстовых данных (Text Data Quality Expert). 
            Твоя задача - анализировать отчеты о качестве текстовых данных и давать рекомендации по их очистке.
            Ты должен учитывать специфику NLP/ML-задачи и предлагать оптимальные стратегии.""".replace("\n", " ").replace("\t", " ")

            user_prompt = f"""Проанализируй отчет о качестве текстовых данных и описание NLP-задачи, затем порекомендуй оптимальную стратегию очистки.

            ОТЧЕТ О КАЧЕСТВЕ ТЕКСТОВЫХ ДАННЫХ:
            {self.quality_report}

            ОПИСАНИЕ NLP-ЗАДАЧИ:
            {task_description}

            ДОСТУПНЫЕ СТРАТЕГИИ:
            - missing: 'fill_empty' (заполнить пустой строкой), 'fill_placeholder' ([MISSING]), 'fill_mode' (мода), 'drop' (удалить)
            - duplicates: 'drop' (удалить все дубликаты), 'keep_first' (оставить первый), 'keep_last' (оставить последний)
            - outliers: 'truncate' (обрезать длинные тексты), 'remove' (удалить аномалии), 'flag' (создать флаг аномалии)
            - text_cleaning: 'basic' (удалить лишние пробелы), 'advanced' (нормализация, удаление URL/email, приведение к нижнему регистру)

            ТВОЯ ЗАДАЧА:
            1. Дай конкретные рекомендации по очистке текстовых данных
            2. Объясни, почему эта стратегия лучше всего подходит для данной задачи
            3. Укажи возможные риски выбранного подхода
            4. Предложи альтернативный вариант

            Формат ответа: 
            - Будь конкретным и практичным
            - Используй маркированные списки для читаемости
            - Ограничь ответ 400 словами""".replace("\n", " ").replace("\t", " ")


            # Запрос к OpenAI
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            recommendation = response.choices[0].message.content
            
            print("\n" + "="*60)
            print("🤖 РЕКОМЕНДАЦИЯ LLM ПО ОЧИСТКЕ ТЕКСТОВЫХ ДАННЫХ:")
            print("="*60)
            print(recommendation)
            
            return recommendation
            
        except Exception as e:
            error_msg = f"❌ Ошибка при обращении к OpenAI: {str(e)}"
            print(error_msg)
            return error_msg
    
    def recommend_strategy_from_llm(self, task_description: str) -> Dict[str, str]:
        """
        Получает от LLM конкретную стратегию в формате словаря.
        
        Parameters:
        -----------
        task_description : str
            Описание ML-задачи
            
        Returns:
        --------
        Dict[str, str] : Словарь со стратегией
        """
        if not self.use_llm or not self.quality_report:
            return {
                'missing': 'fill_placeholder',
                'duplicates': 'drop',
                'outliers': 'flag',
                'text_cleaning': 'advanced'
            }
        
        try:
            prompt = f"""На основе отчета о качестве текстовых данных и описания задачи, 
            верни ТОЛЬКО JSON со стратегией очистки в формате:
            {{"missing": "стратегия", "duplicates": "стратегия", "outliers": "стратегия", "text_cleaning": "стратегия"}}

            Отчет: {self.quality_report}
            Задача: {task_description}
            
            Доступные стратегии:
            - missing: fill_empty, fill_placeholder, fill_mode, drop
            - duplicates: drop, keep_first, keep_last
            - outliers: truncate, remove, flag
            - text_cleaning: basic, advanced
            
            Верни только JSON, без пояснений.""".replace("\n", " ").replace("\t", " ")
            
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            
            import json
            # Извлекаем JSON из ответа
            content = response.choices[0].message.content
            # Находим JSON в строке
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                strategy = json.loads(json_str)
                return strategy
            else:
                raise ValueError("JSON не найден в ответе")
            
        except Exception as e:
            print(f"⚠️ Ошибка при получении стратегии от LLM: {e}")
            # Возвращаем стратегию по умолчанию
            return {
                'missing': 'fill_placeholder',
                'duplicates': 'drop',
                'outliers': 'flag',
                'text_cleaning': 'advanced'
            }


    def run(self): 
        
        print("="*80)
        print("DataQualityAgent - Детектив текстовых данных")
        print("="*80)
        
        print("\n" + "="*80)
        print("ЧАСТЬ 1: ДЕТЕКТИВ - Загрузка и анализ текстовых данных")
        print("="*80)

        # # Создаем агента
        # agent = DataQualityAgent(use_llm=False)

        # Загружаем данные из папки data/raw
        print("\n📂 Загрузка данных...")
        df = self.load_data_from_folder("data/raw", file_pattern="*.csv")
        print(f"\nЗагружен датафрейм размером: {df.shape}")
        print("\nПервые 5 строк:")
        print(df.head())
        
        # Базовая информация о данных
        print("\n📊 Базовая информация о данных:")
        print(f"  • Типы данных:")
        print(df.dtypes)
        print(f"\n  • Статистика по текстовым полям:")
        print(f"    Длина текста: мин={df['content'].astype(str).str.len().min()}, "
            f"макс={df['content'].astype(str).str.len().max()}, "
            f"ср={df['content'].astype(str).str.len().mean():.2f}")
        

        # Анализируем текстовые данные
        print("\n🔍 Запуск детектива...")
        report = self.detect_issues(df, target_col='category', text_cols=['content'])

        # Визуализируем проблемы
        print("\n📊 Визуализация проблем...")
        self.visualize_issues(df, report, text_cols=['content'])

        # Детальный анализ текстовых аномалий
        print("\n" + "="*80)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ ТЕКСТОВЫХ АНОМАЛИЙ")
        print("="*80)

        # Анализируем длину текстов
        text_lengths = df['content'].astype(str).str.len()
        print(f"\n📏 Статистика длины текста:")
        print(f"  • Средняя длина: {text_lengths.mean():.2f}")
        print(f"  • Медианная длина: {text_lengths.median():.2f}")
        print(f"  • Минимальная длина: {text_lengths.min()}")
        print(f"  • Максимальная длина: {text_lengths.max()}")
        print(f"  • Стандартное отклонение: {text_lengths.std():.2f}")
        
        Q1 = text_lengths.quantile(0.25)
        Q3 = text_lengths.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        anomalies = df[(text_lengths < lower_bound) | (text_lengths > upper_bound)]
        print(f"\n⚠️ Аномалии по длине текста:")
        print(f"  • Количество аномалий: {len(anomalies)}")
        print(f"  • Процент аномалий: {(len(anomalies)/len(df))*100:.2f}%")
        print(f"  • Нижняя граница: {lower_bound:.2f}")
        print(f"  • Верхняя граница: {upper_bound:.2f}")

        # Показываем примеры аномалий
        if len(anomalies) > 0:
            print("\n📝 Примеры текстов-аномалий:")
            for i, (idx, row) in enumerate(anomalies.head(5).iterrows()):
                text_preview = str(row['content'])[:100] + "..." if len(str(row['content'])) > 100 else str(row['content'])
                print(f"  {i+1}. Длина={text_lengths[idx]}: {text_preview}")

        # Анализ специальных элементов
        print(f"\n🔍 Анализ специальных элементов:")
        print(f"  • Тексты с URL: {report.outliers.get('text_special_elements', {}).get('has_url', 0)}")
        print(f"  • Тексты с Email: {report.outliers.get('text_special_elements', {}).get('has_email', 0)}")
        print(f"  • Тексты со спецсимволами: {report.outliers.get('text_special_elements', {}).get('has_special_chars', 0)}")

        # ============================================
        # ЧАСТЬ 2: ХИРУРГ - Применение стратегий очистки
        # ============================================

        print("\n" + "="*80)
        print("ЧАСТЬ 2: ХИРУРГ - Применение стратегий очистки текстовых данных")
        print("="*80)

        # Стратегия 1: Мягкая очистка (консервативная)
        print("\n🔧 СТРАТЕГИЯ 1: Мягкая очистка (сохранение данных)")
        strategy1 = {
            'missing': 'fill_empty',
            'duplicates': 'drop',
            'outliers': 'truncate',
            'text_cleaning': 'basic'
        }

        df_clean1 = self.fix(df, strategy1, text_cols=['content'])
        comparison1 = self.compare(df, df_clean1)
        print("\nРезультаты стратегии 1:")
        print(comparison1.summary)

        # Стратегия 2: Агрессивная очистка
        print("\n🔧 СТРАТЕГИЯ 2: Агрессивная очистка (удаление проблем)")
        strategy2 = {
            'missing': 'drop',
            'duplicates': 'drop',
            'outliers': 'remove',
            'text_cleaning': 'advanced'
        }

        df_clean2 = self.fix(df, strategy2, text_cols=['content'])
        comparison2 = self.compare(df, df_clean2)

        # Стратегия 3: Гибридная очистка (рекомендуемая)
        print("\n🔧 СТРАТЕГИЯ 3: Гибридная очистка (с флагами)")
        strategy3 = {
            'missing': 'fill_placeholder',
            'duplicates': 'keep_first',
            'outliers': 'flag',
            'text_cleaning': 'advanced'
        }

        df_clean3 = self.fix(df, strategy3, text_cols=['content'])
        comparison3 = self.compare(df, df_clean3)

        # Сравнительная таблица
        print("\n" + "="*80)
        print("📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА СТРАТЕГИЙ ОЧИСТКИ")
        print("="*80)

        comparison_table = pd.DataFrame({
            'Метрика': ['Количество строк', 'Пропуски', 'Дубликаты', 'Средняя длина текста', 'Память (MB)'],
            'Исходные': [
                len(df), 
                df.isnull().sum().sum(),
                df.duplicated().sum(),
                round(df['content'].astype(str).str.len().mean(), 2),
                round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            ],
            'Стратегия 1 (Мягкая)': [
                len(df_clean1), 
                df_clean1.isnull().sum().sum(),
                df_clean1.duplicated().sum(),
                round(df_clean1['content'].astype(str).str.len().mean(), 2),
                round(df_clean1.memory_usage(deep=True).sum() / 1024**2, 2)
            ],
            'Стратегия 2 (Агрессивная)': [
                len(df_clean2), 
                df_clean2.isnull().sum().sum(),
                df_clean2.duplicated().sum(),
                round(df_clean2['content'].astype(str).str.len().mean(), 2),
                round(df_clean2.memory_usage(deep=True).sum() / 1024**2, 2)
            ],
            'Стратегия 3 (Гибридная)': [
                len(df_clean3), 
                df_clean3.isnull().sum().sum(),
                df_clean3.duplicated().sum(),
                round(df_clean3['content'].astype(str).str.len().mean(), 2),
                round(df_clean3.memory_usage(deep=True).sum() / 1024**2, 2)
            ]
        })

        print(comparison_table.to_string(index=False))

        # Визуализация сравнения стратегий
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Сравнение стратегий очистки текстовых данных', fontsize=14)

        # Сравнение количества строк
        axes[0,0].bar(['Исходные', 'Мягкая', 'Агрессивная', 'Гибридная'], 
                    [len(df), len(df_clean1), len(df_clean2), len(df_clean3)],
                    color=['#2E86AB', '#A23B72', '#F18F01', '#C73D23'])
        axes[0,0].set_title('Количество строк')
        axes[0,0].set_ylabel('Строки')
        for i, v in enumerate([len(df), len(df_clean1), len(df_clean2), len(df_clean3)]):
            axes[0,0].text(i, v + 5, str(v), ha='center', fontweight='bold')

        # Сравнение пропусков
        axes[0,1].bar(['Исходные', 'Мягкая', 'Агрессивная', 'Гибридная'],
                    [df.isnull().sum().sum(), df_clean1.isnull().sum().sum(), 
                    df_clean2.isnull().sum().sum(), df_clean3.isnull().sum().sum()],
                    color=['#2E86AB', '#A23B72', '#F18F01', '#C73D23'])
        axes[0,1].set_title('Пропущенные значения')
        axes[0,1].set_ylabel('Количество')

        # Сравнение дубликатов
        axes[1,0].bar(['Исходные', 'Мягкая', 'Агрессивная', 'Гибридная'],
                    [df.duplicated().sum(), df_clean1.duplicated().sum(), 
                    df_clean2.duplicated().sum(), df_clean3.duplicated().sum()],
                    color=['#2E86AB', '#A23B72', '#F18F01', '#C73D23'])
        axes[1,0].set_title('Дубликаты')
        axes[1,0].set_ylabel('Количество')

        # Сравнение средней длины текста
        avg_lengths = [
            df['content'].astype(str).str.len().mean(),
            df_clean1['content'].astype(str).str.len().mean(),
            df_clean2['content'].astype(str).str.len().mean(),
            df_clean3['content'].astype(str).str.len().mean()
        ]
        axes[1,1].bar(['Исходные', 'Мягкая', 'Агрессивная', 'Гибридная'], avg_lengths,
                    color=['#2E86AB', '#A23B72', '#F18F01', '#C73D23'])
        axes[1,1].set_title('Средняя длина текста')
        axes[1,1].set_ylabel('Символы')

        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*80)
        print("БОНУС: LLM-СОВЕТНИК ДЛЯ ТЕКСТОВЫХ ДАННЫХ")
        print("="*80)

        # Проверяем наличие API ключа
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            print("✅ API ключ найден, подключаю LLM-советника...")
            
            try:
                # Создаем агента с LLM
                agent_with_llm = DataQualityAgent(use_llm=True, api_key=api_key, model="gpt-3.5-turbo")
                
                # Описание NLP-задачи
                task_description = """
                Задача: Классификация текстов по категориям (positive, negative, neutral).
                
                Особенности:
                - Сильный дисбаланс классов: neutral составляет ~70-80% данных
                - Важно сохранить редкие классы (positive, negative) для качественного обучения
                - Тексты могут содержать URL, email, специальные символы, избыточные пробелы
                - Есть пропуски (5%) и дубликаты (около 5% от данных)
                - Некоторые тексты аномально короткие (<10 символов) или длинные (>500 символов)
                
                Требования:
                - Максимально сохранить данные для редких классов
                - Улучшить качество признаков для NLP-модели (BERT/RoBERTa)
                - Добавить метаинформацию о качестве текста как дополнительные признаки
                - Сохранить важную информацию (эмодзи для тональности, если есть)
                
                Бизнес-метрики: F1-score на редких классах (positive, negative)
                """.replace("\n", " ").replace("\t", " ")
                
                # Получаем конкретную стратегию
                print("\n📋 Получаю конкретную стратегию от LLM...")
                strategy = agent_with_llm.recommend_strategy_from_llm(task_description)
                print(f"\nРекомендованная стратегия:")
                for key, value in strategy.items():
                    print(f"  • {key}: {value}")
                
                # Применяем рекомендованную стратегию
                print("\n🔧 Применяю рекомендованную стратегию...")
                df_clean_llm = agent_with_llm.fix(df, strategy, text_cols=['content'])
                comparison_llm = agent_with_llm.compare(df, df_clean_llm)
                print("\n📊 Сравнение результатов:")
                print(comparison_llm)
                
            except Exception as e:
                print(f"⚠️ Ошибка при работе с LLM: {e}")
                print("Продолжаю без LLM-советника...")
        else:
            print("⚠️ API ключ OpenAI не найден. LLM-советник недоступен.")
            print("Для использования LLM установите переменную окружения OPENAI_API_KEY")

        # ============================================
        # ИТОГОВЫЙ ОТЧЕТ
        # ============================================

        print("\n" + "="*80)
        print("ИТОГОВЫЙ ОТЧЕТ ПО ОЧИСТКЕ ТЕКСТОВЫХ ДАННЫХ")
        print("="*80)

        # Сравниваем качество данных после разных стратегий
        print("\n📊 Сводка по стратегиям:")

        strategies_data = {
            'Исходные данные': df,
            'Мягкая очистка': df_clean1,
            'Агрессивная очистка': df_clean2,
            'Гибридная очистка': df_clean3
        }

        summary_stats = []
        for name, data in strategies_data.items():
            stats = {
                'Стратегия': name,
                'Строки': len(data),
                'Уникальные строки': data.drop_duplicates().shape[0],
                'Пропуски': data.isnull().sum().sum(),
                'Дубликаты': data.duplicated().sum(),
                'Ср. длина текста': round(data['content'].astype(str).str.len().mean(), 2),
                'Память (MB)': round(data.memory_usage(deep=True).sum() / 1024**2, 2)
            }
            summary_stats.append(stats)

        summary_df = pd.DataFrame(summary_stats)
        print(summary_df.to_string(index=False))

        # Сохраняем очищенные данные
        print("\n💾 Сохраняем очищенные данные...")
        df_clean3.to_csv("data/clean/text_data_cleaned.csv", index=False)

        print("✅ Очищенные данные сохранены в:")
        print("   - data/clean/text_data_cleaned.csv")

        print("\n" + "="*80)
        print("✅ ВСЕ ЧАСТИ ЗАДАНИЯ УСПЕШНО ВЫПОЛНЕНЫ!")
        print("="*80)

        # Дополнительная информация о созданных флагах
        if 'text_anomaly_flag' in df_clean3.columns:
            print(f"\n📌 Создан новый признак: text_anomaly_flag")
            print(f"   • Количество аномалий: {df_clean3['text_anomaly_flag'].sum()}")


if __name__ == "__main__":
    agent = DataQualityAgent(use_llm=True)
    agent.run()
