# data_quality_agent.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Для бонусной части с LLM (теперь OpenAI)
import os
from openai import OpenAI

@dataclass
class QualityReport:
    """Датакласс для отчета о качестве данных"""
    missing: Dict[str, Dict]
    duplicates: Dict[str, Union[int, float]]
    outliers: Dict[str, Dict]
    imbalance: Optional[Dict] = None
    
    def to_dict(self):
        return asdict(self)
    
    def __str__(self):
        """Красивое строковое представление отчета"""
        lines = ["📊 Отчет о качестве данных:"]
        
        if self.missing:
            lines.append("\n  Пропущенные значения:")
            for col, stats in self.missing.items():
                lines.append(f"    • {col}: {stats['count']} ({stats['percentage']}%)")
        
        lines.append(f"\n  Дубликаты: {self.duplicates['count']} ({self.duplicates['percentage']}%)")
        
        if self.outliers:
            lines.append("\n  Выбросы:")
            for col, stats in self.outliers.items():
                lines.append(f"    • {col}: {stats['count']} ({stats['percentage']}%)")
        
        if self.imbalance:
            lines.append(f"\n  Дисбаланс классов: {'ДА' if self.imbalance['is_imbalanced'] else 'НЕТ'}")
            lines.append(f"    • Доминирующий класс: {self.imbalance['dominant_class']} ({self.imbalance['dominant_percentage']}%)")
        
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
    Агент для автоматического выявления и устранения проблем качества данных.
    Использует навыки (skills) для детекции, исправления и сравнения данных.
    """
    
    def __init__(self, random_state: int = 42, use_llm: bool = False, api_key: str = None, model: str = "openai/gpt-4o-mini"):
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
            self.llm_client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            print("✅ OpenAI клиент инициализирован")
        elif use_llm:
            print("⚠️  LLM отключен: не предоставлен API ключ. Используйте переменную окружения OPENAI_API_KEY")
            # Пробуем взять из переменной окружения
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = OpenAI(
                    api_key=api_key
                )
                self.use_llm = True
                print("✅ OpenAI клиент инициализирован из переменной окружения")
            else:
                self.use_llm = False
    
    def detect_issues(self, df: pd.DataFrame, target_col: Optional[str] = None) -> QualityReport:
        """
        Skill 1: Детектирует проблемы качества данных.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный датафрейм
        target_col : str, optional
            Название целевой колонки для анализа дисбаланса классов
            
        Returns:
        --------
        QualityReport : Отчет о найденных проблемах
        """
        print("🔍 Детектив данных начинает расследование...")
        
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
        
        # 3. Анализ выбросов (метод IQR)
        print("  📈 Выявляю выбросы...")
        outliers_stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # IQR метод
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(df)) * 100
            
            if outlier_count > 0:
                outliers_stats[col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_pct, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'min': round(df[col].min(), 2),
                    'max': round(df[col].max(), 2)
                }
        
        # 4. Анализ дисбаланса классов (если указана целевая колонка)
        imbalance_stats = None
        if target_col and target_col in df.columns:
            print("  ⚖️  Проверяю дисбаланс классов...")
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                value_counts = df[target_col].value_counts()
                value_pcts = (value_counts / len(df)) * 100
                
                # Определяем дисбаланс (если какой-то класс < 10% или > 90%)
                is_imbalanced = any(pct < 10 or pct > 90 for pct in value_pcts)
                
                imbalance_stats = {
                    'is_imbalanced': is_imbalanced,
                    'class_counts': value_counts.to_dict(),
                    'class_percentages': value_pcts.to_dict(),
                    'dominant_class': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'dominant_percentage': round(value_pcts.iloc[0], 2) if len(value_counts) > 0 else 0,
                    'class_count': len(value_counts)
                }
        
        # Создаем отчет
        self.quality_report = QualityReport(
            missing=missing_stats,
            duplicates=duplicates_stats,
            outliers=outliers_stats,
            imbalance=imbalance_stats
        )
        
        print("✅ Расследование завершено! Проблемы найдены.")
        print(self.quality_report)
        return self.quality_report
    
    def fix(self, df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        """
        Skill 2: Исправляет проблемы качества данных согласно стратегии.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный датафрейм
        strategy : Dict
            Стратегия очистки для каждого типа проблем.
            Пример: {
                'missing': 'median' | 'mean' | 'mode' | 'drop',
                'duplicates': 'drop' | 'keep_first',
                'outliers': 'clip_iqr' | 'remove' | 'winsorize' | 'zscore'
            }
            
        Returns:
        --------
        pd.DataFrame : Очищенный датафрейм
        """
        print("🩺 Хирург данных приступает к операции...")
        df_clean = df.copy()
        
        # Сохраняем статистику до очистки
        before_stats = self._get_quality_stats(df_clean)
        
        # 1. Обработка пропущенных значений
        if 'missing' in strategy:
            missing_strategy = strategy['missing']
            print(f"  🧹 Обрабатываю пропуски методом: {missing_strategy}")
            
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if missing_strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif missing_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif missing_strategy == 'mode':
                        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                    elif missing_strategy == 'drop':
                        df_clean.dropna(subset=[col], inplace=True)
                    elif missing_strategy == 'forward_fill':
                        df_clean[col].fillna(method='ffill', inplace=True)
                    elif missing_strategy == 'backward_fill':
                        df_clean[col].fillna(method='bfill', inplace=True)
        
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
        
        # 3. Обработка выбросов
        if 'outliers' in strategy:
            outlier_strategy = strategy['outliers']
            print(f"  📊 Обрабатываю выбросы методом: {outlier_strategy}")
            
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if outlier_strategy == 'clip_iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    
                elif outlier_strategy == 'winsorize':
                    # Винзоризация (замена на граничные значения)
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    
                elif outlier_strategy == 'remove':
                    # Удаление строк с выбросами
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                    df_clean = df_clean[mask]
                    
                elif outlier_strategy == 'zscore':
                    # Метод Z-оценки
                    z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                    threshold = 3
                    outliers_mask = pd.Series(False, index=df_clean.index)
                    outliers_mask[df_clean[col].dropna().index[z_scores > threshold]] = True
                    df_clean = df_clean[~outliers_mask]
        
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
        
        # Добавляем информацию о типах данных
        dtypes_count = df.dtypes.value_counts().to_dict()
        stats['dtypes'] = {str(k): int(v) for k, v in dtypes_count.items()}
        
        return stats
    
    def _generate_comparison_summary(self, before: Dict, after: Dict, changes: Dict) -> str:
        """Генерирует текстовое резюме сравнения"""
        summary = f"\n📈 Итоги очистки данных:\n"
        summary += f"  • Строк: {before['rows']} → {after['rows']} ({changes.get('rows', 0):+d})\n"
        summary += f"  • Пропусков: {before['missing_total']} → {after['missing_total']} ({changes.get('missing_total', 0):+d})\n"
        summary += f"  • Пропуски %: {before['missing_pct']}% → {after['missing_pct']}%\n"
        summary += f"  • Дубликатов: {before['duplicates']} → {after['duplicates']} ({changes.get('duplicates', 0):+d})\n"
        summary += f"  • Память: {before['memory_mb']} MB → {after['memory_mb']} MB"
        
        return summary
    
    def visualize_issues(self, df: pd.DataFrame, report: QualityReport):
        """
        Визуализирует найденные проблемы качества данных.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный датафрейм
        report : QualityReport
            Отчет о качестве данных
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Анализ качества данных', fontsize=16)
        
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
            # Делаем текст более читаемым
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax2.set_title(f'Дубликаты: {report.duplicates["count"]} шт.')
        else:
            ax2.text(0.5, 0.5, 'Нет дубликатов', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Дубликаты')
        
        # 3. Выбросы
        ax3 = axes[1, 0]
        if report.outliers:
            outliers_df = pd.DataFrame(report.outliers).T
            outliers_df['percentage'].plot(kind='bar', ax=ax3, color='orange', edgecolor='black')
            ax3.set_title('Выбросы по колонкам (%)')
            ax3.set_ylabel('Процент выбросов')
            ax3.tick_params(axis='x', rotation=45)
            ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Порог 5%')
            ax3.legend()
            
            # Добавляем значения на столбцы
            for i, v in enumerate(outliers_df['percentage']):
                ax3.text(i, v + 0.5, str(round(v, 1)), ha='center', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Нет выбросов', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Выбросы')
        
        # 4. Дисбаланс классов (если есть)
        ax4 = axes[1, 1]
        if report.imbalance:
            class_pcts = report.imbalance['class_percentages']
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_pcts)))
            bars = ax4.bar(range(len(class_pcts)), list(class_pcts.values()), color=colors, edgecolor='black')
            ax4.set_title('Распределение классов')
            ax4.set_ylabel('Процент')
            ax4.set_xticks(range(len(class_pcts)))
            ax4.set_xticklabels([str(k)[:10] for k in class_pcts.keys()], rotation=45, ha='right')
            
            # Добавляем линии порогов
            ax4.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Порог 10% (миноритарный)')
            ax4.axhline(y=90, color='darkred', linestyle='--', alpha=0.7, label='Порог 90% (мажоритарный)')
            
            # Добавляем значения
            for i, (bar, (_, pct)) in enumerate(zip(bars, class_pcts.items())):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Подсвечиваем, если есть дисбаланс
            if report.imbalance['is_imbalanced']:
                ax4.set_facecolor('#fff3f3')
                ax4.text(0.5, 0.95, '⚠️ Обнаружен дисбаланс классов', 
                        transform=ax4.transAxes, ha='center', color='red', fontweight='bold')
            
            ax4.legend(loc='upper right')
        else:
            ax4.text(0.5, 0.5, 'Нет данных для анализа дисбаланса', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Дисбаланс классов')
        
        plt.tight_layout()
        plt.show()
    
    # Бонусная часть: LLM-советник (теперь с OpenAI)
    def get_llm_recommendation(self, task_description: str) -> str:
        """
        Использует OpenAI API для получения рекомендаций по стратегии очистки.
        
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
            system_prompt = """Ты - эксперт по качеству данных (Data Quality Expert). 
Твоя задача - анализировать отчеты о качестве данных и давать рекомендации по их очистке.
Ты должен учитывать специфику ML-задачи и предлагать оптимальные стратегии."""

            user_prompt = f"""Проанализируй отчет о качестве данных и описание ML-задачи, затем порекомендуй оптимальную стратегию очистки.

ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ:
{self.quality_report}

ОПИСАНИЕ ML-ЗАДАЧИ:
{task_description}

ДОСТУПНЫЕ СТРАТЕГИИ:
- missing: 'median' (медиана), 'mean' (среднее), 'mode' (мода), 'drop' (удалить строки), 'forward_fill' (заполнение предыдущим), 'backward_fill' (заполнение следующим)
- duplicates: 'drop' (удалить все дубликаты), 'keep_first' (оставить первый), 'keep_last' (оставить последний)
- outliers: 'clip_iqr' (ограничить по IQR), 'remove' (удалить), 'winsorize' (винзоризация), 'zscore' (удалить по Z-оценке)

ТВОЯ ЗАДАЧА:
1. Дай конкретные рекомендации по каждой проблеме (пропуски, дубликаты, выбросы, дисбаланс)
2. Объясни, почему эта стратегия лучше всего подходит для данной задачи
3. Укажи возможные риски выбранного подхода
4. Предложи альтернативный вариант на случай, если основной не сработает

Формат ответа: 
- Будь конкретным и практичным
- Используй маркированные списки для читаемости
- Ограничь ответ 400 словами"""

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
            print("🤖 РЕКОМЕНДАЦИЯ LLM:")
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
                'missing': 'median',
                'duplicates': 'drop',
                'outliers': 'clip_iqr'
            }
        
        try:
            prompt = f"""На основе отчета о качестве данных и описания задачи, 
            верни ТОЛЬКО JSON со стратегией очистки в формате:
            {{"missing": "стратегия", "duplicates": "стратегия", "outliers": "стратегия"}}

            Отчет: {self.quality_report}
            Задача: {task_description}
            
            Доступные стратегии:
            - missing: median, mean, mode, drop, forward_fill, backward_fill
            - duplicates: drop, keep_first, keep_last
            - outliers: clip_iqr, remove, winsorize, zscore
            
            Верни только JSON, без пояснений."""
            
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            import json
            strategy = json.loads(response.choices[0].message.content)
            return strategy
            
        except:
            # Возвращаем стратегию по умолчанию в случае ошибки
            return {
                'missing': 'median',
                'duplicates': 'drop',
                'outliers': 'clip_iqr'
            }


# Функция для создания тестового датасета с проблемами
def create_test_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Создает тестовый датасет с типичными проблемами качества"""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'age': np.random.normal(40, 15, n_samples).astype(int),
        'income': np.random.exponential(50000, n_samples).astype(int),
        'education_years': np.random.randint(8, 22, n_samples),
        'satisfaction_score': np.random.uniform(1, 10, n_samples),
        'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # Сильный дисбаланс
    })
    
    # Добавляем пропуски
    missing_mask = np.random.random(n_samples) < 0.1
    df.loc[missing_mask, 'income'] = np.nan
    
    missing_mask = np.random.random(n_samples) < 0.05
    df.loc[missing_mask, 'satisfaction_score'] = np.nan
    
    # Добавляем дубликаты
    duplicate_indices = np.random.choice(n_samples, size=20, replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Добавляем выбросы
    outlier_indices = np.random.choice(len(df), size=15, replace=False)
    df.loc[outlier_indices, 'income'] = df.loc[outlier_indices, 'income'] * 10
    df.loc[outlier_indices, 'age'] = df.loc[outlier_indices, 'age'] + 100
    
    return df