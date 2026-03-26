# al_agent_legal.py - Active Learning Agent для юридических документов
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import glob
from datetime import datetime
import json

warnings.filterwarnings('ignore')

@dataclass
class TrainingMetrics:
    """Метрики обучения"""
    iteration: int
    n_labeled: int
    accuracy: float
    f1: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class ActiveLearningAgent:
    """
    Active Learning Agent для классификации юридических документов (кодексов РФ).
    Выбирает наиболее информативные примеры для разметки.
    """
    
    def __init__(self, model_type: str = 'logreg', 
                 vectorizer: str = 'tfidf',
                 random_state: int = 42,
                 **kwargs):
        """
        Инициализация агента активного обучения.
        
        Parameters:
        -----------
        model_type : str
            Тип модели ('logreg', 'svm', 'random_forest')
        vectorizer : str
            Тип векторизатора ('tfidf', 'count')
        random_state : int
            Seed для воспроизводимости
        **kwargs : dict
            Дополнительные параметры для модели
        """
        self.model_type = model_type
        self.vectorizer_type = vectorizer
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = None
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.training_history = []
        self.feature_names = None
        
        self._init_vectorizer()
    
    def _init_vectorizer(self):
        """Инициализирует векторизатор текста с учетом юридической лексики"""
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 3),  # Увеличил до 3-грамм для юридических терминов
                stop_words=None,  # Не удаляем стоп-слова, т.к. в юр. текстах они важны
                token_pattern=r'(?u)\b\w[\w-]+\b'  # Учитываем дефисы в терминах
            )
        elif self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 3),
                stop_words=None,
                token_pattern=r'(?u)\b\w[\w-]+\b'
            )
        else:
            self.vectorizer = None
    
    def _init_model(self, n_classes: int):
        """Инициализирует модель классификации"""
        # Базовые параметры модели
        base_params = {
            'max_iter': 1000,
            'random_state': self.random_state,
            'class_weight': 'balanced'
        }
        
        # Добавляем специфичные параметры в зависимости от типа модели
        if self.model_type == 'logreg':
            # Параметры по умолчанию для логистической регрессии
            default_params = {'C': 1.0}
            # Объединяем с пользовательскими параметрами, пользовательские имеют приоритет
            model_params = {**base_params, **default_params, **self.kwargs}
            self.model = LogisticRegression(**model_params)
            
        elif self.model_type == 'svm':
            default_params = {'kernel': 'linear', 'C': 1.0}
            model_params = {**base_params, **default_params, **self.kwargs}
            self.model = SVC(probability=True, **model_params)
            
        elif self.model_type == 'random_forest':
            default_params = {'n_estimators': 100, 'max_depth': 10}
            model_params = {**base_params, **default_params, **self.kwargs}
            self.model = RandomForestClassifier(**model_params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def load_data_from_folder(self, folder_path: str = "data/raw", 
                            file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        Загружает CSV файлы с юридическими документами.
        
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
        
        print(f"\n✅ Всего загружено: {len(combined_df)} записей, {len(combined_df.columns)} колонок")
        print(f"📋 Доступные колонки: {list(combined_df.columns)}")
        
        # Проверяем наличие колонки content
        if 'content' not in combined_df.columns:
            print("⚠️ Внимание: колонка 'content' не найдена в данных")
            # Ищем альтернативные названия
            alt_cols = ['text', 'document', 'article', 'content_text']
            for col in alt_cols:
                if col in combined_df.columns:
                    combined_df = combined_df.rename(columns={col: 'content'})
                    print(f"  • Переименована колонка '{col}' в 'content'")
                    break
        
        return combined_df
    
    def fit(self, labeled_df: pd.DataFrame, 
            text_column: str = 'content',
            label_column: str = 'label') -> 'ActiveLearningAgent':
        """
        Skill 1: Обучает базовую модель на размеченных юридических документах.
        
        Parameters:
        -----------
        labeled_df : pd.DataFrame
            Датафрейм с размеченными данными
        text_column : str
            Колонка с текстом документа
        label_column : str
            Колонка с метками (тип кодекса)
            
        Returns:
        --------
        self : ActiveLearningAgent
        """
        print(f"\n📚 Обучение модели на {len(labeled_df)} размеченных документах...")
        
        # Извлекаем тексты и метки
        X_text = labeled_df[text_column].fillna('').astype(str).values
        y = labeled_df[label_column].values
        
        # Кодируем метки
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        print(f"  • Классы: {list(self.classes_)}")
        
        # Векторизуем текст
        if self.vectorizer is not None:
            X = self.vectorizer.fit_transform(X_text)
            self.feature_names = self.vectorizer.get_feature_names_out()
            print(f"  • Размерность признаков: {X.shape[1]}")
            
            # Показываем топ-10 важных слов для каждого класса (для LogReg)
            if hasattr(self, 'model') and self.model is not None:
                self._show_top_features()
        
        # Инициализируем и обучаем модель
        self._init_model(len(self.classes_))
        self.model.fit(X, y_encoded)
        
        # Вычисляем метрики на обучающих данных
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_encoded, y_pred)
        f1 = f1_score(y_encoded, y_pred, average='weighted')
        
        print(f"\n✅ Обучение завершено")
        print(f"  • Accuracy: {accuracy:.4f}")
        print(f"  • F1-score: {f1:.4f}")
        
        return self
    
    def _show_top_features(self):
        """Показывает топ-10 важных признаков для каждого класса"""
        if hasattr(self.model, 'coef_'):
            print(f"\n  • Топ-5 важных признаков для каждого класса:")
            for i, class_name in enumerate(self.classes_):
                coef = self.model.coef_[i]
                top_indices = np.argsort(coef)[-5:][::-1]
                top_words = [self.feature_names[idx] for idx in top_indices if idx < len(self.feature_names)]
                print(f"    {class_name}: {', '.join(top_words)}")
    
    def _predict_proba(self, X_text: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности для текстов.
        
        Parameters:
        -----------
        X_text : np.ndarray
            Массив текстов документов
            
        Returns:
        --------
        np.ndarray : Массив вероятностей
        """
        if self.vectorizer is not None:
            X = self.vectorizer.transform(X_text)
        else:
            X = X_text.reshape(-1, 1)
        
        return self.model.predict_proba(X)
    
    def query(self, pool_df: pd.DataFrame, 
              strategy: str = 'entropy',
              batch_size: int = 20,
              text_column: str = 'content') -> List[int]:
        """
        Skill 2: Выбирает наиболее информативные юридические документы из пула.
        
        Parameters:
        -----------
        pool_df : pd.DataFrame
            Датафрейм с неразмеченными документами
        strategy : str
            Стратегия отбора: 'entropy', 'margin', 'confidence', 'random'
        batch_size : int
            Количество документов для отбора
        text_column : str
            Колонка с текстом документа
            
        Returns:
        --------
        List[int] : Индексы выбранных документов
        """
        if len(pool_df) == 0:
            print("⚠️ Пул данных пуст")
            return []
        
        print(f"\n🔍 Отбор документов по стратегии: {strategy.upper()} (batch_size={batch_size})")
        
        # Извлекаем тексты
        X_text = pool_df[text_column].fillna('').astype(str).values
        
        # Получаем вероятности
        proba = self._predict_proba(X_text)
        
        # Вычисляем scores в зависимости от стратегии
        if strategy == 'entropy':
            # Энтропия: чем выше, тем неопределеннее
            scores = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            
        elif strategy == 'margin':
            # Margin: разница между двумя самыми вероятными классами
            sorted_proba = np.sort(proba, axis=1)
            scores = -(sorted_proba[:, -1] - sorted_proba[:, -2])
            
        elif strategy == 'confidence':
            # Confidence: уверенность модели
            scores = 1 - np.max(proba, axis=1)
            
        elif strategy == 'random':
            # Случайный отбор
            scores = np.random.random(len(pool_df))
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Выбираем top-k примеров
        n_select = min(batch_size, len(pool_df))
        selected_indices = np.argsort(scores)[-n_select:][::-1]
        
        # Возвращаем индексы из датафрейма
        selected_ids = pool_df.index[selected_indices].tolist()
        
        print(f"✅ Выбрано {len(selected_ids)} документов для разметки")
        print(f"  • Средний score: {scores[selected_indices].mean():.4f}")
        print(f"  • Диапазон scores: [{scores[selected_indices].min():.4f}, {scores[selected_indices].max():.4f}]")
        
        # Показываем примеры выбранных документов
        print(f"\n📝 Топ-3 выбранных документа:")
        for i, idx in enumerate(selected_ids[:3]):
            content_preview = str(pool_df.loc[idx, text_column])[:150]
            score = scores[selected_indices[i]]
            print(f"\n  {i+1}. Score={score:.4f}")
            print(f"     {content_preview}...")
        
        return selected_ids
    
    def evaluate(self, labeled_df: pd.DataFrame, 
                 test_df: pd.DataFrame,
                 text_column: str = 'content',
                 label_column: str = 'label') -> Dict[str, float]:
        """
        Skill 3: Оценивает качество модели на тестовых документах.
        
        Parameters:
        -----------
        labeled_df : pd.DataFrame
            Датафрейм с размеченными документами (используется для переобучения)
        test_df : pd.DataFrame
            Датафрейм с тестовыми документами
        text_column : str
            Колонка с текстом
        label_column : str
            Колонка с метками
            
        Returns:
        --------
        Dict[str, float] : Метрики качества
        """
        # Переобучаем модель на всех размеченных данных
        self.fit(labeled_df, text_column, label_column)
        
        # Оцениваем на тесте
        X_test = test_df[text_column].fillna('').astype(str).values
        y_test = test_df[label_column].values
        
        # Векторизуем
        if self.vectorizer is not None:
            X_test_vec = self.vectorizer.transform(X_test)
        else:
            X_test_vec = X_test.reshape(-1, 1)
        
        # Предсказываем
        y_pred_encoded = self.model.predict(X_test_vec)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Вычисляем метрики
        y_test_encoded = self.label_encoder.transform(y_test)
        
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
        
        # Дополнительные метрики
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall']
        }
        
        print(f"\n📊 Оценка модели на тестовых документах:")
        print(f"  • Accuracy: {accuracy:.4f}")
        print(f"  • F1-score: {f1:.4f}")
        print(f"  • Precision: {metrics['precision']:.4f}")
        print(f"  • Recall: {metrics['recall']:.4f}")
        
        # Показываем confusion matrix
        self._plot_confusion_matrix(y_test, y_pred)
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Визуализирует матрицу ошибок"""
        cm = confusion_matrix(y_true, y_pred, labels=self.classes_)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes_, yticklabels=self.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100)
        plt.show()
    
    def run_cycle(self, 
                  labeled_df: pd.DataFrame,
                  pool_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  strategy: str = 'entropy',
                  n_iterations: int = 5,
                  batch_size: int = 20,
                  text_column: str = 'content',
                  label_column: str = 'label') -> List[TrainingMetrics]:
        """
        Запускает полный цикл активного обучения для юридических документов.
        
        Parameters:
        -----------
        labeled_df : pd.DataFrame
            Исходные размеченные документы
        pool_df : pd.DataFrame
            Пул неразмеченных документов
        test_df : pd.DataFrame
            Тестовые документы для оценки
        strategy : str
            Стратегия отбора
        n_iterations : int
            Количество итераций
        batch_size : int
            Количество документов на итерацию
        text_column : str
            Колонка с текстом
        label_column : str
            Колонка с метками
            
        Returns:
        --------
        List[TrainingMetrics] : История обучения
        """
        print("\n" + "="*80)
        print(f"🚀 ЗАПУСК ЦИКЛА АКТИВНОГО ОБУЧЕНИЯ ДЛЯ ЮРИДИЧЕСКИХ ДОКУМЕНТОВ")
        print(f"   Стратегия: {strategy.upper()}")
        print(f"   Итераций: {n_iterations}")
        print(f"   Batch size: {batch_size}")
        print("="*80)
        
        history = []
        current_labeled = labeled_df.copy()
        current_pool = pool_df.copy()
        
        for iteration in range(1, n_iterations + 1):
            print(f"\n{'='*60}")
            print(f"📊 ИТЕРАЦИЯ {iteration}/{n_iterations}")
            print(f"{'='*60}")
            
            print(f"\n📚 Размеченных документов: {len(current_labeled)}")
            print(f"📦 Неразмеченных документов: {len(current_pool)}")
            
            # Обучаем модель на текущих размеченных данных
            self.fit(current_labeled, text_column, label_column)
            
            # Оцениваем на тесте
            metrics = self.evaluate(current_labeled, test_df, text_column, label_column)
            
            # Сохраняем метрики
            training_metrics = TrainingMetrics(
                iteration=iteration,
                n_labeled=len(current_labeled),
                accuracy=metrics['accuracy'],
                f1=metrics['f1'],
                precision=metrics.get('precision'),
                recall=metrics.get('recall')
            )
            history.append(training_metrics)
            
            # Если это последняя итерация, не отбираем новые примеры
            if iteration == n_iterations:
                break
            
            # Отбираем новые документы для разметки
            selected_indices = self.query(
                current_pool, 
                strategy=strategy,
                batch_size=batch_size,
                text_column=text_column
            )
            
            if not selected_indices:
                print("⚠️ Нет больше документов для отбора")
                break
            
            # Добавляем выбранные документы в размеченный датасет
            # В реальном сценарии здесь нужна ручная разметка
            # В нашем случае используем существующие метки из pool_df
            new_samples = current_pool.loc[selected_indices].copy()
            
            # Обновляем датасеты
            current_labeled = pd.concat([current_labeled, new_samples], ignore_index=True)
            current_pool = current_pool.drop(selected_indices)
            
            print(f"\n✅ Добавлено {len(selected_indices)} новых документов")
            print(f"📊 Текущий размер размеченного датасета: {len(current_labeled)}")
            
            # Показываем распределение классов после добавления
            if 'label' in current_labeled.columns:
                print(f"\n📊 Распределение классов:")
                for label, count in current_labeled['label'].value_counts().items():
                    print(f"  • {label}: {count} ({count/len(current_labeled)*100:.1f}%)")
        
        self.training_history = history
        return history
    
    def report(self, history: List[TrainingMetrics], 
               save_path: str = "learning_curve.png",
               compare_with: Optional[List[TrainingMetrics]] = None,
               baseline_name: str = "random") -> None:
        """
        Skill 4: Визуализирует кривую обучения для юридических документов.
        
        Parameters:
        -----------
        history : List[TrainingMetrics]
            История обучения
        save_path : str
            Путь для сохранения графика
        compare_with : List[TrainingMetrics], optional
            История для сравнения
        baseline_name : str
            Название baseline стратегии
        """
        print("\n📈 Генерация кривой обучения...")
        
        # Создаем датафрейм из истории
        df_history = pd.DataFrame([vars(m) for m in history])
        
        # Настройка графиков
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График 1: Accuracy
        ax1 = axes[0]
        ax1.plot(df_history['n_labeled'], df_history['accuracy'], 
                'bo-', linewidth=2, markersize=8, label='Active Learning (Entropy)')
        
        if compare_with:
            df_compare = pd.DataFrame([vars(m) for m in compare_with])
            ax1.plot(df_compare['n_labeled'], df_compare['accuracy'], 
                    'rs--', linewidth=2, markersize=8, label=f'Baseline ({baseline_name})')
        
        ax1.set_xlabel('Количество размеченных документов', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Кривая обучения (Accuracy) - Юридические документы', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # График 2: F1-score
        ax2 = axes[1]
        ax2.plot(df_history['n_labeled'], df_history['f1'], 
                'go-', linewidth=2, markersize=8, label='Active Learning (Entropy)')
        
        if compare_with:
            ax2.plot(df_compare['n_labeled'], df_compare['f1'], 
                    'rs--', linewidth=2, markersize=8, label=f'Baseline ({baseline_name})')
        
        ax2.set_xlabel('Количество размеченных документов', fontsize=12)
        ax2.set_ylabel('F1-score', fontsize=12)
        ax2.set_title('Кривая обучения (F1-score) - Юридические документы', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ График сохранен в {save_path}")
        
        # Выводим статистику
        print("\n📊 СТАТИСТИКА ОБУЧЕНИЯ:")
        print(f"  • Начальная точность (N={df_history.iloc[0]['n_labeled']}): {df_history.iloc[0]['accuracy']:.4f}")
        print(f"  • Финальная точность (N={df_history.iloc[-1]['n_labeled']}): {df_history.iloc[-1]['accuracy']:.4f}")
        print(f"  • Улучшение: +{df_history.iloc[-1]['accuracy'] - df_history.iloc[0]['accuracy']:.4f}")
        
        if compare_with:
            final_al = df_history.iloc[-1]['accuracy']
            final_baseline = df_compare.iloc[-1]['accuracy']
            improvement = final_al - final_baseline
            print(f"\n  • Сравнение с {baseline_name} стратегией:")
            print(f"    - Active Learning: {final_al:.4f}")
            print(f"    - Baseline: {final_baseline:.4f}")
            print(f"    - Преимущество: +{improvement:.4f}")
            
            # Подсчет сэкономленных примеров
            target_acc = final_al
            baseline_accs = df_compare['accuracy'].values
            baseline_n = df_compare['n_labeled'].values
            
            for i in range(len(baseline_accs)):
                if baseline_accs[i] >= target_acc:
                    saved = baseline_n[i] - df_history.iloc[-1]['n_labeled']
                    print(f"\n  • Сэкономлено документов: ~{saved} ({(saved/baseline_n[i])*100:.1f}%)")
                    break
    
    def compare_strategies(self, 
                          labeled_df: pd.DataFrame,
                          pool_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          strategies: List[str] = ['entropy', 'margin', 'random'],
                          n_iterations: int = 5,
                          batch_size: int = 20,
                          text_column: str = 'content',
                          label_column: str = 'label') -> Dict[str, List[TrainingMetrics]]:
        """
        Сравнивает различные стратегии активного обучения.
        """
        print("\n" + "="*80)
        print("🔬 СРАВНЕНИЕ СТРАТЕГИЙ АКТИВНОГО ОБУЧЕНИЯ")
        print("="*80)
        
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"📊 ТЕСТИРОВАНИЕ СТРАТЕГИИ: {strategy.upper()}")
            print(f"{'='*60}")
            
            # Создаем копии данных для каждой стратегии
            current_labeled = labeled_df.copy()
            current_pool = pool_df.copy()
            
            history = []
            
            for iteration in range(1, n_iterations + 1):
                # Обучаем модель
                self.fit(current_labeled, text_column, label_column)
                
                # Оцениваем
                metrics = self.evaluate(current_labeled, test_df, text_column, label_column)
                
                history.append(TrainingMetrics(
                    iteration=iteration,
                    n_labeled=len(current_labeled),
                    accuracy=metrics['accuracy'],
                    f1=metrics['f1']
                ))
                
                if iteration == n_iterations:
                    break
                
                # Отбираем документы
                selected_indices = self.query(
                    current_pool,
                    strategy=strategy,
                    batch_size=batch_size,
                    text_column=text_column
                )
                
                if not selected_indices:
                    break
                
                # Добавляем в размеченный датасет
                new_samples = current_pool.loc[selected_indices].copy()
                current_labeled = pd.concat([current_labeled, new_samples], ignore_index=True)
                current_pool = current_pool.drop(selected_indices)
            
            results[strategy] = history
        
        # Визуализируем сравнение
        self._plot_strategy_comparison(results, strategies)
        
        return results
    
    def _plot_strategy_comparison(self, results: Dict[str, List[TrainingMetrics]], 
                                  strategies: List[str]):
        """Визуализирует сравнение стратегий"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = {'entropy': 'blue', 'margin': 'green', 'random': 'red'}
        markers = {'entropy': 'o', 'margin': 's', 'random': '^'}
        
        # График Accuracy
        ax1 = axes[0]
        for strategy in strategies:
            df = pd.DataFrame([vars(m) for m in results[strategy]])
            ax1.plot(df['n_labeled'], df['accuracy'], 
                    marker=markers[strategy], linestyle='-', 
                    linewidth=2, markersize=6,
                    color=colors.get(strategy, 'black'),
                    label=strategy.upper())
        
        ax1.set_xlabel('Количество размеченных документов', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Сравнение стратегий (Accuracy)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # График F1
        ax2 = axes[1]
        for strategy in strategies:
            df = pd.DataFrame([vars(m) for m in results[strategy]])
            ax2.plot(df['n_labeled'], df['f1'], 
                    marker=markers[strategy], linestyle='-', 
                    linewidth=2, markersize=6,
                    color=colors.get(strategy, 'black'),
                    label=strategy.upper())
        
        ax2.set_xlabel('Количество размеченных документов', fontsize=12)
        ax2.set_ylabel('F1-score', fontsize=12)
        ax2.set_title('Сравнение стратегий (F1-score)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("strategy_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Сравнение стратегий сохранено в strategy_comparison.png")
        
        # Выводим статистику сравнения
        print("\n📊 ИТОГИ СРАВНЕНИЯ СТРАТЕГИЙ:")
        final_results = []
        for strategy in strategies:
            last = results[strategy][-1]
            first = results[strategy][0]
            final_results.append({
                'Стратегия': strategy.upper(),
                'Начальная точность': f"{first.accuracy:.4f}",
                'Финальная точность': f"{last.accuracy:.4f}",
                'Улучшение': f"{last.accuracy - first.accuracy:+.4f}",
                'Финальный F1': f"{last.f1:.4f}"
            })
        
        comparison_df = pd.DataFrame(final_results)
        print(comparison_df.to_string(index=False))


# Функция для создания реального датасета из CSV
def load_legal_dataset_from_csv(folder_path: str = "data/raw", 
                                label_column: str = 'auto_label_document_type',
                                text_column: str = 'content') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает реальные юридические документы из CSV файлов и разделяет на labeled/unlabeled/test.
    
    Parameters:
    -----------
    folder_path : str
        Путь к папке с CSV файлами
    label_column : str
        Колонка с метками (тип кодекса)
    text_column : str
        Колонка с текстом документа
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] : (labeled, unlabeled, test)
    """
    print(f"\n📂 Загрузка юридических документов из {folder_path}")
    
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"CSV файлы не найдены в {folder_path}")
    
    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Проверяем наличие необходимых колонок
    if text_column not in combined_df.columns:
        raise ValueError(f"Колонка '{text_column}' не найдена в данных")
    
    if label_column not in combined_df.columns:
        print(f"⚠️ Колонка '{label_column}' не найдена. Создаю неразмеченный датасет...")
        # Если нет меток, все данные считаем неразмеченными
        unlabeled_df = combined_df[[text_column]].copy()
        labeled_df = pd.DataFrame(columns=[text_column, label_column])
        test_df = pd.DataFrame(columns=[text_column, label_column])
    else:
        # Разделяем на labeled, unlabeled и test
        # Берем 70% размеченных для обучения, 30% для теста
        labeled_data = combined_df[combined_df[label_column].notna()]
        unlabeled_data = combined_df[combined_df[label_column].isna()]
        
        if len(labeled_data) > 0:
            # Разделяем размеченные данные на train и test
            from sklearn.model_selection import train_test_split
            labeled_df, test_df = train_test_split(
                labeled_data, 
                test_size=0.3, 
                random_state=42,
                stratify=labeled_data[label_column] if len(labeled_data[label_column].unique()) > 1 else None
            )
        else:
            labeled_df = pd.DataFrame(columns=[text_column, label_column])
            test_df = pd.DataFrame(columns=[text_column, label_column])
        
        unlabeled_df = unlabeled_data[[text_column]].copy() if len(unlabeled_data) > 0 else pd.DataFrame(columns=[text_column])
    
    print(f"\n✅ Данные загружены:")
    print(f"  • Размеченные документы (train): {len(labeled_df)}")
    print(f"  • Тестовые документы: {len(test_df)}")
    print(f"  • Неразмеченные документы (pool): {len(unlabeled_df)}")
    
    if len(labeled_df) > 0 and label_column in labeled_df.columns:
        print(f"\n📊 Распределение классов в размеченных данных:")
        print(labeled_df[label_column].value_counts())
    
    return labeled_df, unlabeled_df, test_df