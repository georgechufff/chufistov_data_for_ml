# annotation_agent.py — AnnotationAgent для автоматической разметки текстовых данных
# Задание 3: auto_label, generate_spec, check_quality, export_to_labelstudio

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import re
import os
import warnings
from datetime import datetime
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# ──────────────────────────── Датаклассы ────────────────────────────

@dataclass
class QualityMetrics:
    """Метрики качества разметки."""
    kappa: Optional[float]
    percent_agreement: Optional[float]
    label_distribution: Dict[str, int]
    confidence_mean: float
    confidence_std: float
    low_confidence_count: int
    low_confidence_threshold: float
    total_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnnotationSpec:
    """Спецификация разметки (Markdown)."""
    task: str
    description: str
    classes: Dict[str, str]           # class_name → definition
    examples: List[Dict[str, str]]    # [{text, label, explanation}, ...]
    edge_cases: List[str]
    guidelines: List[str]

    def to_markdown(self) -> str:
        md = f"# Спецификация разметки\n\n"
        md += f"## Задача: {self.task}\n\n"
        md += f"### Описание\n{self.description}\n\n"

        md += "### Классы разметки\n\n"
        for cls, defn in self.classes.items():
            md += f"#### {cls}\n{defn}\n\n"

        md += "### Примеры\n\n"
        for i, ex in enumerate(self.examples, 1):
            md += f"**Пример {i}**\n"
            md += f"- **Текст**: {ex.get('text', '')}\n"
            md += f"- **Метка**: {ex.get('label', '')}\n"
            if ex.get('explanation'):
                md += f"- **Пояснение**: {ex['explanation']}\n"
            md += "\n"

        md += "### Граничные случаи\n\n"
        for case in self.edge_cases:
            md += f"- {case}\n"

        md += "\n### Рекомендации по разметке\n\n"
        for g in self.guidelines:
            md += f"- {g}\n"

        md += f"\n---\n*Создано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        return md


# ──────────────── Словари для классификации кодексов РФ ─────────────

# Полные названия → краткие коды (как в датасете)
CODEX_FULL_TO_SHORT = {
    "АПК РФ": "АПК РФ",
    "БК РФ": "БК РФ",
    "ГК РФ": "ГК РФ",
    "ГПК РФ": "ГПК РФ",
    "ГрК РФ": "ГрК РФ",
    "ЖК РФ": "ЖК РФ",
    "ЗК РФ": "ЗК РФ",
    "КоАП РФ": "КоАП РФ",
    "ЛК РФ": "ЛК РФ",
    "НК РФ": "НК РФ",
    "СК РФ": "СК РФ",
    "ТК РФ": "ТК РФ",
    "УК РФ": "УК РФ",
    "УПК РФ": "УПК РФ",
    "Отдельный нормативно-правовой акт": "Отдельный НПА",
}

CODEX_DESCRIPTIONS = {
    "АПК РФ": "Арбитражный процессуальный кодекс — регулирует судопроизводство в арбитражных судах, экономические споры юридических лиц",
    "БК РФ": "Бюджетный кодекс — регулирует бюджетные отношения, доходы/расходы бюджета, межбюджетные трансферты",
    "ГК РФ": "Гражданский кодекс — регулирует гражданско-правовые отношения: сделки, договоры, собственность, обязательства",
    "ГПК РФ": "Гражданский процессуальный кодекс — регулирует гражданское судопроизводство, иски, апелляции",
    "ГрК РФ": "Градостроительный кодекс — регулирует планировку территорий, строительство, архитектурную деятельность",
    "ЖК РФ": "Жилищный кодекс — регулирует жилищные отношения, права собственников, ЖКХ, управление домами",
    "ЗК РФ": "Земельный кодекс — регулирует земельные отношения, кадастр, категории земель, аренду участков",
    "КоАП РФ": "Кодекс об административных правонарушениях — административные штрафы, нарушения, ответственность",
    "ЛК РФ": "Лесной кодекс — регулирует лесные отношения, лесопользование, охрану лесов",
    "НК РФ": "Налоговый кодекс — регулирует налоги, сборы, налоговую базу, ставки, порядок уплаты",
    "СК РФ": "Семейный кодекс — регулирует брак, развод, алименты, права детей, усыновление",
    "ТК РФ": "Трудовой кодекс — регулирует трудовые договоры, рабочее время, отпуска, увольнение, охрану труда",
    "УК РФ": "Уголовный кодекс — определяет преступления и наказания, виды ответственности",
    "УПК РФ": "Уголовно-процессуальный кодекс — регулирует уголовное судопроизводство, следствие, дознание",
    "Отдельный нормативно-правовой акт": "Отдельные НПА — постановления Правительства, федеральные законы, приказы министерств и ведомств",
}

# Ключевые маркеры для rule-based классификации
CODEX_KEYWORDS = {
    "АПК РФ": ["арбитражн", "арбитражного суда", "экономических споров", "апк"],
    "БК РФ": ["бюджет", "бюджетн", "межбюджет", "казначей", "субсид"],
    "ГК РФ": ["гражданск", "сделк", "договор", "обязательств", "собственност", "наследов"],
    "ГПК РФ": ["гражданского судопроизводства", "исковое заявление", "гпк", "апелляцион"],
    "ГрК РФ": ["градостроительн", "застройщик", "строительств", "планировк", "архитектур"],
    "ЖК РФ": ["жилищн", "жилого помещения", "коммунальн", "многоквартирн", "нанимател"],
    "ЗК РФ": ["земельн", "земельного участка", "кадастр", "категор.*земель"],
    "КоАП РФ": ["административн.*правонарушен", "штраф", "коап", "протокол.*правонарушен"],
    "ЛК РФ": ["лесн", "лесопользован", "лесного фонда", "лесосек", "древесин"],
    "НК РФ": ["налог", "налогоплательщик", "налоговая база", "ставк.*налог", "нк рф", "ндс", "ндфл"],
    "СК РФ": ["брак", "алимент", "супруг", "родител.*прав", "усыновлен", "опек"],
    "ТК РФ": ["трудов", "работник", "работодател", "рабочее время", "отпуск", "увольнен"],
    "УК РФ": ["преступлен", "наказан", "лишение свободы", "штраф.*лет", "умышлен.*деян"],
    "УПК РФ": ["уголовн.*судопроизводств", "следствен", "дознан", "обвиняем", "подозревае"],
}


# ──────────────────────────── AnnotationAgent ────────────────────────

class AnnotationAgent:
    """
    Агент автоматической разметки текстовых данных.
    Поддерживает модальность text для корпуса статей НПА РФ.

    Скиллы:
      - auto_label(df)          → DataFrame с метками и confidence
      - generate_spec(df, task) → AnnotationSpec (Markdown)
      - check_quality(df)       → QualityMetrics
      - export_to_labelstudio(df) → JSON-файл
      - export_low_confidence(df) → CSV для ручной проверки (бонус)
    """

    def __init__(self, modality: str = 'text',
                 confidence_threshold: float = 0.55,
                 random_state: int = 42):
        """
        Parameters
        ----------
        modality : str
            Модальность данных ('text').
        confidence_threshold : float
            Порог уверенности для human-in-the-loop.
        random_state : int
            Seed для воспроизводимости.
        """
        self.modality = modality
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state

        # Для TF-IDF zero-shot
        self._tfidf = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            token_pattern=r'(?u)\b\w[\w-]+\b',
        )
        self._fitted = False
        self._class_vectors = None  # TF-IDF vectors описаний классов

        print(f"AnnotationAgent инициализирован (modality={modality}, "
              f"threshold={confidence_threshold})")

    # ──────────────── Skill 1: auto_label ────────────────

    def auto_label(self, df: pd.DataFrame,
                   text_column: str = 'content',
                   candidate_labels: List[str] = None,
                   method: str = 'hybrid') -> pd.DataFrame:
        """
        Автоматически размечает текстовые данные.

        Гибридный метод:
          1) Rule-based: ищет ключевые маркеры кодексов в тексте
          2) TF-IDF zero-shot: cosine similarity с описаниями классов
          3) Объединяет результаты → метка + confidence

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм с текстовыми данными.
        text_column : str
            Колонка с текстом.
        candidate_labels : list, optional
            Список допустимых меток. По умолчанию — все 15 кодексов.
        method : str
            'rule', 'tfidf' или 'hybrid' (по умолчанию).

        Returns
        -------
        pd.DataFrame
            Копия df с колонками: auto_label, confidence, low_confidence.
        """
        if candidate_labels is None:
            candidate_labels = list(CODEX_DESCRIPTIONS.keys())

        print(f"\n{'='*60}")
        print(f"auto_label: разметка {len(df)} документов (метод={method})")
        print(f"{'='*60}")

        df_out = df.copy()
        texts = df_out[text_column].fillna('').astype(str).values

        # ── Rule-based scores ──
        rule_labels, rule_confs = self._rule_based_classify(texts, candidate_labels)

        # ── TF-IDF zero-shot scores ──
        tfidf_labels, tfidf_confs = self._tfidf_zero_shot(texts, candidate_labels)

        # ── Combine ──
        final_labels = []
        final_confs = []

        for i in range(len(texts)):
            if method == 'rule':
                final_labels.append(rule_labels[i])
                final_confs.append(rule_confs[i])
            elif method == 'tfidf':
                final_labels.append(tfidf_labels[i])
                final_confs.append(tfidf_confs[i])
            else:  # hybrid
                # Если rule-based дал высокую уверенность, берём его
                if rule_confs[i] >= 0.6:
                    final_labels.append(rule_labels[i])
                    final_confs.append(min(rule_confs[i] * 0.7 + tfidf_confs[i] * 0.3, 1.0)
                                       if rule_labels[i] == tfidf_labels[i]
                                       else rule_confs[i] * 0.8)
                # Если TF-IDF уверен, берём его
                elif tfidf_confs[i] >= 0.5:
                    final_labels.append(tfidf_labels[i])
                    final_confs.append(tfidf_confs[i] * 0.8)
                # Иначе берём тот, кто увереннее
                elif rule_confs[i] >= tfidf_confs[i]:
                    final_labels.append(rule_labels[i])
                    final_confs.append(rule_confs[i] * 0.7)
                else:
                    final_labels.append(tfidf_labels[i])
                    final_confs.append(tfidf_confs[i] * 0.7)

        df_out['auto_label'] = final_labels
        df_out['confidence'] = np.round(final_confs, 4)
        df_out['low_confidence'] = df_out['confidence'] < self.confidence_threshold

        # Статистика
        n_low = df_out['low_confidence'].sum()
        print(f"\nРезультаты:")
        print(f"  Всего размечено: {len(df_out)}")
        print(f"  Средняя уверенность: {df_out['confidence'].mean():.3f}")
        print(f"  Низкая уверенность (<{self.confidence_threshold}): {n_low} ({n_low/len(df_out)*100:.1f}%)")
        print(f"\n  Распределение меток:")
        for label, count in df_out['auto_label'].value_counts().items():
            print(f"    {label}: {count} ({count/len(df_out)*100:.1f}%)")

        return df_out

    def _rule_based_classify(self, texts: np.ndarray,
                             candidate_labels: List[str]) -> Tuple[List[str], List[float]]:
        """Классифицирует тексты по ключевым маркерам."""
        labels = []
        confs = []
        for text in texts:
            text_lower = text.lower()
            scores = {}
            for label in candidate_labels:
                keywords = CODEX_KEYWORDS.get(label, [])
                if not keywords:
                    scores[label] = 0.0
                    continue
                hits = sum(1 for kw in keywords if re.search(kw, text_lower))
                scores[label] = hits / len(keywords)

            best_label = max(scores, key=scores.get) if scores else candidate_labels[0]
            best_score = scores.get(best_label, 0.0)

            # Если ни один маркер не сработал — "Отдельный НПА"
            if best_score == 0.0:
                best_label = "Отдельный нормативно-правовой акт"
                best_score = 0.3

            labels.append(best_label)
            confs.append(min(best_score, 1.0))
        return labels, confs

    def _tfidf_zero_shot(self, texts: np.ndarray,
                         candidate_labels: List[str]) -> Tuple[List[str], List[float]]:
        """Zero-shot через TF-IDF cosine similarity с описаниями классов."""
        # Описания классов
        class_descriptions = [CODEX_DESCRIPTIONS.get(l, l) for l in candidate_labels]

        # Обучаем TF-IDF на текстах + описаниях
        all_docs = list(texts) + class_descriptions
        tfidf_matrix = self._tfidf.fit_transform(all_docs)
        self._fitted = True

        n_texts = len(texts)
        text_vectors = tfidf_matrix[:n_texts]
        class_vectors = tfidf_matrix[n_texts:]

        # Cosine similarity
        sim_matrix = cosine_similarity(text_vectors, class_vectors)

        labels = []
        confs = []
        for i in range(n_texts):
            sims = sim_matrix[i]
            best_idx = np.argmax(sims)
            best_sim = sims[best_idx]
            labels.append(candidate_labels[best_idx])
            confs.append(float(best_sim))

        return labels, confs

    # ──────────────── Skill 2: generate_spec ────────────────

    def generate_spec(self, df: pd.DataFrame, task: str = 'document_type_classification',
                      text_column: str = 'content',
                      output_path: str = 'annotation_spec.md') -> AnnotationSpec:
        """
        Генерирует спецификацию разметки.

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм (желательно с колонкой auto_label после auto_label()).
        task : str
            Название задачи.
        text_column : str
            Колонка с текстом.
        output_path : str
            Путь для сохранения Markdown-файла.

        Returns
        -------
        AnnotationSpec
        """
        print(f"\nГенерация спецификации разметки: {task}")

        # Примеры: 3+ на каждый класс (берём из данных, если есть auto_label)
        examples = []
        label_col = 'auto_label' if 'auto_label' in df.columns else None

        if label_col:
            for label in sorted(df[label_col].unique()):
                subset = df[df[label_col] == label]
                # Берём до 3 примеров с наибольшей confidence
                if 'confidence' in subset.columns:
                    subset = subset.sort_values('confidence', ascending=False)
                for _, row in subset.head(3).iterrows():
                    text_preview = str(row[text_column])[:300]
                    examples.append({
                        'text': text_preview,
                        'label': label,
                        'explanation': CODEX_DESCRIPTIONS.get(label, ''),
                    })
        else:
            # Без разметки — просто показываем первые примеры
            for _, row in df.head(15).iterrows():
                text_preview = str(row[text_column])[:300]
                examples.append({
                    'text': text_preview,
                    'label': '(не размечен)',
                    'explanation': '',
                })

        edge_cases = [
            "Текст может содержать ссылки на несколько кодексов одновременно — выбирайте основной предмет регулирования",
            "Статья может быть в новой редакции (с указанием ФЗ о внесении изменений) — классифицируйте по содержанию, а не по ссылке на ФЗ",
            "Текст может содержать только номер статьи и заголовок без тела — классифицируйте по заголовку и контексту",
            "Постановления Правительства, приказы министерств, федеральные законы → «Отдельный нормативно-правовой акт»",
            "Короткие фрагменты (менее 100 символов) могут быть недостаточны для уверенной классификации — допустимо отметить как low-confidence",
        ]

        guidelines = [
            "Читайте первые 2-3 предложения — обычно они содержат название кодекса или ключевую терминологию",
            "Обращайте внимание на специфические термины: «налогоплательщик» → НК РФ, «работодатель» → ТК РФ, «арбитражный суд» → АПК РФ",
            "Если текст содержит «Правительство Российской Федерации постановляет» — скорее всего, Отдельный НПА",
            "Если сомневаетесь между двумя классами, выбирайте тот, к которому относится основная тема статьи",
            "При разметке сохраняйте единообразие: одинаковые типы текстов должны получать одинаковые метки",
        ]

        spec = AnnotationSpec(
            task=task,
            description=(
                "Классификация статей нормативно-правовых актов Российской Федерации "
                "по принадлежности к кодексу. Каждый текст — одна статья или фрагмент НПА. "
                "Необходимо определить, к какому из 15 кодексов (или к категории «Отдельный НПА») "
                "относится данный текст."
            ),
            classes=CODEX_DESCRIPTIONS,
            examples=examples,
            edge_cases=edge_cases,
            guidelines=guidelines,
        )

        # Сохраняем
        md_text = spec.to_markdown()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_text)
        print(f"  Спецификация сохранена: {output_path}")
        print(f"  Классов: {len(spec.classes)}, примеров: {len(spec.examples)}, "
              f"граничных случаев: {len(spec.edge_cases)}")

        return spec

    # ──────────────── Skill 3: check_quality ────────────────

    def check_quality(self, df_labeled: pd.DataFrame,
                      auto_label_col: str = 'auto_label',
                      ground_truth_col: Optional[str] = None,
                      confidence_col: str = 'confidence') -> QualityMetrics:
        """
        Оценивает качество разметки.

        Parameters
        ----------
        df_labeled : pd.DataFrame
            Размеченный датафрейм.
        auto_label_col : str
            Колонка с автоматическими метками.
        ground_truth_col : str, optional
            Колонка с эталонными метками (для Cohen's kappa).
        confidence_col : str
            Колонка с уверенностью.

        Returns
        -------
        QualityMetrics
        """
        print(f"\nОценка качества разметки")
        print("-" * 40)

        # Распределение меток
        label_dist = df_labeled[auto_label_col].value_counts().to_dict()

        # Confidence stats
        conf_mean = df_labeled[confidence_col].mean() if confidence_col in df_labeled.columns else 0.0
        conf_std = df_labeled[confidence_col].std() if confidence_col in df_labeled.columns else 0.0
        low_conf_count = int((df_labeled[confidence_col] < self.confidence_threshold).sum()) \
            if confidence_col in df_labeled.columns else 0

        # Cohen's kappa & percent agreement
        kappa = None
        percent_agreement = None

        if ground_truth_col and ground_truth_col in df_labeled.columns:
            valid = df_labeled[[auto_label_col, ground_truth_col]].dropna()
            if len(valid) > 0:
                y_auto = valid[auto_label_col].values
                y_true = valid[ground_truth_col].values

                # Percent agreement
                percent_agreement = float(np.mean(y_auto == y_true) * 100)

                # Cohen's kappa
                le = LabelEncoder()
                all_labels = list(set(y_auto) | set(y_true))
                le.fit(all_labels)
                kappa = float(cohen_kappa_score(
                    le.transform(y_true), le.transform(y_auto)
                ))

                print(f"  Cohen's kappa: {kappa:.3f}")
                print(f"  Percent agreement: {percent_agreement:.1f}%")
        else:
            print("  (Эталонные метки не предоставлены — kappa не рассчитан)")

        print(f"  Средняя уверенность: {conf_mean:.3f} (std={conf_std:.3f})")
        print(f"  Низкая уверенность: {low_conf_count} из {len(df_labeled)}")
        print(f"  Распределение меток:")
        for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
            print(f"    {label}: {count} ({count/len(df_labeled)*100:.1f}%)")

        return QualityMetrics(
            kappa=kappa,
            percent_agreement=percent_agreement,
            label_distribution=label_dist,
            confidence_mean=round(conf_mean, 4),
            confidence_std=round(conf_std, 4),
            low_confidence_count=low_conf_count,
            low_confidence_threshold=self.confidence_threshold,
            total_samples=len(df_labeled),
        )

    # ──────────────── Skill 4: export_to_labelstudio ────────────────

    def export_to_labelstudio(self, df_labeled: pd.DataFrame,
                              output_path: str = 'labelstudio_import.json',
                              text_column: str = 'content',
                              label_column: str = 'auto_label') -> str:
        """
        Экспортирует размеченные данные в формат LabelStudio.

        Parameters
        ----------
        df_labeled : pd.DataFrame
        output_path : str
        text_column : str
        label_column : str

        Returns
        -------
        str : путь к JSON-файлу
        """
        print(f"\nЭкспорт в LabelStudio: {output_path}")

        tasks = []
        for idx, row in df_labeled.iterrows():
            text = str(row[text_column]) if pd.notna(row.get(text_column)) else ""
            label = str(row[label_column]) if label_column in row and pd.notna(row[label_column]) else ""

            task = {
                "id": int(idx) if isinstance(idx, (int, np.integer)) else idx,
                "data": {
                    "text": text
                },
                "predictions": [{
                    "result": [{
                        "value": {
                            "choices": [label]
                        },
                        "from_name": "label",
                        "to_name": "text",
                        "type": "choices"
                    }],
                    "score": float(row.get('confidence', 0.0)),
                }]
            }
            tasks.append(task)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(f"  Экспортировано {len(tasks)} задач → {output_path}")
        return output_path

    # ──────────── Бонус: Human-in-the-loop ────────────

    def export_low_confidence(self, df_labeled: pd.DataFrame,
                              output_path: str = 'low_confidence_review.csv',
                              text_column: str = 'content') -> pd.DataFrame:
        """
        Экспортирует примеры с низкой уверенностью для ручной разметки.

        Parameters
        ----------
        df_labeled : pd.DataFrame
            Датафрейм с колонками auto_label, confidence, low_confidence.
        output_path : str
            Путь для CSV-файла.
        text_column : str

        Returns
        -------
        pd.DataFrame : только low-confidence записи
        """
        print(f"\nЭкспорт примеров с низкой уверенностью (<{self.confidence_threshold})")

        if 'low_confidence' not in df_labeled.columns:
            print("  Колонка low_confidence не найдена. Сначала запустите auto_label().")
            return pd.DataFrame()

        low_df = df_labeled[df_labeled['low_confidence']].copy()

        if len(low_df) == 0:
            print("  Нет примеров с низкой уверенностью.")
            return low_df

        # Добавляем колонки для ручной проверки
        low_df['manual_label'] = ''
        low_df['reviewer_notes'] = ''
        low_df['reviewed'] = False

        # Выбираем нужные колонки
        cols = [text_column, 'auto_label', 'confidence', 'manual_label', 'reviewer_notes', 'reviewed']
        cols = [c for c in cols if c in low_df.columns]

        low_df = low_df[cols].sort_values('confidence', ascending=True)
        low_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"  Экспортировано {len(low_df)} примеров → {output_path}")
        print(f"  Средн��я confidence: {low_df['confidence'].mean():.3f}")

        return low_df

    # ──────────── Утилиты ────────────

    def compare_with_human(self, df_labeled: pd.DataFrame,
                           human_labels_path: str,
                           text_column: str = 'content') -> QualityMetrics:
        """
        Сравнивает авторазметку с ручной разметкой от однокурсника.

        Parameters
        ----------
        df_labeled : pd.DataFrame
            Датафрейм с auto_label.
        human_labels_path : str
            Путь к CSV с колонками: content (или index), manual_label.
        text_column : str

        Returns
        -------
        QualityMetrics
        """
        print(f"\nСравнение с ручной разметкой: {human_labels_path}")

        human_df = pd.read_csv(human_labels_path)

        # Пробуем объединить по тексту или по индексу
        if 'manual_label' in human_df.columns and text_column in human_df.columns:
            merged = df_labeled.merge(
                human_df[[text_column, 'manual_label']],
                on=text_column,
                how='inner'
            )
        elif 'manual_label' in human_df.columns:
            merged = df_labeled.copy()
            merged['manual_label'] = human_df['manual_label'].values[:len(merged)]
        else:
            print("  Файл должен содержать колонку 'manual_label'")
            return None

        # Убираем пустые
        merged = merged[merged['manual_label'].notna() & (merged['manual_label'] != '')]

        if len(merged) == 0:
            print("  Нет совпадающих записей для сравнения")
            return None

        return self.check_quality(
            merged,
            auto_label_col='auto_label',
            ground_truth_col='manual_label',
            confidence_col='confidence',
        )
