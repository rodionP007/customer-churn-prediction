# Customer Churn Prediction

Machine Learning-проект по прогнозированию оттока клиентов телекоммуникационной компании.

Проект построен как последовательный ML pipeline: от исследовательского анализа данных и baseline-модели до подбора гиперпараметров, выбора threshold, финальной оценки и использования сохранённой модели для новых данных.

---

## 📌 Project Overview

**Задача:** предсказать, уйдёт ли клиент телекоммуникационной компании.

Целевая переменная:

- `Churn = 0` — клиент остался;
- `Churn = 1` — клиент ушёл.

Основной акцент проекта — построение воспроизводимого pipeline с корректным разделением train/test, preprocessing внутри `Pipeline`, подбором гиперпараметров через `GridSearchCV` и выбором threshold на out-of-fold предсказаниях.

---

## 📊 Dataset

Используется датасет **Telco Customer Churn**.

Источник: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Размер исходного датасета:

- **7043 наблюдения**
- **21 признак**

Целевая переменная — `Churn`.

В исходных данных присутствует умеренный дисбаланс классов:

- около **73.5%** — `Churn = 0`;
- около **26.5%** — `Churn = 1`.

---

## 🔎 Exploratory Data Analysis

EDA выполнен в `01_eda.ipynb`.

Основные этапы:

- проверка структуры и качества данных;
- анализ целевой переменной;
- обработка `TotalCharges`;
- анализ числовых признаков;
- анализ категориальных признаков;
- корреляционный анализ числовых признаков;
- формирование правил preprocessing.

### Основные наблюдения

Наиболее заметные различия в доле оттока наблюдаются для:

- `Contract`;
- `InternetService`;
- `PaymentMethod`;
- `TechSupport`;
- `OnlineSecurity`.

Ушедшие клиенты в среднем имеют меньший `tenure` и более высокие `MonthlyCharges`.

Также наблюдается:

- сильная положительная связь между `tenure` и `TotalCharges`;
- умеренная положительная связь между `MonthlyCharges` и `TotalCharges`.

Эти зависимости рассматриваются как статистические взаимосвязи, а не как доказательство причинности.

### `TotalCharges`

В исходном датасете `TotalCharges` представлен как строковый признак.

После преобразования в числовой тип обнаружены **11 пропущенных значений**. Они относятся к клиентам с `tenure = 0`.

В проекте используется правило:

```text
TotalCharges = 0, если tenure = 0
```

---

## ⚙️ Preprocessing

Preprocessing выполняется внутри `scikit-learn Pipeline`.

### Числовые признаки

Используется:

```python
StandardScaler()
```

### Категориальные признаки

Используется:

```python
OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
```

`customerID` исключается, поскольку является идентификатором клиента.

Вся предварительная обработка обучается только на `X_train`, что предотвращает утечку информации из тестовой выборки.

---

## 🧪 Train/Test Split

Данные разделяются в соотношении **80/20** со стратификацией по целевой переменной.

Тестовая выборка сохраняется до этапа финальной оценки.

Она **не используется** для:

- выбора гиперпараметров;
- выбора threshold;
- обучения preprocessing.

---

## 🤖 Models

В проекте исследуются три модели:

1. Logistic Regression;
2. Decision Tree;
3. Random Forest.

### Baseline

В качестве baseline используется Logistic Regression со стандартным:

```text
threshold = 0.5
```

Baseline формирует исходную точку для дальнейшего сравнения.

---

## 🔧 Hyperparameter Optimization

Для финального подбора гиперпараметров используется `GridSearchCV` с **5-fold cross-validation**.

Оптимизируемая метрика:

```text
F1-score
```

### Logistic Regression

Исследуемые параметры:

```text
C = {0.01, 0.1, 1, 10, 100}
class_weight = {None, balanced}
```

Финальная конфигурация:

```text
C = 0.1
class_weight = balanced
```

### Decision Tree

Исследуемые параметры:

```text
max_depth = {2, 3, 4, 5, 6, 8, 10}
min_samples_leaf = {1, 2, 5, 10, 20}
```

Финальная конфигурация:

```text
max_depth = 8
min_samples_leaf = 20
```

### Random Forest

Исследуемые параметры:

```text
n_estimators = {100, 200, 500}
max_depth = {6, 8, 10}
min_samples_leaf = {1, 5, 10}
```

Финальная конфигурация:

```text
n_estimators = 500
max_depth = 10
min_samples_leaf = 10
```

---

## 🎚️ Threshold Selection

Помимо гиперпараметров модели, исследуется порог классификации.

После выбора гиперпараметров threshold подбирается на **out-of-fold предсказаниях `X_train`**.

Проверяются значения:

```text
0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
```

Тестовая выборка при выборе threshold не используется.

Финальные значения:

| Model | Threshold |
|---|---:|
| Logistic Regression | 0.5 |
| Decision Tree | 0.3 |
| Random Forest | 0.3 |

Threshold применяется после получения вероятности класса `Churn = 1`:

```python
y_pred = (y_proba >= threshold).astype(int)
```

---

## 📈 Final Evaluation

Финальная оценка выполняется один раз на отложенной тестовой выборке.

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5 | 0.742 | 0.510 | 0.783 | 0.617 | 0.841 |
| Decision Tree | 0.3 | 0.751 | 0.522 | 0.730 | 0.609 | 0.820 |
| **Random Forest** | **0.3** | **0.759** | **0.532** | **0.778** | **0.632** | **0.845** |

### Final Model

В качестве финальной модели проекта используется **Random Forest** со следующей конфигурацией:

```text
n_estimators     = 500
max_depth        = 10
min_samples_leaf = 10
threshold        = 0.3
```

Результаты на тестовой выборке:

```text
Accuracy  = 0.759
Precision = 0.532
Recall    = 0.778
F1        = 0.632
ROC-AUC   = 0.845
```

Выбор основан на сравнении исследованных моделей на отложенной тестовой выборке. Random Forest показал наиболее высокие значения F1-score и ROC-AUC среди рассмотренных конфигураций.

---

## 🔍 Error Analysis

Для анализа ошибок используются:

- Confusion Matrix;
- False Positive (FP);
- False Negative (FN);
- ROC Curve;
- Precision-Recall Curve;
- распределения предсказанных вероятностей.

Финальные confusion matrices:

### Logistic Regression

```text
[[753 282]
 [ 81 293]]
```

### Decision Tree

```text
[[785 250]
 [101 273]]
```

### Random Forest

```text
[[779 256]
 [ 83 291]]
```

Анализ FP/FN позволяет отдельно рассматривать:

- клиентов, ошибочно классифицированных как ушедшие;
- ушедших клиентов, которых модель пропустила.

---

## 💾 Saved Model

Финальная модель сохраняется как полный `scikit-learn Pipeline`:

```text
models/churn_random_forest.joblib
```

Внутри Pipeline находятся:

```text
Pipeline
├── ColumnTransformer
│   ├── StandardScaler
│   └── OneHotEncoder
│
└── RandomForestClassifier
```

Сохранение всего Pipeline позволяет при inference автоматически применять те же preprocessing-преобразования, которые использовались при обучении.

---

## 🚀 Prediction

Для новых данных используется:

```text
predict.py
```

Запуск:

```bash
python predict.py path/to/input.csv
```

Например:

```bash
python predict.py data/new_customers.csv
```

Скрипт:

1. загружает сохранённый Pipeline;
2. читает CSV;
3. удаляет `customerID`;
4. преобразует `TotalCharges` в числовой формат;
5. обрабатывает `TotalCharges` для клиентов с `tenure = 0`;
6. получает вероятность `P(Churn = 1)`;
7. применяет `threshold = 0.3`;
8. выводит вероятность и итоговый прогноз.

Пример результата:

```text
Churn_Probability  Churn_Prediction
0.728265           1
0.018634           0
0.689172           1
0.114779           0
0.592432           1
```

---

## 📁 Project Structure

```text
customer_churn_prediction/
│
├── data/
│   └── ...
│
├── models/
│   └── churn_random_forest.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── predict.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Joblib

---

## ▶️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd customer_churn_prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📚 Notebook Pipeline

Проект разделён на четыре этапа:

```text
01_eda
   ↓
Исследование данных
   ↓
02_baseline
   ↓
Baseline Logistic Regression
   ↓
03_modeling
   ↓
GridSearchCV + OOF threshold selection
   ↓
04_evaluation
   ↓
Финальная оценка и выбор модели
```

---

## 📌 Project Status

**Version 1.0 — completed**

Текущая версия содержит полный цикл:

- EDA;
- preprocessing;
- baseline;
- несколько ML-моделей;
- hyperparameter optimization;
- threshold selection;
- final evaluation;
- error analysis;
- сохранение модели;
- inference через `predict.py`.

