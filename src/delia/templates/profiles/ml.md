# Machine Learning Profile

Load this profile for: scikit-learn, data science, Jupyter, pandas, feature engineering.

## Project Structure

```
project/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── load.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
└── tests/
```

## Data Loading & Exploration

```python
import pandas as pd
import numpy as np

def load_and_explore(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Missing:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    # Statistics
    print(df.describe())

    return df
```

## Data Preprocessing

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def create_preprocessor(numeric_cols: list, categorical_cols: list):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
```

## Feature Engineering

```python
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Date features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Aggregations
    df["user_mean"] = df.groupby("user_id")["value"].transform("mean")
    df["user_std"] = df.groupby("user_id")["value"].transform("std")

    # Interactions
    df["feature_ratio"] = df["feature_a"] / (df["feature_b"] + 1e-8)

    # Binning
    df["value_bin"] = pd.qcut(df["value"], q=5, labels=False)

    return df
```

## Model Training

```python
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_and_evaluate(X, y, model, cv=5):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    print(f"CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train final model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model


def save_model(model, preprocessor, path: str):
    joblib.dump({"model": model, "preprocessor": preprocessor}, path)
```

## Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "learning_rate": uniform(0.01, 0.3),
}

search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=50,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
    random_state=42,
)

search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_:.4f}")
```

## Jupyter Best Practices

```python
# Cell 1: Imports and config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
%load_ext autoreload
%autoreload 2

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 100)

# Cell 2: Data loading
# ...

# Use markdown cells for documentation
# Keep cells focused and executable independently
```

## Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance["feature"], importance["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig
```

## Best Practices

```
ALWAYS:
- Explore data before modeling
- Use train/val/test splits
- Track experiments (MLflow, W&B)
- Version control data + code
- Document assumptions

AVOID:
- Data leakage (fit on test data)
- Ignoring class imbalance
- Overfitting to validation set
- Skipping cross-validation
```

