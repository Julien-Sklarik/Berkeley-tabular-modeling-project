import pandas as pd
from packaging import version
import sklearn

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    num = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=True)
    )

    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        cat = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)
        )
    else:
        cat = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse=False)
        )

    pre = ColumnTransformer(
        transformers=[
            ("num", num, num_cols),
            ("cat", cat, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre

def make_pipeline(X: pd.DataFrame, rnd: int = 42):
    pre = build_preprocessor(X)
    model = HistGradientBoostingClassifier(random_state=rnd)
    pipe = Pipeline([("pre", pre), ("model", model)])

    cv = StratifiedKFold(n_splits=5, shuffle=False)
    scoring = "roc_auc"

    return pipe, cv, scoring
