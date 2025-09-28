from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

def load_adult(csv_path: Path):
    """
    Load the Adult dataset as a DataFrame X and a binary Series y.
    If csv_path exists, read that file. Otherwise fetch from OpenML and cache it.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        openml = fetch_openml(name="adult", version=2, as_frame=True)
        df = openml.frame.copy()
        df.to_csv(csv_path, index=False)

    target = "income"
    y_raw = df[target].astype(str).str.strip()
    y = (y_raw == ">50K").astype(int)
    X = df.drop(columns=[target]).copy()

    # Ensure categorical columns are string typed for the encoder
    for c in X.select_dtypes(exclude=["number", "bool"]).columns:
        X[c] = X[c].astype(str)

    return X, y
