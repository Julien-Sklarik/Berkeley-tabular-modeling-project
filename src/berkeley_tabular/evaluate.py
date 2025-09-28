from pathlib import Path
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve, confusion_matrix
from sklearn.base import clone
from sklearn.inspection import permutation_importance

def small_grid_search(pipe, cv, scoring, X, y):
    param_grid = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [None, 6, 12]
    }
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=1, verbose=1)
    gs.fit(X, y)
    best = gs.best_estimator_

    rows = []
    for k, (tr, te) in enumerate(cv.split(X, y), start=1):
        m = clone(best).fit(X.iloc[tr], y.iloc[tr])
        if hasattr(m.named_steps["model"], "predict_proba"):
            s = m.predict_proba(X.iloc[te])[:, 1]
        elif hasattr(m.named_steps["model"], "decision_function"):
            s = m.decision_function(X.iloc[te])
        else:
            s = m.predict(X.iloc[te]).astype(float)
        yhat = (s >= 0.5).astype(int)

        rows.append({
            "fold": k,
            "AUC": roc_auc_score(y.iloc[te], s),
            "ACC": accuracy_score(y.iloc[te], yhat),
            "F1": f1_score(y.iloc[te], yhat)
        })

    cv_table = pd.DataFrame(rows)
    return best, cv_table

def oof_scores(best, cv, X, y):
    oof = np.zeros(len(y), dtype=float)
    rows = []
    for k, (tr, te) in enumerate(cv.split(X, y), start=1):
        m = clone(best).fit(X.iloc[tr], y.iloc[tr])
        if hasattr(m.named_steps["model"], "predict_proba"):
            s = m.predict_proba(X.iloc[te])[:, 1]
        elif hasattr(m.named_steps["model"], "decision_function"):
            s = m.decision_function(X.iloc[te])
        else:
            s = m.predict(X.iloc[te]).astype(float)
        oof[te] = s
        yhat = (s >= 0.5).astype(int)

        rows.append({
            "fold": k,
            "AUC": roc_auc_score(y.iloc[te], s),
            "ACC": accuracy_score(y.iloc[te], yhat),
            "F1": f1_score(y.iloc[te], yhat)
        })
    fold_df = pd.DataFrame(rows)
    return oof, fold_df

def choose_threshold_max_f1(y, oof):
    tgrid = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_f1 = 0.0
    for t in tgrid:
        pred = (oof >= t).astype(int)
        f1 = f1_score(y, pred)
        if f1 >= best_f1:
            best_f1 = f1
            best_t = float(t)

    pred = (oof >= best_t).astype(int)
    cm = confusion_matrix(y, pred)
    rep = {
        "threshold": float(best_t),
        "confusion_matrix": cm.tolist()
    }
    txt = []
    txt.append(f"Chosen threshold {float(best_t):.4f}")
    txt.append("")
    txt.append("Confusion matrix")
    txt.append(str(cm))
    report_text = "\n".join(txt)
    return best_t, report_text, cm

def shuffled_target_auc(best, cv, X, y):
    y_shuf = pd.Series(y.values.copy()).sample(frac=1.0, random_state=42).reset_index(drop=True)
    oof = np.zeros(len(y_shuf), dtype=float)
    for tr, te in cv.split(X, y_shuf):
        m = clone(best).fit(X.iloc[tr], y_shuf.iloc[tr])
        if hasattr(m.named_steps["model"], "predict_proba"):
            s = m.predict_proba(X.iloc[te])[:, 1]
        elif hasattr(m.named_steps["model"], "decision_function"):
            s = m.decision_function(X.iloc[te])
        else:
            s = m.predict(X.iloc[te]).astype(float)
        oof[te] = s
    auc = roc_auc_score(y_shuf, oof)
    return auc

def permutation_importances_raw(best, X, y):
    model = clone(best).fit(X, y)
    res = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=1)
    imp = pd.Series(res.importances_mean, index=X.columns).sort_values(ascending=False)
    top = imp.head(20).reset_index()
    top.columns = ["feature", "importance"]

    fig = plt.figure(figsize=(8, 5))
    plt.bar(top["feature"], top["importance"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Top permutation importances on raw columns")
    plt.tight_layout()

    return top, fig

def save_report_files(results_dir: Path, fold_df, oof, y, thresh, report_text, cm, cv_table, imp_df, fig):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cv_table.to_csv(results_dir / "per_fold.csv", index=False)
    fold_df.to_csv(results_dir / "oof_by_fold.csv", index=False)

    metrics = {
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_f1_at_chosen_threshold": float(f1_score(y, (oof >= thresh).astype(int))),
        "threshold": float(thresh)
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report_text)

    imp_df.to_csv(results_dir / "importances.csv", index=False)
    fig.savefig(results_dir / "top_importances.png", dpi=160)
    plt.close(fig)
