import os, json, warnings, sys
from pathlib import Path

# ensure src is on path
root = Path(__file__).resolve().parent
sys.path.append(str(root / "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from berkeley_tabular.data import load_adult
from berkeley_tabular.model import make_pipeline
from berkeley_tabular.evaluate import (
    small_grid_search,
    oof_scores,
    choose_threshold_max_f1,
    shuffled_target_auc,
    permutation_importances_raw,
    save_report_files
)

warnings.filterwarnings("ignore")

def main():
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # load data
    X, y = load_adult(root / "data" / "adult.csv")

    # build pipeline
    pipe, cv, scoring = make_pipeline(X)

    # grid search
    best, cv_table = small_grid_search(pipe, cv, scoring, X, y)

    # honest out of fold scores
    oof, fold_df = oof_scores(best, cv, X, y)

    # threshold selection
    thresh, report_text, cm = choose_threshold_max_f1(y, oof)

    # sanity check
    auc_shuf = shuffled_target_auc(best, cv, X, y)

    # permutation importances on raw columns
    imp_df, fig = permutation_importances_raw(best, X, y)

    # save files
    save_report_files(results_dir, fold_df, oof, y, thresh, report_text, cm, cv_table, imp_df, fig)

    print("Done. See the results folder for outputs.")

if __name__ == "__main__":
    main()
