from berkeley_tabular.data import load_adult
from berkeley_tabular.model import make_pipeline
from berkeley_tabular.evaluate import small_grid_search, shuffled_target_auc

def test_label_shuffle_auc_range():
    X, y = load_adult("data/adult.csv")
    pipe, cv, scoring = make_pipeline(X)
    best, _ = small_grid_search(pipe, cv, scoring, X, y)
    auc = shuffled_target_auc(best, cv, X, y)
    assert 0.45 <= auc <= 0.55
