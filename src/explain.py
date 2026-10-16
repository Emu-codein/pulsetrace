"""SHAP for the chosen model: which features drive the predictions?

    python -m src.explain     # writes reports/figures/shap_*.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.evaluation import TARGETS, load_clean, split_train_test
from src.run_ladder import ALL_FEATURES, xgb

OUT = 'reports/figures'


def main(n_sample=3000):
    os.makedirs(OUT, exist_ok=True)
    train, _ = split_train_test(load_clean())
    train = train.dropna(subset=ALL_FEATURES).reset_index(drop=True)
    model = xgb(max_depth=6, n_estimators=600, learning_rate=0.05, min_child_weight=50)
    model.fit(train[ALL_FEATURES].to_numpy(), train[TARGETS].to_numpy())

    # random windows from the training data (long recordings contribute more windows)
    sample = train.sample(n_sample, random_state=0)
    X = sample[ALL_FEATURES]
    importance = {}
    for est, t in zip(model.estimators_, TARGETS):
        sv = shap.TreeExplainer(est).shap_values(X)
        importance[t] = pd.Series(np.abs(sv).mean(axis=0), index=ALL_FEATURES)
        plt.figure()
        shap.summary_plot(sv, X, show=False, max_display=len(ALL_FEATURES))
        plt.title(f'SHAP: {t}')
        plt.tight_layout()
        plt.savefig(f'{OUT}/shap_{t}_beeswarm.png', dpi=130)
        plt.close()

    imp = pd.DataFrame(importance).sort_values('sys', ascending=True)
    imp.plot.barh(figsize=(7, 4.5))
    plt.xlabel('mean |SHAP| (mmHg)')
    plt.title('Feature importance by target')
    plt.tight_layout()
    plt.savefig(f'{OUT}/shap_importance.png', dpi=130)
    print(imp.sort_values('sys', ascending=False).round(2).to_string())


if __name__ == '__main__':
    main()
