
# Automatic Binary Classification by way of Logistic Regression

Automatically train a binary classifier on some set of data without any preparation needed from user. After the model is trained, make predictions on a new sample containing at least the same feature space. If features beyond the feature space are included in an unseen sample, the new features are omitted.

### Usage
This package leverages the **scikit-learn** framework so much of mechanics will look similar.
```python
from bin_classifier import BinClassifier

import numpy as np
import pandas as pd

from bin_classifier.datasets import load_nba_rookie_lasts_5yr
df=load_nba_rookie_lasts_5yr()
X, y = df[[x for x in df.columns if x!='TARGET_5Yrs']], df['TARGET_5Yrs']

clf=BinClassifier()
clf.fit(X, y)
clf.predict(X)
# [0 1 1 ... 0 1 0]
clf.predict_proba(X)
# [[0.55  0.45]
#  [0.68 0.32]
#  [0.11 0.89]
#  ...
#  [0.15 0.85]
#  [0.24 0.76]
#  [0.33 0.67]]
clf.evaluate(X, y)
# {'f1_score': 0.672, 'logloss': 0.571}
clf.tune_parameters(X, y)
# ([0.663, 0.635, 0.645, 0.641, 0.658], {'tol': 0.02, 'fit_intercept': False, 'solver': 'sag', 'scores': {'f1_score': 0.672, 'logloss': 0.571}})
```