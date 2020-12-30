import numpy as np
import pandas as pd
import bin_classifier


fil = bin_classifier.__file__
fil = fil.replace("__init__.py", "")

def load_nba_rookie_survival_5yr():
    return pd.read_csv(fil + "data/nba_rookie_survival_5yr.csv")