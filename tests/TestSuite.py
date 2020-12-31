import unittest

import numpy as np
import pandas as pd

from bin_classifier.lr_bin_classifier import LrBinClassifier


class GetCategoricalTest(unittest.TestCase):
    def test_get_categorical(self):
        df = pd.DataFrame({"a": [1 if x % 2 == 0 else 0 for x in range(100)]})
        df = LrBinClassifier().get_categorical(df)
        assert df.shape == (100, 2)


class ImputeMeanTest(unittest.TestCase):
    def test_impute_mean(self):
        df = pd.DataFrame({"a": [1 if x % 2 == 0 else 0 for x in range(40)]})
        df.a.iloc[[9]] = np.nan
        df = LrBinClassifier().impute_mean(df)
        assert 0.50 < df.a.iloc[9] < 0.52


class ScaleDataTest(unittest.TestCase):
    def test_scale_data(self):
        df = pd.DataFrame({"a": [1 if x % 2 == 0 else 0 for x in range(40)]})
        df = LrBinClassifier().scale_data(df)
        assert min(df["a"]) == 0
        assert max(df["a"]) == 1


class RemoveUnhelpfulDataTest(unittest.TestCase):
    def test_remove_unhelpful_data(self):
        df = pd.DataFrame({"a": [x for x in range(10)]})
        df["b"] = [0 for x in range(10)]
        df = LrBinClassifier().remove_unhelpful_data(df)
        assert len(df.columns) == 1


class PrevOverfitTest(unittest.TestCase):
    def test_prev_overfit(self):
        df = pd.DataFrame({"y": [1 if x % 2 == 0 else 0 for x in range(100)]})
        for l in "abcdefghijklmnopqrstuvwxz":
            df[l] = [np.random.randint(50) for x in range(100)]
        X, y = df[[c for c in df.columns if c != "y"]], df[["y"]]
        X = LrBinClassifier().prev_overfit(X, y)
        assert len(X.columns) == (100 ** 0.5) // 2


class ValidateDataTest(unittest.TestCase):
    def test_validate_data(self):
        df = pd.DataFrame({"a": [1 if x % 2 == 0 else 0 for x in range(200)]})
        LrBinClassifier().validate_data(df)
        assert "a_0" in df.columns
        assert "a_1" in df.columns


class FitTest(unittest.TestCase):
    def test_fit(self):
        df = pd.DataFrame({"X": [x + 0.01 for x in range(100)]})
        df["y"] = [1 if x % 2 == 0 else 0 for x in range(100)]
        clf = LrBinClassifier()
        clf.fit(df[["X"]], df["y"], max_categories=10)
        assert clf.clf is not None


class PredictTest(unittest.TestCase):
    def test_predict(self):
        df = pd.DataFrame({"X": [x + 0.01 for x in range(1000)]})
        df["y"] = [1 if x % 2 == 0 else 0 for x in range(1000)]
        clf = LrBinClassifier()
        clf.fit(df[["X"]], df["y"], max_categories=10)
        df["added_feature"] = 1  # adding random feature to test model durability
        preds = clf.predict(df[["X"]])
        assert sum(preds) > 0


class PredictProbaTest(unittest.TestCase):
    def test_predict_proba(self):
        df = pd.DataFrame({"X": [x + 0.01 for x in range(1000)]})
        df["y"] = [1 if x % 2 == 0 else 0 for x in range(1000)]
        clf = LrBinClassifier()
        clf.fit(df[["X"]], df["y"], max_categories=10)
        df["added_feature"] = 1  # adding random feature to test model durability
        probas = clf.predict_proba(df[["X"]])
        assert sum(1 if x[0] < 0.50 else 0 for x in probas) > 0


class EvaluateTest(unittest.TestCase):
    def test_evaluate(self):
        df = pd.DataFrame({"X": [x + 0.01 for x in range(100)]})
        df["y"] = [1 if x % 2 == 0 else 0 for x in range(100)]
        clf = LrBinClassifier()
        clf.fit(df[["X"]], df["y"], max_categories=10)
        clf.predict(df[["X"]])
        clf.predict_proba(df[["X"]])
        evals = clf.evaluate(df[["X"]], df["y"])
        assert evals["f1_score"] > 0
        assert evals["logloss"] > 0


class TuneParametersTest(unittest.TestCase):
    def test_tune_parameters(self):
        df = pd.DataFrame({"X": [x + 0.01 for x in range(100)]})
        df["y"] = [1 if x % 2 == 0 else 0 for x in range(100)]
        clf = LrBinClassifier()
        clf.fit(df[["X"]], df["y"], max_categories=10)
        clf.predict(df[["X"]])
        clf.predict_proba(df[["X"]])
        clf.evaluate(df[["X"]], df["y"])
        scores, params_metrics = clf.tune_parameters(df[["X"]], df["y"])
        assert np.average(scores) > 0
        assert params_metrics["scores"]["f1_score"] > 0
        assert params_metrics["scores"]["logloss"] > 0


def create_suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(GetCategoricalTest())
    test_suite.addTest(ImputeMeanTest())
    test_suite.addTest(ScaleDataTest())
    test_suite.addTest(RemoveUnhelpfulDataTest())
    test_suite.addTest(PrevOverfitTest())
    test_suite.addTest(ValidateDataTest())
    test_suite.addTest(FitTest())
    test_suite.addTest(PredictTest())
    test_suite.addTest(PredictProbaTest())
    test_suite.addTest(EvaluateTest())
    test_suite.addTest(TuneParametersTest())
    return test_suite


if __name__ == "__main__":
    unittest.main()
