from abc import abstractmethod

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import metrics


class BinClassifier:
    """Classifier expects a pandas dataframe with a binary target.
    
    If categorical features are not handled prior to submitting
    to classifier, the classifier will attempt to abstract
    categories using heuristics. Missing float formatted data 
    is imputed using mean of feature. All data is scaled before
    training time. If categories in test set differ from those in 
    the feature space of the trained model, they are omitted.
    
    :param clf: Trained binary classifier model using logistic regression.
    :param clf_cv: Trained binary classifier model using logistic regression
                    and cross validation.
    :param d_cv: Hash table of cross validation metrics.
    :param proba: Predicted probability associated with target.
    :param preds: Predictions produced by model on test set.
    :param max_categories: Maximum number of categories to abstract using heuristic.
    :param req_fts: Required features for model to make predictions.
    """
    def __init__(self):
        self.clf = None
        self.coef_ = None
        self.clf_cv = None
        self.d_cv = {}
        self.feature_space = None
        self.proba = None
        self.preds = None
        self.max_categories = None
        self.req_fts = None
        
    def get_categorical(self, data, max_categories=None, proportion_cat=0.05):
        """Obtain categorical data.
        Heuristic to determine if feature is categorical.
        If ratio of number of unique values to the total 
        number of unique values is less than proportion_cat, assume
        feature is categorical.
        
        :param data: Pandas dataframe or matrix.
        :return: Transformed data set with handled categorical data.
        :rtype: Pandas dataframe.
        """
        data_len = len(data)
        
        # Heuristic to determine which potential categorical 
        # features meets proportion_cat threshold or features with equal to or
        # fewer than max_categories
        categorical_ft = {}
        for ft in data.columns:
            categorical_ft[ft] = (1.*data[ft].nunique()/data[ft].dropna().count() < proportion_cat)
            
        if max_categories:
            for key in categorical_ft:
                if categorical_ft[key]:
                    uniq_ct = data[ft].nunique()
                    if uniq_ct > max_categories:
                        categorical_ft[key] = False
        
        # Make categorical columns
        data = pd.get_dummies(data=data, columns=[ft for ft in data.columns if categorical_ft[ft]==True])
        
        # Remove string columns not considered to be categorical
        data = data[[c for c in data.columns if data[c].dtypes!=object]]
        
        # Remove na categorized features
        data = data[[c for c in data.columns if '_na' not in c]]
        return data
    
    def impute_mean(self, data):
        """Impute mean for nan values in columns with int/float type.
        
        :param data: Pandas dataframe.
        :return: Pandas dataframe with imputed data.
        :rtype: Pandas dataframe
        """
        # Impute nan values with feature mean for fit
        for ft in data.columns:
            if data[ft].dtype == np.float:
                data[ft].fillna((data[ft].mean()), inplace=True)
        return data
    
    def scale_data(self, data):
        """Scale data of int/float type.
        
        :param data: Pandas dataframe.
        :type data: Pandas dataframe.
        :return: Pandas dataframe with scaled data.
        :rtype: Pandas dataframe
        """
        for ft in data.columns:
            if data[ft].dtype == np.float:
                data[ft] = preprocessing.MinMaxScaler().fit_transform(data[[ft]])
        return data
    
    def remove_unhelpful_data(self, data):
        """Remove data that isn't helpful for model.
        
        :param data: Pandas dataframe.
        :type data: Pandas dataframe.
        :return: Pandas dataframe with scaled data.
        :rtype: Pandas dataframe"""
        
        # Remove columns with single value
        keep = []
        for ft in data.columns:
            if data[ft].nunique()>1:
                keep.append(ft)
        data = data[keep]
        
        # Remove id columns
        keep = []
        for ft in data.columns:
            if ft!='id' and ft!='ID' and ft!='Id':
                keep.append(ft)
        data = data[keep]
        return data
    
    def prev_overfit(self, X, y):
        """Reduce feature space to floor(sqrt(size(data)))//2
        to prevent overfitting.
        
        :param data: Pandas dataframe.
        :type data: Pandas dataframe.
        :return: Pandas dataframe having reduced feature space.
        :rtype: Pandas dataframe"""
        hi = int(len(X)**0.5)//2
        
        var_vec = []
        
        for c in X.columns:
            if X[c].nunique()==1:
                continue
            lr = LogisticRegression()
            lr.fit(X[[c]] / np.std(X[[c]], 0), y) # variance retention
            var_vec.append([c, abs(lr.coef_)])
        
        var_vec = sorted(var_vec, key=lambda x: x[1])
        self.feature_space = [x[0] for x in var_vec[-hi:]]
        return X[self.feature_space]
        
    def validate_data(self, data, max_categories=None):
        """Validate data is suitable for model to fit to.
        
        :param data: Training data for model to fit to.
        :type data: Pandas dataframe.
        :return: Validated data.
        :rtype: Pandas dataframe.
        """
        # Remove unhelpful data
        data = self.remove_unhelpful_data(data)
        
        # Get categorical data
        data = self.get_categorical(data, max_categories=max_categories)
        
        # Impute mean
        data = self.impute_mean(data)
        
        # Scale data
        data = self.scale_data(data)
        return data
    
    @abstractmethod
    def fit(self, X, y, max_categories=None):
        """Fit a logistic regression model 
        
        :param X: Training data mapping to y target vector.
        :type X: Pandas dataframe or vector.
        :param y: Target vector (X must map to this target vector).
        :type y: Pandas dataframe or vector.
        :return: Fitted Model.
        """
        if not self.clf:
            clf = LogisticRegression()
            X = self.validate_data(X, max_categories=max_categories)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            
            # Prevent overfitting by limiting feature space to floor(sqrt(size(data)))//2
            X_train = self.prev_overfit(X_train, y_train.values.ravel())
            clf.fit(X_train, y_train.values.ravel())
            self.clf = clf
            self.coef_ = clf.coef_
            self.max_categories = max_categories
            self.req_fts = X_train.columns
        return self
    
    def predict(self, X):
        """Make predictions using fitted model. 
        Model expects same feature space it was fitted on.
        
        :param X: Data for model to produce predictions for.
        :type X: Pandas dataframe or matrix.
        :return: Predictions.
        :rtype: Vector.
        """
        if not self.preds:
            X = self.validate_data(X, self.max_categories)
            X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)
            diff = set(X_test.columns)-set(self.req_fts)
            if diff:
                print("WARNING: The following features were dropped due to not being included in model: ")
                print(diff)
            X_test = X_test[self.req_fts] # handle any newly introduced features
            X_test = self.scale_data(X_test)
            self.preds = self.clf.predict(X_test)
        return self.preds
    
    def predict_proba(self, X):
        """Produce probabilities for predictions using fitted model. 
        Model expects same feature space it was fitted on.
        
        :param X: Data for model to produce predictions for.
        :type X: Pandas dataframe or matrix.
        :return: Probabilities for predictions.
        :rtype: Vector.
        """
        if not self.proba:
            X = self.validate_data(X, self.max_categories)
            X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)
            diff = set(X_test.columns)-set(self.req_fts)
            if diff:
                print("WARNING: The following features were dropped due to not being included in model: ")
                print(diff)
            X_test = X_test[self.req_fts] # handle any newly introduced features
            X_test = self.scale_data(X_test)
            self.proba = self.clf.predict_proba(X_test)
        return self.proba
    
    def evaluate(self, X, y):
        """Evaluate model with F1-score, LogLoss
        
        :param X: Data for trained model to be evaluated on.
        :type X: Pandas dataframe or matrix.
        :param y: True target.
        :type y: Vector.
        :return: Table of F1-score, LogLoss.
        :rtype: Hash table.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        d_eval = {'f1_score': False, 'logloss': False}
        d_eval['f1_score'] = metrics.f1_score(y_test, self.preds, average='weighted')
        d_eval['logloss'] = metrics.log_loss(y_test, self.proba)
        return d_eval
    
    def tune_parameters(self, X, y, solver='sag'):
        """Evaluate model with F1-score, LogLoss
        
        :param X: Data for trained model to be evaluated on.
        :type X: Pandas dataframe or matrix.
        :param y: True target.
        :type y: Vector.
        :return: Table of F1-score, LogLoss.
        :rtype: Hash table.
        """
        if not self.clf_cv:
            X = self.validate_data(X, self.max_categories)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            
            # Prevent overfit
            X_train = self.prev_overfit(X_train, y_train.values.ravel())
            clf_cv = LogisticRegressionCV(cv=5, solver='sag', tol=0.02)
            clf_cv.fit(X_train, y_train.values.ravel())
            self.clf_cv = clf_cv
            
            d_info = {'tol': 0.02, 'fit_intercept': False, 'solver': solver}
            
            f1_score = metrics.f1_score(y_test, self.preds, average='weighted')
            log_loss = metrics.log_loss(y_test, self.proba)
            d_info['scores'] = {'f1_score': f1_score, 'logloss': log_loss}
            
            self.d_cv = d_info
        return [np.average(x) for x in self.clf_cv.scores_[1]], self.d_cv
        
        
# def run():
#     # Test Set 1: 
#         # f1_score: 0.672, 'logloss': 0.571
#     from bin_classifier.datasets import load_nba_rookie_lasts_5yr
#     df=load_nba_rookie_lasts_5yr()
#     X, y = df[[x for x in df.columns if x!='TARGET_5Yrs']], df['TARGET_5Yrs']
    
#     # Test Set 2
#         # f1_score: 0.942, logloss: 0.190
# #     from sklearn.datasets import load_breast_cancer
# #     import numpy as np
# #     import pandas as pd
# #     data = load_breast_cancer()
# #     df = pd.DataFrame(np.c_[data['data'], data['target']], columns= np.append(data['feature_names'], ['target']))
# #     X, y = df[[c for c in df.columns if c!='target']], df[['target']]
    
#     clf=BinClassifier()
#     clf.fit(X, y)
#     print(clf.predict(X))
#     print(clf.predict_proba(X))
#     print(clf.evaluate(X, y))
#     print(clf.tune_parameters(X, y))
#     return 1
    
# if __name__=="__main__":
#     run()