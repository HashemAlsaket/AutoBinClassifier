from abc import abstractmethod, ABC


class BinClassifier(ABC):
    """Base Abstract Class Definition.

    All child classes need to implement the following
    method signatures to maintain a consistent
    interface.
    """
    @abstractmethod
    def fit(self, X, y, max_categories=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X, y):
        raise NotImplementedError
