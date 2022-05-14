import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RoundTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, significant):
        self.significant = significant

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # creating a copy to avoid changes to original dataset
        X_ = X.copy()
        X_ = np.round(X_, self.significant)
        return X_
