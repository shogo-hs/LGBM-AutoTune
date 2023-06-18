import numpy as np


class BaggingModels:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = np.column_stack(
            [model.predict(X) for model in self.models]
        )
        return np.where(np.mean(predictions, axis=0) > 0.5, 1, 0)

    def predict_proba(self, X):
        predictions = [model.predict_proba(X) for model in self.models]
        return np.mean(predictions, axis=0)
