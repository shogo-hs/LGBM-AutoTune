import os

from joblib import dump, load


class ModelRegistry:
    def __init__(self, study_name):
        self._models = {}
        save_model_dir = os.path.join(
            os.path.dirname(__file__), f".{study_name}_best_model/"
        )
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)
        self.best_model_file = os.path.join(save_model_dir, "model.joblib")

    def register(self, trial_id, model):
        self._models[trial_id] = model

    def retrieve(self, trial_id):
        return self._models.get(trial_id)

    def save_best_model(self, best_trial_id):
        best_model = self.retrieve(best_trial_id)
        if best_model is not None:
            dump(best_model, self.best_model_file)

    def load_best_model(self):
        if os.path.exists(self.best_model_file):
            return load(self.best_model_file)
        else:
            return None
