class StudyCallback:
    def __init__(self, model_registry):
        self._model_registry = model_registry

    def __call__(self, study, trial):
        if study.best_trial.number == trial.number:
            self._model_registry.save_best_model(study.best_trial.number)
