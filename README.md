# LGBM-AutoTune

LGBM-AutoTune is a Python library designed to simplify and streamline the process of hyperparameter tuning and model management with LightGBM and Optuna. With utilities for managing the model training process and model registry, LGBM-AutoTune allows users to tune, train, and retrieve their best models with ease.

## Features
- __Automatic Hyperparameter Tuning:__ Use Optuna's efficient optimization algorithms to find the best hyperparameters for your LightGBM model.
- __Model Registry:__ Keep track of your models with a built-in model registry, and save and load models effortlessly.
- __Bagging Models:__ Train multiple models with different hyperparameters and average their predictions with the bagging models utility.

## Usage
Below is a basic usage example:

```python
from lgbm_autotune.model.model_registry import ModelRegistry
from lgbm_autotune.model.objective import Objective
from lgbm_autotune.model.study_callback import StudyCallback
from optuna import create_study

study_name="hogehoge"

# initialize model registry and callback
model_registry = ModelRegistry(study_name=study_name)
callback = StudyCallback(model_registry)

# Your objective function should call model_registry.register() for each trial
objective = Objective(X_train, y_train, model_registry)

study = create_study(
    direction="maximize",
    storage="sqlite:///optuna_study.db",
    study_name=study_name,
    load_if_exists=True,
)
study.optimize(objective, n_trials=10, callbacks=[callback])

# Load the best model from the file after all trials
best_model = model_registry.load_best_model()

# ... proceed with best_model as usual ...
For more detailed examples and usage, please refer to the Documentation.

```
