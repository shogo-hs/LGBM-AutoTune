import os

import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lgbm_autotune.model.model_registry import ModelRegistry
from lgbm_autotune.model.objective import Objective
from lgbm_autotune.model.study_callback import StudyCallback

if __name__ == "__main__":
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]
    label = "Survived"
    study_name = "test"

    train_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../data/train.csv")
    )
    test_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../data/test.csv")
    )

    for f in features:
        if train_df[f].dtype == "object":
            le = LabelEncoder()
            train_df[f] = le.fit_transform(train_df[f])
            test_df[f] = le.transform(test_df[f])

    X_train, X_test, y_train, y_test = train_test_split(
        train_df[features], train_df[label]
    )

    # initialize model registry and callback
    model_registry = ModelRegistry(study_name=study_name)
    callback = StudyCallback(model_registry)

    # Your objective function should call model_registry.register() for each trial
    objective = Objective(X_train, y_train, model_registry)

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_study.db",
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10, callbacks=[callback])

    best_model = model_registry.retrieve(study.best_trial.number)

    # y_pred = best_model.predict_proba(test_df[features])[:, 1]
    # print(y_pred)
