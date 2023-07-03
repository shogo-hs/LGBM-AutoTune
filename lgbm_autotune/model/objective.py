from typing import Any, List, Tuple

from lightgbm import LGBMClassifier
from optuna import Trial
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lgbm_autotune.model.bagging_models import BaggingModels
from lgbm_autotune.model.model_registry import ModelRegistry


def select_features(trial: Trial, features: List[str]) -> List[str]:
    """
    Selects features for model training based on the trial's suggestions.

    Parameters:
    trial (optuna.Trial): The trial object.
    features (List[str]): The list of all available features.

    Returns:
    List[str]: The list of selected features.
    """
    selected_features = [trial.suggest_categorical(f, [0, 1]) for f in features]
    subset_features = [f for f, sf in zip(features, selected_features) if sf == 1]
    return subset_features


def train_and_evaluate_model(
    train_index: List[int],
    valid_index: List[int],
    X: DataFrame,
    y: DataFrame,
    subset_features: List[str],
    params: dict,
) -> Tuple[float, LGBMClassifier]:
    """
    Trains and evaluates the model using the specified indices for training and validation,
    the subset of features, and the parameters for the model.

    Parameters:
    train_index (List[int]): The indices for the training data.
    valid_index (List[int]): The indices for the validation data.
    X (DataFrame): The feature data.
    y (DataFrame): The target data.
    subset_features (List[str]): The list of selected features.
    params (dict): The parameters for the model.

    Returns:
    Tuple[float, LGBMClassifier]: The ROC AUC score and the trained model.
    """
    model = LGBMClassifier(**params)

    X_train, X_valid = (
        X.iloc[train_index][subset_features],
        X.iloc[valid_index][subset_features],
    )
    y_train, y_valid = (
        y.iloc[train_index],
        y.iloc[valid_index],
    )

    model.fit(X_train, y_train, eval_metric="auc")

    y_pred = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, y_pred)

    return score, model


class Objective:
    """
    Optuna tuning, stratified k-fold and feature selection.

    This class defines the objective function for the optimization, performing feature selection
    and stratified k-fold cross-validation. The objective is used in an Optuna study to guide the
    search for optimal hyperparameters.

    Attributes:
    X (DataFrame): The feature data.
    y (DataFrame): The target data.
    model_registry (ModelRegistry): The model registry for storing the models.
    """

    def __init__(self, X: DataFrame, y: DataFrame, model_registry: ModelRegistry):
        """
        Initialize the Objective with feature data, target data and a model registry.

        Parameters:
        X (DataFrame): The feature data.
        y (DataFrame): The target data.
        model_registry (ModelRegistry): The model registry for storing the models.
        """
        self.X = X
        self.y = y
        self.model_registry = model_registry

    def __call__(self, trial: Trial) -> float:
        """
        Define the objective function for the Optuna trial.

        This function performs feature selection, trains a model using stratified k-fold
        cross-validation, and returns the average score across all folds.

        Parameters:
        trial (Trial): The Optuna trial.

        Returns:
        float: The average score across all folds.
        """
        static_params = {
            "random_state": 123,
            "boosting_type": "gbdt",
            "objective": "binary",
        }

        tune_params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 10000),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 200),
            "max_depth": trial.suggest_int("max_depth", 1, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        }

        all_params = {
            **tune_params,
            **static_params,
        }

        trial.set_user_attr("all_params", all_params)

        # Feature selection
        subset_features = select_features(trial, list(self.X.columns))

        # 5-Fold Stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        scores = []
        models = []

        for train_index, valid_index in skf.split(self.X, self.y):
            score, model = train_and_evaluate_model(
                train_index, valid_index, self.X, self.y, subset_features, all_params
            )
            scores.append(score)
            models.append(model)

        bagging_model = BaggingModels(models=models)
        self.model_registry.register(trial.number, bagging_model)

        return sum(scores) / len(scores)
