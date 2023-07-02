from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lgbm_autotune.model.bagging_models import BaggingModels


class Objective:
    "optuna tuning and stratified k-fold"

    def __init__(self, X, y, model_registry):
        self.X = X
        self.y = y
        self.model_registry = model_registry

    def __call__(self, trial):
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

        # 5-Fold Stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        # 各分割でのスコアを保存するリスト
        scores = []
        models = []

        for train_index, valid_index in skf.split(self.X, self.y):
            model = LGBMClassifier(**all_params)

            # データの分割
            X_train, X_valid = (
                self.X.iloc[train_index],
                self.X.iloc[valid_index],
            )
            y_train, y_valid = (
                self.y.iloc[train_index],
                self.y.iloc[valid_index],
            )

            model.fit(X_train, y_train, eval_metric="auc")
            models.append(model)

            # モデルの評価
            y_pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, y_pred)

            # スコアの保存
            scores.append(score)

        bagging_model = BaggingModels(models=models)
        self.model_registry.register(trial.number, bagging_model)

        return sum(scores) / len(scores)
