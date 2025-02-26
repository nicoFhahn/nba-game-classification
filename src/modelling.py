from datetime import date, timedelta, datetime
from math import floor, ceil
from functools import partial

import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import os
import polars as pl
import numpy as np
import lightgbm as lgbm
import json
import shap
import gc
import glob
import joblib

class lgbm_model():
    def __init__(
            self,
            data_folder: str = 'data',
            model_folder: str = 'models',
            random_state: int = 7918
    ):
        self.data_folder = data_folder
        self.model_folder = model_folder
        self.random_state = random_state
        self.full_data = None

    def accuracy_metric(y_true, y_pred):
        # LightGBM expects the custom metric to return two values:
        # (eval_name, eval_result, is_higher_better)
        accuracy = accuracy_score(y_true, np.round(y_pred))
        return 'accuracy', accuracy, True

    def load_data(
            self,
            train_size: float = 0.85,
            remove_size: float = 0.05
    ) -> pl.DataFrame:
        """

        :param train_size:
        :param remove_size:
        :return:
        """
        self.full_data = pl.read_parquet(
            os.path.join(self.data_folder, 'season_df.parquet')
        ).sort('date')
        start_date = date.today().replace(day=1)
        train_start = self.full_data[
            :floor(
                self.full_data.filter(pl.col('date') < start_date).shape[0] * remove_size
            )
        ]['date'].max()
        train_end = self.full_data[
            :floor(
                self.full_data.filter(
                    (pl.col('date') > train_start) & (pl.col('date') < start_date)
                ).shape[0] * train_size
            )
        ]['date'].max()
        val_start = self.full_data.filter(
            (pl.col('date') > train_end) &
            (pl.col('date') < start_date)
        )[:ceil(
            self.full_data.filter(
                (pl.col('date') > train_end) &
                (pl.col('date') < start_date)
            ).shape[0] * 0.5)]['date'].max()
        val_end = start_date + timedelta(days=-1)
        df = self.full_data.to_dummies([
            'game_type', 'month', 'weekday'
        ]).drop([
            'home_team_id', 'away_team_id'
        ])
        train = df.filter(
            (pl.col('date') >= train_start) &
            (pl.col('date') < train_end)
        ).sort(['date', 'game_id'])
        test = df.filter(
            (pl.col('date') >= train_end) &
            (pl.col('date') < val_start)
        ).drop_nulls('is_home_win').sort(['date', 'game_id'])
        val = df.filter(
            (pl.col('date') >= val_start) &
            (pl.col('date') < val_end)
        ).drop_nulls('is_home_win').sort(['date', 'game_id'])
        y_train = train.select('is_home_win')
        y_test = test.select('is_home_win')
        y_val = val.select('is_home_win')
        X_train = train.drop(['is_home_win', 'game_id'])
        X_test = test.drop(['is_home_win', 'game_id'])
        X_val = val.drop(['is_home_win', 'game_id'])
        self.data_sets = {
            'train': train,
            'test': test,
            'val': val
        }
        self.training_timestamps = {
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end
        }
        self.X = {
            'train': X_train,
            'test': X_test,
            'val': X_val
        }
        self.y = {
            'train': y_train,
            'test': y_test,
            'val': y_val
        }

    def feature_selection(
            self,
            n_folds: int = 5,
            use_shapley: bool = True
    ):
        """

        :param n_folds:
        :param use_shapley:
        :return:
        """
        split = TimeSeriesSplit(n_folds)
        filename = f'{self.model_folder}/lgbm_evaluation_{self.training_timestamps['val_end'].strftime('%B')}_{self.training_timestamps['val_end'].year}2.json'.lower()
        if filename in os.listdir(self.model_folder):
            with open(filename, 'r') as f:
                d = json.loads(f.read())
                precision_list = d['train']['precision']
                recall_list = d['train']['recall']
                accuracy_list = d['train']['accuracy']
                f1_list = d['train']['f1']
                feature_list_1 = d['features']['trained_on']
                feature_list_2 = d['features']['next_trained_on']
                X_train = self.X['train'][feature_list_2[-1]]
        else:
            precision_list = []
            recall_list = []
            accuracy_list = []
            f1_list = []
            feature_list_1 = []
            feature_list_2 = []
            X_train = self.X['train']
        y_train = self.y['train']
        while X_train.shape[0] > 1:
            feature_list_1.append(X_train.columns)
            if X_train.shape[1] % 50 == 0 or X_train.shape[1] < 5:
                print(f'Running Feature Selection for {X_train.shape[1]} features.')
            importance_list = []
            temp_precision_list = []
            temp_recall_list = []
            temp_accuracy_list = []
            temp_f1_list = []
            for train_idx, test_idx in split.split(X_train, y_train):
                # Split the data into train and test sets
                temp_X_train, temp_X_test = X_train[train_idx], X_train[test_idx]
                temp_y_train, temp_y_test = y_train[train_idx], y_train[test_idx]
                sample_weights = temp_X_train.with_columns([
                    (
                            1 + (
                            pl.col('date') - pl.col('date').min()
                    ).dt.total_days() / (
                        (
                                pl.col('date').max() - pl.col('date').min()
                        ).dt.total_days())
                    ).alias('sample_weight')
                ]).select(pl.col('sample_weight')).to_numpy().ravel()
                temp_X_train = temp_X_train.drop('date')
                temp_X_test = temp_X_test.drop('date')
                temp_X_train = temp_X_train.to_numpy()
                temp_X_test = temp_X_test.to_numpy()
                temp_y_train = temp_y_train.to_numpy().ravel()
                temp_y_test = temp_y_test.to_numpy().ravel()
                dtrain = lgbm.Dataset(
                    data=temp_X_train,
                    label=temp_y_train
                )
                mod = lgbm.train(
                    params={
                        'objective': 'binary',
                        'metric': None,
                        'first_metric_only': True,
                        'verbose': -1,
                        'random_state': self.random_state,
                        'sample_weights': sample_weights
                    },
                    train_set=dtrain,
                    feval=self.accuracy_metric
                )
                # Calculate SHAP values on the test set for the current fold
                if use_shapley:
                    explainer = shap.Explainer(mod)
                    shap_values = explainer.shap_values(temp_X_test)
                    importance_list.append(shap_values)
                else:
                    # Store the SHAP values for this fold
                    importance_list.append(mod.feature_importances_)
                temp_precision_list.append(
                    precision_score(
                        temp_y_test, mod.predict(temp_X_test) > 0.5
                    )
                )
                temp_recall_list.append(
                    recall_score(
                        temp_y_test, mod.predict(temp_X_test) > 0.5
                    )
                )
                temp_accuracy_list.append(
                    accuracy_score(
                        temp_y_test, mod.predict(temp_X_test) > 0.5
                    )
                )
                temp_f1_list.append(
                    f1_score(
                        temp_y_test, mod.predict(temp_X_test) > 0.5
                    )
                )

            abs_importance_values = np.absolute(np.mean(np.vstack(importance_list), axis=0))
            drop_cols = np.array(X_train.drop('date').columns)[
                np.where(abs_importance_values == abs_importance_values.min())
            ]
            precision_list.append(np.mean(temp_precision_list))
            recall_list.append(np.mean(temp_recall_list))
            accuracy_list.append(np.mean(temp_accuracy_list))
            f1_list.append(np.mean(temp_f1_list))
            X_train = X_train.drop(drop_cols)
            feature_list_2.append(X_train.columns)
            eval_dict = {
                'performance': {
                    'precision': precision_list,
                    'recall': recall_list,
                    'accuracy': accuracy_list,
                    'f1': f1_list
                },
                'features': {
                    'trained_on': feature_list_1,
                    'next_trained_on': feature_list_2
                },
                'last_train_date': (self.training_timestamps['val_start'] + timedelta(days=-1)).isoformat()
            }
            with open(filename, 'w') as f:
                json.dump(eval_dict, f)
            del mod
            if use_shapley:
                del explainer
            gc.collect()
        print('Feature selection completed.')

    def objective_lgbm(
            self,
            trial,
            X_train: pl.DataFrame,
            y_train: pl.Series,
            X_test: pl.DataFrame,
            y_test: pl.Series,
            random_state: int
    ):
        # Define the hyperparameters to tune
        params = {
            'n_estimators': trial.suggest_int('iterations', 50, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10),
            'subsample': trial.suggest_float('subsample', 0.01, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1),
            'random_state': random_state,
            'verbose': -1,
            'metric': None,
            'objective': 'binary',
            'first_metric_only': True
        }
        sample_weights = X_train.with_columns([
            (1 + (pl.col('date') - pl.col('date').min()).dt.total_days() / (
                (pl.col('date').max() - pl.col('date').min()).dt.total_days())).alias('sample_weight')
        ]).select(pl.col('sample_weight')).to_numpy().ravel()
        params['sample_weights'] = sample_weights
        dtrain = lgbm.Dataset(
            data=X_train.drop('date').to_numpy(),
            label=y_train.to_numpy().ravel()
        )
        mod = lgbm.train(
            params=params,
            train_set=dtrain,
            feval=self.accuracy_metric
        )
        predictions = mod.predict(X_test.drop('date').to_numpy())
        accuracy = accuracy_score(y_test.to_numpy().ravel(), predictions >= 0.5)

        return accuracy

    def tune_hyperparameters(self):
        study = optuna.create_study(direction='maximize')
        best_features = self.load_best_features()[0]
        obj = partial(
            self.objective_lgbm,
            X_train=self.X['train'].select(best_features),
            y_train=self.y['train'],
            X_test=self.X['test'].select(best_features),
            y_test=self.y['test'],
            random_state=self.random_state
        )
        study.optimize(obj, n_trials=200)
        filename = f'{self.model_folder}/params_{self.training_timestamps['val_end'].strftime('%B')}_{self.training_timestamps['val_end'].year}.json'.lower()
        best_params = study.best_params
        best_params['verbose'] = -1
        best_params['metric'] = None
        best_params['objective'] = 'binary'
        best_params['random_state'] = self.random_state
        best_params['first_metric_only'] = True
        with open(filename, 'w') as f:
            json.dump(best_params, f)
        print('Hyperparameter tuning completed.')

    def load_best_features(self):
        eval_files = glob.glob(f'{self.model_folder}/lgbm*.json')
        with open(max(eval_files, key=os.path.getmtime), 'r') as f:
            evaluation = json.loads(f.read())
        best_features = evaluation['features']['trained_on'][
            evaluation['performance']['accuracy'].index(max(evaluation['performance']['accuracy']))
        ]
        last_train_date = datetime.strptime(evaluation['last_train_date'], '%Y-%m-%d').date()
        return best_features, last_train_date

    def load_best_params(self):
        param_files = glob.glob(f'{self.model_folder}/params*.json')
        with open(max(param_files, key=os.path.getmtime), 'r') as f:
            params = json.loads(f.read())
        return params

    def train_final_model(self):
        best_features = self.load_best_features()[0]
        best_params = self.load_best_params()
        X_train = pl.concat([
            self.X['train'], self.X['test']
        ])
        y_train = pl.concat([
            self.y['train'], self.y['test']
        ])
        sample_weights = X_train.with_columns([
            (1 + (
                    pl.col('date') - pl.col('date').min()
            ).dt.total_days() / (
                (
                        pl.col('date').max() - pl.col('date').min()
                ).dt.total_days()
            )).alias('sample_weight')
        ]).select('sample_weight').to_numpy().ravel()
        best_params['sample_weights'] = sample_weights
        X_train = X_train.select(best_features).drop('date')
        dtrain = lgbm.Dataset(
            data=X_train.to_numpy(),
            label=y_train.to_numpy().ravel()
        )
        mod = lgbm.train(
            params=best_params,
            train_set=dtrain,
            feval=self.accuracy_metric
        )
        self.model = mod
        joblib.dump(mod, f'{self.model_folder}/lgbm_model.pkl')
        print('Final model trained and saved')

    def load_model(self):
        pkl_files = glob.glob(f'{self.model_folder}/*.pkl')
        mod = joblib.load(max(pkl_files, key=os.path.getmtime))
        self.model = mod

    def predict(self):
        best_features, cutoff_date = self.load_best_features()
        if 'predictions.parquet' in os.listdir(self.data_folder):
            previous_predictions = pl.read_parquet(
                os.path.join(self.data_folder, 'predictions.parquet')
            )
            cutoff_date = self.full_data.join(
                previous_predictions,
                on='game_id',
                how='inner'
            )['date'].max()
        X_new = self.full_data.filter(
            pl.col('date') > cutoff_date
        ).to_dummies([
            'game_type', 'month', 'weekday'
        ])
        game_ids = X_new['game_id']
        X_new = X_new.select(best_features).drop('date')
        predictions = self.model.predict(X_new.to_numpy())
        prediction_df = pl.DataFrame({
            'game_id': game_ids,
            'probability': predictions,
            'is_home_win': predictions >= 0.5
        })
        if 'predictions.parquet' in os.listdir(self.data_folder):
            prediction_df = pl.concat([previous_predictions, prediction_df])
        prediction_df.write_parquet(
            os.path.join(self.data_folder, 'predictions.parquet')
        )

    def evaluate_performance(self):
        predictions = pl.read_parquet(
            os.path.join(self.data_folder, 'predictions.parquet')
        )
        if self.full_data is None:
            self.load_data()
        prediction_df = self.full_data.select(pl.col(["date", "game_id", "home_team_id", "away_team_id", "is_home_win"])).join(
            predictions,
            on="game_id"
        ).rename({
            "is_home_win_right": "is_predicted_home_win"
        })
        performance_df = prediction_df.group_by("date").agg([
            ((pl.col("is_home_win") == pl.col("is_predicted_home_win")) & (pl.col("is_home_win"))).sum().alias(
                "true_positives"),
            ((pl.col("is_home_win") == pl.col("is_predicted_home_win")) & (~pl.col("is_home_win"))).sum().alias(
                "true_negatives"),
            ((pl.col("is_home_win") != pl.col("is_predicted_home_win")) & (pl.col("is_home_win"))).sum().alias(
                "false_positives"),
            ((pl.col("is_home_win") != pl.col("is_predicted_home_win")) & (~pl.col("is_home_win"))).sum().alias(
                "false_negatives")
        ]).with_columns([
            pl.col("true_positives").cum_sum(),
            pl.col("true_negatives").cum_sum(),
            pl.col("false_positives").cum_sum(),
            pl.col("false_negatives").cum_sum()
        ]).with_columns([
            pl.col("true_positives").rolling_sum(window_size=14, min_samples=1).alias('true_positives_rolling'),
            pl.col("true_negatives").rolling_sum(window_size=14, min_samples=1).alias('true_negatives_rolling'),
            pl.col("false_positives").rolling_sum(window_size=14, min_samples=1).alias('false_positives_rolling'),
            pl.col("false_negatives").rolling_sum(window_size=14, min_samples=1).alias('false_negatives_rolling'),
            (
                    (
                            pl.col("true_positives") + pl.col("true_negatives")
                    ) /
                    (
                            pl.col("true_positives") + pl.col("true_negatives") +
                            pl.col("false_positives") + pl.col("false_negatives")
                    )
            ).alias("accuracy"),
            (
                    pl.col("true_positives") /
                    (
                            pl.col("true_positives") + pl.col("false_positives")
                    )
            ).alias("precision"),
            (
                    pl.col("true_positives") /
                    (
                            pl.col("true_positives") + pl.col("false_negatives")
                    )
            ).alias("recall")
        ]).with_columns([
            (
                    2 * pl.col("precision") * pl.col("recall") /
                    (
                            pl.col("precision") + pl.col("recall")
                    )
            ).alias("f1_score"),
            (
                    (
                            pl.col("true_positives_rolling") + pl.col("true_negatives_rolling")
                    ) /
                    (
                            pl.col("true_positives_rolling") + pl.col("true_negatives_rolling") +
                            pl.col("false_positives_rolling") + pl.col("false_negatives_rolling")
                    )
            ).alias("accuracy_rolling"),
            (
                    pl.col("true_positives_rolling") /
                    (
                            pl.col("true_positives_rolling") + pl.col("false_positives_rolling")
                    )
            ).alias("precision_rolling"),
            (
                    pl.col("true_positives_rolling") /
                    (
                            pl.col("true_positives_rolling") + pl.col("false_negatives_rolling")
                    )
            ).alias("recall_rolling")
        ]).with_columns([
            (
                    2 * pl.col("precision_rolling") * pl.col("recall_rolling") /
                    (
                            pl.col("precision_rolling") + pl.col("recall_rolling")
                    )
            ).alias("f1_score_rolling")
        ])

        def team_performance(df):
            return pl.DataFrame({
                "true_positives": df.filter(pl.col("is_home_win") & (pl.col("is_predicted_home_win"))).shape[0],
                "true_negatives": df.filter(~pl.col("is_home_win") & (~pl.col("is_predicted_home_win"))).shape[0],
                "false_positives": df.filter(~pl.col("is_home_win") & (pl.col("is_predicted_home_win"))).shape[0],
                "false_negatives": df.filter(pl.col("is_home_win") & (~pl.col("is_predicted_home_win"))).shape[0]
            })

        team_ids = list(set(prediction_df["home_team_id"]))
        performance_by_team = [
            team_performance(
                prediction_df.filter((pl.col("home_team_id") == team_id) | (pl.col("away_team_id") == team_id))
            ) for team_id in team_ids
        ]
        performance_by_team_df = pl.concat(performance_by_team).with_columns([
            pl.Series("team_id", team_ids)
        ]).with_columns([
            (
                (
                    pl.col("true_positives") + pl.col("true_negatives")
                ) /
                (
                    pl.col("true_positives") + pl.col("true_negatives") +
                    pl.col("false_positives") + pl.col("false_negatives")
                )
            ).alias("accuracy"),
            (
                pl.col("true_positives") /
                (
                    pl.col("true_positives") + pl.col("false_positives")
                )
            ).alias("precision"),
            (
                pl.col("true_positives") /
                (
                    pl.col("true_positives") + pl.col("false_negatives")
                )
            ).alias("recall")
        ]).with_columns([
            (
                2 * pl.col("precision") * pl.col("recall") /
                (
                    pl.col("precision") + pl.col("recall")
                )
            ).alias("f1_score")
        ])

        self.performance = {
            "over_time": performance_df,
            "by_team": performance_by_team_df
        }