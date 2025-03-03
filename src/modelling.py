import os
import json
import gc
import fnmatch
import sys
import warnings
import pathlib
from datetime import date, timedelta, datetime
from math import floor, ceil
from functools import partial
from google.cloud import storage

import numpy as np
import polars as pl
import optuna
import lightgbm as lgbm
import joblib
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from supabase import Client
sys.path.append(os.path.join(os.path.abspath(__file__), '..'))
from data_collection import collect_all_data
warnings.filterwarnings("ignore")

class lgbm_model():
    def __init__(
            self,
            connection: Client,
            bucket: storage.Bucket,
            random_state: int = 7918,
            data_origin: str = "google"
    ):
        self.connection = connection
        self.random_state = random_state
        self.bucket = bucket
        self.data_origin = data_origin
        self.load_data()

    def accuracy_metric(y_true, y_pred):
        # LightGBM expects the custom metric to return two values:
        # (eval_name, eval_result, is_higher_better)
        accuracy = accuracy_score(y_true, np.round(y_pred))
        return 'accuracy', accuracy, True

    def load_data(
            self,
            train_size: float = 0.85,
            remove_size: float = 0.05
    ):
        '''

        :param train_size:
        :param remove_size:
        :return:
        '''
        if self.data_origin == 'google':
            blob = self.bucket.blob('final_data.parquet')
            blob.download_to_filename('final_data.parquet')
            self.full_data = pl.read_parquet('final_data.parquet')
            pathlib.Path.unlink('final_data.parquet')
        elif self.data_origin == 'supabase':
            df_1 = collect_all_data(
                'schedule',
                self.connection
            )
            df_2 = collect_all_data(
                'elo',
                self.connection
            ).drop('id')
            df_3 = collect_all_data(
                'statistics_previous',
                self.connection
            )
            df_4 = collect_all_data(
                'statistics_recent_games',
                self.connection
            )
            df_5 = collect_all_data(
                'statistics_season',
                self.connection
            )
            df_6 = collect_all_data(
                'statistics_remainder',
                self.connection
            )
            current_elo = df_2.with_columns(pl.col('date').str.to_date()).group_by('team_id').tail(1).select(
                ['team_id', 'elo_after'])
            temp_df = df_1.join(
                df_3, on='game_id'
            ).join(
                df_4, on='game_id'
            ).join(
                df_5, on='game_id'
            ).join(
                df_6, on='game_id'
            ).join(
                df_2.drop(['elo_after', 'date']),
                left_on=['game_id', 'home_team_id'],
                right_on=['game_id', 'team_id'],
                how='left'
            ).join(
                df_2.drop(['elo_after', 'date']),
                left_on=['game_id', 'away_team_id'],
                right_on=['game_id', 'team_id'],
                how='left'
            ).join(
                current_elo,
                left_on='home_team_id',
                right_on='team_id'
            ).join(
                current_elo,
                left_on='away_team_id',
                right_on='team_id'
            ).with_columns([
                pl.coalesce(pl.col('elo_before'), pl.col('elo_after')).alias('elo_home_team'),
                pl.coalesce(pl.col('elo_before_right'), pl.col('elo_after_right')).alias('elo_away_team'),
                pl.col('date').str.to_date()
            ]).drop([
                'elo_before', 'elo_before_right', 'elo_after', 'elo_after_right', 'season_id'
            ])
            self.full_data = temp_df.sort('date')
            temp_df.write_parquet('final_data.parquet')
            blob = self.bucket.blob('final_data.parquet')
            blob.upload_from_filename('final_data.parquet')
            pathlib.Path.unlink('final_data.parquet')
        start_date = date.today().replace(day=1)
        self.full_data = self.full_data.sort('date')
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
        '''

        :param n_folds:
        :param use_shapley:
        :return:
        '''
        split = TimeSeriesSplit(n_folds)
        filename = f'lgbm_evaluation_{self.training_timestamps['val_end'].strftime('%B')}_{self.training_timestamps['val_end'].year}.json'.lower()
        last_train_date = (self.training_timestamps['val_start'] + timedelta(days=-1)).isoformat()
        if filename in [b.name for b in self.bucket.list_blobs()]:
            blob = self.bucket.blob(filename)
            blob.download_to_filename(filename)
            with open(filename, 'r') as f:
                d = json.loads(f.read())
                precision = d['performance']['precision']
                recall = d['performance']['recall']
                accuracy = d['performance']['accuracy']
                f1 = d['performance']['f1']
                feature_list_2 = d['features']['next_trained_on']
                X_train = self.X['train'][feature_list_2]
                accuracy_features = d['features']['accuracy']
                precision_features = d['features']['precision']
                recall_features = d['features']['recall']
                f1_features = d['features']['f1']
                training_complete = X_train.shape[1] == 1
        else:
            blob = self.bucket.blob(filename)
            precision = 0
            recall = 0
            accuracy = 0
            f1 = 0
            X_train = self.X['train']
            accuracy_features = []
            precision_features = []
            recall_features = []
            f1_features = []
            training_complete = False
        y_train = self.y['train']
        if not training_complete:
            while X_train.shape[1] > 1:
                feature_list_1 = X_train.columns
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
                cv_accuracy = np.mean(temp_accuracy_list)
                if cv_accuracy > accuracy:
                    accuracy = cv_accuracy
                    accuracy_features = X_train.columns
                cv_precision = np.mean(temp_precision_list)
                if cv_precision > precision:
                    precision = cv_precision
                    precision_features = X_train.columns
                cv_recall = np.mean(temp_recall_list)
                if cv_recall > accuracy:
                    recall = cv_recall
                    recall_features = X_train.columns
                cv_f1 = np.mean(temp_f1_list)
                if cv_f1 > f1:
                    f1 = cv_f1
                    f1_features = X_train.columns
                X_train = X_train.drop(drop_cols)
                feature_list_2 = X_train.columns
                del mod
                if use_shapley:
                    del explainer
                gc.collect()
            eval_dict = {
                'performance': {
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'f1': f1
                },
                'features': {
                    'trained_on': feature_list_1,
                    'next_trained_on': feature_list_2,
                    'accuracy': accuracy_features,
                    'precision': precision_features,
                    'recall': recall_features,
                    'f1': f1_features
                },
                'last_train_date': last_train_date
            }
            with open(filename, 'w') as f:
                json.dump(eval_dict, f)
            blob.upload_from_filename(filename)
            pathlib.Path.unlink(filename)
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
        filename = f'params_{self.training_timestamps['val_end'].strftime('%B')}_{self.training_timestamps['val_end'].year}.json'.lower()
        best_params = study.best_params
        best_params['verbose'] = -1
        best_params['metric'] = None
        best_params['objective'] = 'binary'
        best_params['random_state'] = self.random_state
        best_params['first_metric_only'] = True
        with open(filename, 'w') as f:
            json.dump(best_params, f)
        blob = self.bucket.blob(filename)
        blob.upload_from_filename(filename)
        pathlib.Path.unlink(filename)
        print('Hyperparameter tuning completed.')

    def load_best_features(self):
        filename = download_newest_matching_json_file("lgbm", "", "lgbm*.json")
        with open(filename, 'r') as f:
            evaluation = json.loads(f.read())
        pathlib.Path.unlink(filename)
        best_features = evaluation['features']['accuracy']
        last_train_date = datetime.strptime(evaluation['last_train_date'], '%Y-%m-%d').date()
        return best_features, last_train_date

    def load_best_params(self):
        filename = download_newest_matching_json_file("lgbm", "", "params*.json")
        with open(filename, 'r') as f:
            params = json.loads(f.read())
        pathlib.Path.unlink(filename)
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
        filename = 'lgbm_model.pkl'
        joblib.dump(mod, filename)
        blob = self.bucket.blob(filename)
        blob.upload_from_filename(filename)
        pathlib.Path.unlink(filename)
        print('Final model trained and saved')

    def load_model(self):
        filename = 'lgbm_model.pkl'
        blob = self.bucket.blob(filename)
        blob.download_to_filename(filename)
        mod = joblib.load(filename)
        pathlib.Path.unlink(filename)
        self.model = mod

    def predict(self):
        best_features, cutoff_date = self.load_best_features()
        previous_predictions = collect_all_data('predictions', self.connection)
        cutoff_date = self.full_data.join(
            previous_predictions,
            on='game_id',
            how='inner'
        )['date'].max()
        X_new = self.full_data.to_dummies([
            'game_type', 'month', 'weekday'
        ]).filter(
            (pl.col('date') >= cutoff_date) &
            (pl.col('is_home_win').is_not_null()) &
            (~pl.col('game_id').is_in(previous_predictions['game_id']))
        )
        game_ids = X_new['game_id']
        X_new = X_new.select(best_features).drop('date')
        predictions = self.model.predict(X_new.to_numpy())
        prediction_df = pl.DataFrame({
            'game_id': game_ids,
            'probability': predictions,
            'is_home_win': predictions >= 0.5
        })
        print(f"{prediction_df.shape[0]} predictions added")
        response = self.connection.table('predictions').insert(prediction_df.to_dicts()).execute()

    def evaluate_performance(self):
        predictions = collect_all_data('predictions', self.connection)
        if self.full_data is None:
            self.load_data()
        prediction_df = self.full_data.select(pl.col(['date', 'game_id', 'home_team_id', 'away_team_id', 'is_home_win'])).join(
            predictions,
            on='game_id'
        ).rename({
            'is_home_win_right': 'is_predicted_home_win'
        })
        performance_df = prediction_df.group_by('date').agg([
            ((pl.col('is_home_win') == pl.col('is_predicted_home_win')) & (pl.col('is_home_win'))).sum().alias(
                'true_positives'),
            ((pl.col('is_home_win') == pl.col('is_predicted_home_win')) & (~pl.col('is_home_win'))).sum().alias(
                'true_negatives'),
            ((pl.col('is_home_win') != pl.col('is_predicted_home_win')) & (pl.col('is_home_win'))).sum().alias(
                'false_positives'),
            ((pl.col('is_home_win') != pl.col('is_predicted_home_win')) & (~pl.col('is_home_win'))).sum().alias(
                'false_negatives')
        ]).with_columns([
            pl.col('true_positives').cum_sum(),
            pl.col('true_negatives').cum_sum(),
            pl.col('false_positives').cum_sum(),
            pl.col('false_negatives').cum_sum()
        ]).with_columns([
            pl.col('true_positives').rolling_sum(window_size=14, min_samples=1).alias('true_positives_rolling'),
            pl.col('true_negatives').rolling_sum(window_size=14, min_samples=1).alias('true_negatives_rolling'),
            pl.col('false_positives').rolling_sum(window_size=14, min_samples=1).alias('false_positives_rolling'),
            pl.col('false_negatives').rolling_sum(window_size=14, min_samples=1).alias('false_negatives_rolling'),
            (
                    (
                            pl.col('true_positives') + pl.col('true_negatives')
                    ) /
                    (
                            pl.col('true_positives') + pl.col('true_negatives') +
                            pl.col('false_positives') + pl.col('false_negatives')
                    )
            ).alias('accuracy'),
            (
                    pl.col('true_positives') /
                    (
                            pl.col('true_positives') + pl.col('false_positives')
                    )
            ).alias('precision'),
            (
                    pl.col('true_positives') /
                    (
                            pl.col('true_positives') + pl.col('false_negatives')
                    )
            ).alias('recall')
        ]).with_columns([
            (
                    2 * pl.col('precision') * pl.col('recall') /
                    (
                            pl.col('precision') + pl.col('recall')
                    )
            ).alias('f1_score'),
            (
                    (
                            pl.col('true_positives_rolling') + pl.col('true_negatives_rolling')
                    ) /
                    (
                            pl.col('true_positives_rolling') + pl.col('true_negatives_rolling') +
                            pl.col('false_positives_rolling') + pl.col('false_negatives_rolling')
                    )
            ).alias('accuracy_rolling'),
            (
                    pl.col('true_positives_rolling') /
                    (
                            pl.col('true_positives_rolling') + pl.col('false_positives_rolling')
                    )
            ).alias('precision_rolling'),
            (
                    pl.col('true_positives_rolling') /
                    (
                            pl.col('true_positives_rolling') + pl.col('false_negatives_rolling')
                    )
            ).alias('recall_rolling')
        ]).with_columns([
            (
                    2 * pl.col('precision_rolling') * pl.col('recall_rolling') /
                    (
                            pl.col('precision_rolling') + pl.col('recall_rolling')
                    )
            ).alias('f1_score_rolling')
        ])

        def team_performance(df):
            return pl.DataFrame({
                'true_positives': df.filter(pl.col('is_home_win') & (pl.col('is_predicted_home_win'))).shape[0],
                'true_negatives': df.filter(~pl.col('is_home_win') & (~pl.col('is_predicted_home_win'))).shape[0],
                'false_positives': df.filter(~pl.col('is_home_win') & (pl.col('is_predicted_home_win'))).shape[0],
                'false_negatives': df.filter(pl.col('is_home_win') & (~pl.col('is_predicted_home_win'))).shape[0]
            })

        team_ids = list(set(prediction_df['home_team_id']))
        performance_by_team = [
            team_performance(
                prediction_df.filter((pl.col('home_team_id') == team_id) | (pl.col('away_team_id') == team_id))
            ) for team_id in team_ids
        ]
        performance_by_team_df = pl.concat(performance_by_team).with_columns([
            pl.Series('team_id', team_ids)
        ]).with_columns([
            (
                (
                    pl.col('true_positives') + pl.col('true_negatives')
                ) /
                (
                    pl.col('true_positives') + pl.col('true_negatives') +
                    pl.col('false_positives') + pl.col('false_negatives')
                )
            ).alias('accuracy'),
            (
                pl.col('true_positives') /
                (
                    pl.col('true_positives') + pl.col('false_positives')
                )
            ).alias('precision'),
            (
                pl.col('true_positives') /
                (
                    pl.col('true_positives') + pl.col('false_negatives')
                )
            ).alias('recall')
        ]).with_columns([
            (
                2 * pl.col('precision') * pl.col('recall') /
                (
                    pl.col('precision') + pl.col('recall')
                )
            ).alias('f1_score')
        ])

        self.performance = {
            'over_time': performance_df,
            'by_team': performance_by_team_df
        }


def download_newest_matching_json_file(bucket_name, destination_folder, pattern):
    # Initialize a client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # List all blobs in the bucket
    blobs = bucket.list_blobs()

    # Filter blobs matching the pattern and find the newest one
    newest_blob = None
    newest_time = None

    for blob in blobs:
        if fnmatch.fnmatch(blob.name, pattern):
            blob_time = blob.updated
            if newest_time is None or blob_time > newest_time:
                newest_blob = blob
                newest_time = blob_time

    if newest_blob:
        # Define the local path to save the file
        local_file_path = os.path.join(destination_folder, os.path.basename(newest_blob.name))

        # Download the newest blob to the local destination
        newest_blob.download_to_filename(local_file_path)
        print(f"Downloaded newest matching file: {newest_blob.name} to {local_file_path}")
        return local_file_path
    else:
        print(f"No files matching the pattern '{pattern}' found in the bucket.")