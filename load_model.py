import xgboost as xgb

import MetaTrader5 as mt

import datetime as dt

import pandas as pd

import numpy as np

from config import build_file_path

class loading_model:

    def __init__(self, start, end, time_frame):
        if not mt.initialize():
            print("MT5の初期化に失敗", mt.last_error())
            quit()

        self.path = build_file_path(start, end)
        dt_now = dt.datetime.now()
        dt_now_hour = dt_now.hour
        dt_now_minutes = dt_now.minute
        total_minutes = (dt_now_hour * 12) + (dt_now_minutes / 5) #今日の日付から今現在の経過時間までの5分足の本数
        int_total_minutes = int(total_minutes)
        rates = mt.copy_rates_from_pos("USDJPY.cl", time_frame, 1, int_total_minutes)
        self.pd_rates = pd.DataFrame(rates)

    def model_high(self, window, tree_batch_size):
        params = {"objective": "reg:squarederror", "learning_rate": 0.1,}
        feature_columns = []
        x_buffer = []
        y_buffer_model_high = []

        ohlcts_cols = [ "open", "high", "low", "close", "tick_volume", "spread"]
        for j in range(window):

            for col in ohlcts_cols:
                feature_columns.append(f"{col}_{j}")

        booster = xgb.Booster()
        booster.load_model(rf"{self.path.save_trained_model_high()}")

        for i in range(window, len(self.pd_rates)):
            window_data = self.pd_rates.iloc[i - window: i][ohlcts_cols]
            feature_row = window_data.values.flatten()#一次元配列に変換
            x_buffer.append(feature_row)
            y_buffer_model_high.append(self.pd_rates.iloc[i]["high"])

            if len(x_buffer) >= tree_batch_size:
                train_x = pd.DataFrame(x_buffer, columns = feature_columns)
                train_y = pd.Series(y_buffer_model_high)
                dtrain = xgb.DMatrix(train_x, label = train_y)#xgb.DMatrix(train_x, label=train_y)は特徴量とlabelの行数を一致させる必要がある
                booster = xgb.train(params, dtrain, num_boost_round = 1, xgb_model = booster)
                x_buffer = []
                y_buffer_model_high = []     

            if i + 1 == len(self.pd_rates):
                print(f"今日のhighの{i}行目まで学習完了。{booster.num_boosted_rounds()}")
                booster.save_model(rf"{self.path.save_trained_model_high()}")
                break

        return y_buffer_model_high

    def model_low(self, window, tree_batch_size):
        ohlcts_cols = [ "open", "high", "low", "close", "tick_volume", "spread"]
        params = {"objective": "reg:squarederror", "learning_rate": 0.01,}
        feature_columns = []
        x_buffer = []
        y_buffer_model_low = []

        for j in range(window):

            for col in ohlcts_cols:
                feature_columns.append(f"{col}_{j}")

        booster = xgb.Booster()
        booster.load_model(rf"{self.path.save_trained_model_low()}")

        for i in range(window, len(self.pd_rates)):
            window_data = self.pd_rates.iloc[i - window: i][ohlcts_cols]
            feature_row = window_data.values.flatten()#一次元配列に変換
            x_buffer.append(feature_row)
            y_buffer_model_low.append(self.pd_rates.iloc[i]["low"])

            if len(x_buffer) >= tree_batch_size:
                train_x = pd.DataFrame(x_buffer, columns = feature_columns)
                train_y = pd.Series(y_buffer_model_low)
                dtrain = xgb.DMatrix(train_x, label = train_y)#xgb.DMatrix(train_x, label=train_y)は特徴量とlabelの行数を一致させる必要がある
                booster = xgb.train(params, dtrain, num_boost_round = 1, xgb_model = booster)
                x_buffer = []
                y_buffer_model_low = []     

            if i + 1 == len(self.pd_rates):
                print(f"今日のlowの{i}行目まで学習完了。{booster.num_boosted_rounds()}")
                booster.save_model(rf"{self.path.save_trained_model_low()}")
                break

        return x_buffer, y_buffer_model_low