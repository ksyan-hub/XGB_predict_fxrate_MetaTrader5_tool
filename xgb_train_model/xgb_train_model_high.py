import pandas as pd
import xgboost as xgb
import csv
from config import build_file_path

def train_model_high(start, end, window, tree_batch_size):
    path = build_file_path(start, end)

    with open(path.train_data(), "r"):
        train_df_USDJPY_5m =  pd.read_csv(path.train_data())

    params = {"objective": "reg:squarederror", "learning_rate": 0.1,}
    booster = None
    x_buffer = []
    y_buffer = []
    se_buffer = []
    feature_columns = []
    prcssng_float = 1000 #小数点第4以降を切り捨てるための数

    ohlcts_cols = [ "open", "high", "low", "close", "tick_volume", "spread"]
    for j in range(window):

        for col in ohlcts_cols:
            feature_columns.append(f"{col}_{j}")

    with open(path.trained_model_predictions_high(), "w", newline = "") as predict_file:
        writer = csv.writer(predict_file)
        writer.writerow(["0","predicted","actual","se","mse"])

    with open(path.trained_model_predictions_high(), "a", newline ="") as predict_file:
        writer = csv.writer(predict_file)

        for i in range(window, len(train_df_USDJPY_5m)):
            window_data= train_df_USDJPY_5m.iloc[i - window: i][ohlcts_cols] #6列12行のデータを抽出
            feature_row = window_data.values.flatten()#6列12行を一次元配列に変換 
            x_buffer.append(feature_row)
            y_buffer.append(train_df_USDJPY_5m.iloc[i]["high"])

            if booster is not None:
                pred_x = pd.DataFrame([feature_row], columns = feature_columns)
                pred_dmatrix = xgb.DMatrix(pred_x)
                predictions_data = booster.predict(pred_dmatrix)
                predictions = float(predictions_data[0]) #スカラー値に変換
                se = int(((predictions - train_df_USDJPY_5m.iloc[i]["high"])**2)*prcssng_float)/prcssng_float #小数点を切り捨て、誤差を2乗
                se_buffer.append(se)
                prcssng_predictions = int(float(predictions)*prcssng_float)/prcssng_float

                if i % 12 == 0: #60分/5分＝12
                    sum_se = 0
                    count_se = len(se_buffer)

                    for val in se_buffer:
                        sum_se = sum_se + val
                        mse = int((sum_se /  count_se)*prcssng_float)/prcssng_float
                    print(mse)

                    se_buffer = [] #bufferをリセット

                else: mse = None

            else:
                prcssng_predictions = se = mse = None

            writer.writerow([ i - (window - 1), prcssng_predictions, train_df_USDJPY_5m.iloc[i]["high"],se,mse])

            if len(x_buffer) >= tree_batch_size:
                train_x = pd.DataFrame(x_buffer, columns = feature_columns)
                train_y = pd.Series(y_buffer)
                dtrain = xgb.DMatrix(train_x, label = train_y)#xgb.DMatrix(train_x, label=train_y)は特徴量とlabelの行数を一致させる必要がある
                booster = xgb.train(params, dtrain, num_boost_round = 1, xgb_model = booster)
                x_buffer = []
                y_buffer = []     

            if i + 1 == len(train_df_USDJPY_5m):
                print(f"highの{i}行目まで学習完了。{booster.num_boosted_rounds()}")
                booster.save_model(path.save_trained_model_high())
                break