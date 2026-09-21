import MetaTrader5 as mt
import xgboost as xgb
import pandas as pd
import threading
import time
from config import build_file_path
from monitoring_cmd import cmd_monitor

def predict_real_rates(start, end, time_frame, window, tree_batch_size, count, leftover_y_buffer_model_high, leftover_x_buffer, leftover_y_buffer_model_low):
    if not mt.initialize():
        print("MT5の初期化に失敗", mt.last_error())
        quit()

    path = build_file_path(start, end)
    monitor_exit = cmd_monitor().exit_run
    event = threading.Event()
    monitor = threading.Thread(target = monitor_exit, args = (event,), daemon = True)
    monitor.start()
    collation_rates = None
    running_count = 0
    feature_columns = []

    ohlcts_cols = [ "open", "high", "low", "close", "tick_volume", "spread"]
    for j in range(window):

        for col in ohlcts_cols:
            feature_columns.append(f"{col}_{j}")

    x_buffer = pd.DataFrame( columns = feature_columns, dtype = "float64")
    high_buffer = []
    low_buffer = []
    len_leftover_data = len(leftover_x_buffer)
    params = {"objective": "reg:squarederror", "learning_rate": 0.1,}
    model_high = xgb.Booster()
    model_high.load_model(rf"{path.save_trained_model_high()}")
    model_low = xgb.Booster()
    model_low.load_model(rf"{path.save_trained_model_low()}")

    if not  len_leftover_data == 0:
        pd_leftover_x = pd.DataFrame(leftover_x_buffer, columns = feature_columns)
        x_buffer = pd.concat([x_buffer, pd_leftover_x], ignore_index=True)
        high_buffer.extend(leftover_y_buffer_model_high)
        low_buffer.extend(leftover_y_buffer_model_low)

    try:
        while not event.is_set():
            now_rates = mt.copy_rates_from_pos("USDJPY.cl", time_frame, 1, window)

            if now_rates is None or len(now_rates) == 0:
                time.sleep(10)
                continue

            pd_now_rates = pd.DataFrame(now_rates)
            monitoring_rate = pd_now_rates.loc[0,"time"]

            if collation_rates is None:
                collation_rates = monitoring_rate

            elif not collation_rates == monitoring_rate:
                collation_rates = None
                pd_now_rates.drop(columns = ["time", "real_volume"], inplace = True)
                print("pd_now_rate")
                print(pd_now_rates)

                feature_row = pd_now_rates.values.flatten()
                pd_feature_row= pd.DataFrame([feature_row], columns = feature_columns)
                x_buffer = pd.concat([x_buffer, pd_feature_row], ignore_index=True)

                if not running_count == 0:
                    high_buffer.append(pd_now_rates.iloc[window - 1 ]["high"])
                    low_buffer.append(pd_now_rates.iloc[window - 1]["low"])

                pred_dmatrix = xgb.DMatrix(pd_feature_row)
                prediction_rate_high = model_high.predict(pred_dmatrix)
                print("prediction high")
                print(prediction_rate_high)      

                prediction_rate_low = model_low.predict(pred_dmatrix)
                print("prediction low")
                print(prediction_rate_low)

                running_count = running_count + 1

                if running_count == count:
                    print(f"{count}本の予測が完了\n終了します")
                    break

                if len(high_buffer) >= tree_batch_size and len(low_buffer) >= tree_batch_size:
                    train_x = x_buffer.iloc[: -1]
                    train_high = pd.Series(high_buffer)
                    dtrain = xgb.DMatrix(train_x, label = train_high)
                    model_high = xgb.train(params, dtrain, num_boost_round = 1, xgb_model = model_high)
                    model_high.save_model(path.save_trained_model_high())
                    train_low = pd.Series(low_buffer)
                    dtrain = xgb.DMatrix(train_x, label = train_low)
                    model_low = xgb.train(params, dtrain, num_boost_round = 1, xgb_model = model_low)
                    model_low.save_model(path.save_trained_model_low())

                if len(high_buffer) >= tree_batch_size:
                    x_buffer = x_buffer.tail(1).reset_index(drop = True)
                    high_buffer = []
                    low_buffer = []
                continue

            time.sleep(10)


    except KeyboardInterrupt:
        print("Ctrl + c で中断されました")

    finally:
        mt.shutdown()