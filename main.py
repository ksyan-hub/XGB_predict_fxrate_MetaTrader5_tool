import MetaTrader5 as mt
from pathlib import Path as plp
import datetime as dt
from send2trash import send2trash
from config import build_file_path
from copy_rates_to_csv import copy_train_data
from xgb_train_model.xgb_train_model_high import train_model_high
from xgb_train_model.xgb_train_model_low import train_model_low
from load_model import loading_model
from run_model_train import predict_real_rates

select_function = input("既存の学習モデルを稼働させますか？y/n:")

if select_function == "n":

    print("新しいモデルを学習させます")
    traindata_start_year = int(input("学習させたい過去から今日までの期間を設定します\n始点\n年:"))
    traindata_start_month = int(input("月:"))
    traindata_start_day = int(input("日:"))
    dt_now = dt.datetime.now()
    traindata_end_year = dt_now.year
    traindata_end_month = dt_now.month
    traindata_end_day = dt_now.day
    time_frame =  int(mt.TIMEFRAME_M5)
    window = 12 # 60分 / 5分
    tree_batch_size = 12
    prediction_range = int(input("予測する本数:"))

    class input_setting:
        def __init__(self):
            self.start = (traindata_start_year, traindata_start_month, traindata_start_day)
            self.end = (traindata_end_year, traindata_end_month, traindata_end_day)
            self.time_frame = (time_frame)

    setting = input_setting()

    path = build_file_path(setting.start, setting.end)
    check_file = plp(path.train_data())
    check_pre_folder = plp(path.predictions_folder())
    check_save_folder = plp(path.save_folder())

    for file in plp(rf".\csv_rate_data\train_data_rates").iterdir():
        if file.is_file() and file.name == check_file.name:
            send2trash(str(file))
            print(f"ゴミ箱に移動しました: {file}")
        else:
            print(f"ファイルが存在しません: {check_pre_folder}")

    from send2trash import send2trash

    if check_pre_folder.exists():
        send2trash(str(check_pre_folder))
        print(f"ゴミ箱に移動しました: {check_pre_folder}")
    else:
        print(f"フォルダが存在しません: {check_pre_folder}")


    if check_save_folder.exists():
        send2trash(str(check_save_folder))
        print(f"ゴミ箱に移動しました: {check_save_folder}")
    else:
        print(f"フォルダが存在しません: {check_save_folder}")

    copy_train_data(setting.start, setting.end, setting.time_frame)
    train_model_high(setting.start, setting.end, window, tree_batch_size)
    train_model_low(setting.start, setting.end, window, tree_batch_size)
    predict_real_rates(setting.start, setting.end, setting.time_frame, window, tree_batch_size, prediction_range, 
                       loading_model(setting.start, setting.end, setting.time_frame).model_high(window, tree_batch_size),
                       *loading_model(setting.start, setting.end, setting.time_frame).model_low(window, tree_batch_size))
