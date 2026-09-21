from datetime import datetime
import pandas as pd
import pytz
import MetaTrader5 as mt5
from config import build_file_path

def copy_train_data(start, end, time_frame):

    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    path = build_file_path(start, end)
    print(f"{start}から{end}のデータ取得を実行")
    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime(*start, tzinfo = timezone)
    utc_to = datetime(*end, tzinfo = timezone)
    rates = mt5.copy_rates_range("USDJPY.cl", time_frame, utc_from, utc_to) 

    if rates is not None:
        print("データ取得に成功")

    mt5.shutdown()
    rates_frame = pd.DataFrame(rates)
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    print(f"{time_frame}\n{rates_frame}")
    rates_frame.to_csv(path.train_data(),index=False)
