# XGB_predict_fxrate_MetaTrader5_tool

MetaTrader5(MT5)と XGBoost を組み合わせた、USDJPY 5分足の高値・安値予測ツールです。
試作1〜4で検証した「過去window本の値動きを特徴量にした学習」を、
実際に MT5 とつないでリアルタイムに動かすところまで実装しています。

## できること

現段階ではアプリ全体の流れの中の初回学習の分岐を作りました。

1. `copy_rates_to_csv.py` で取得した過去データを読み込む
2. `xgb_train_model`がwindowごとの特徴量で、1行ずつ「予測→学習」を繰り返す
3. `load_model.py`が学習済みモデルを、今日のここまでの値動きに追いつかせる
4. `run_model_train.py` がリアルタイム予測を行い、リアルタイムで5分足を監視し、予測 → 実際の値が確定したら追加学習、を繰り返す`exit_run` と入力すればいつでも安全に停止できる

## ファイル構成

| ファイル | 役割 |
|---|---|
| `main.py` | エントリーポイント |
| `config.py` | ファイルパス生成 |
| `copy_rates_to_csv.py` | 過去データ取得 |
| `xgb_train_model/xgb_train_model_high.py`, `_low.py` | オフライン学習 |
| `load_model.py` | 現在時刻までの追いつき学習 |
| `run_model_train.py` | リアルタイム予測ループ |
| `monitoring_cmd.py` | 終了コマンドの監視 |

## 学習の考え方

- 過去12本の `open, high, low, close, tick_volume, spread` を1行に平坦化(72特徴量)
- 各行で「その時点のモデルで予測 → 正解を使って学習」の順に処理し、未来のデータを先読みしない
- 12行たまるごとに XGBoost を追加学習(boosting)を行います
- 学習完了後、モデルをJSON形式で保存

## 動作環境

- MetaTrader5はwindows専用ですが、wine10.0を使用してlinux環境でも動作可能 `https://www.mql5.com/ja/articles/625?utm_source=www.metatrader5.com&utm_campaign=download.mt5.linux`
- MetaTrader5のアカウントが必要
- 銘柄のコードにUSDJPY.clと書いてあるが、証券会社ごとに銘柄のコードが違うので注意
- 必要パッケージ: pandas, xgboost, MetaTrader5, pytz, send2trash

## 実行方法

1. `python main.py` を実行
2. 「既存の学習モデルを稼働させますか？」に `n` と回答
3. 学習開始日(年・月・日)と、予測したい本数を入力
4. 過去データの取得 → high/low モデルの学習 → リアルタイム予測が自動で始まる
5. 停止するときはコンソールで `exit_run` と入力

## 今後の課題

- `_high.py` と `_low.py` の重複コードを、対象カラムを引数化した1つの関数にまとめる
- 予測結果をWEBページでリアルタイムのチャートと一緒に見れるようにしたい
- 別端末から遠隔で稼働の停止をできるようにしたい
- MSEの集計タイミング(`i % 12`)を `tree_batch_size` から算出する形にし、値を変えてもずれないようにする
- 学習ログをファイルに残す
