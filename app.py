from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import time
import ccxt
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# 初始化 exchange 对象
exchange = ccxt.binance()


# 定义根路径
@app.route('/')
def home():
    return "Welcome to the AI Trading Bot API 2024"


# 获取历史数据
def fetch_ohlcv(symbol, timeframe='5m', limit=1000, retries=5, delay=5):
    for attempt in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except ccxt.NetworkError as e:
            print(f'Network error: {e}, retrying in {delay} seconds...')
            time.sleep(delay)
        except Exception as e:
            print(f'Error in fetch_ohlcv: {e}')
            break
    return None


# 特征工程：计算价格变化和技术指标
def feature_engineering(df):
    df['return'] = df['close'].pct_change()
    df['sma'] = df['close'].rolling(window=5).mean()
    df['ema'] = df['close'].ewm(span=5, adjust=False).mean()
    df['volatility'] = df['close'].rolling(window=5).std()
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['direction'] = np.where(df['return'] > 0, 1, 0)
    df.dropna(inplace=True)
    return df


# 训练模型
def train_model(df):
    X = df[['return', 'sma', 'ema', 'volatility', 'momentum']]
    y = df['direction']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model


# 使用模型预测
def predict(model, df):
    X = df[['return', 'sma', 'ema', 'volatility', 'momentum']]
    df = df.copy()
    df['prediction'] = model.predict(X)
    return df


# API 路由
@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    symbol = data['symbol']
    df = fetch_ohlcv(symbol)

    # 检查 df 是否为 None
    if df is None:
        return jsonify({'error': 'Failed to fetch OHLCV data'}), 500

    df = feature_engineering(df)
    model = train_model(df)
    prediction = predict(model, df.tail(1))

    # 转换 prediction 类型
    prediction_value = int(prediction['prediction'].values[0])

    return jsonify({'prediction': prediction_value})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
