from flask import Flask, request, render_template_string, url_for
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from binance.client import Client
import joblib
import os

app = Flask(__name__)
client = Client()

# HTML con formulario y visor
HTML_FORM = """
<!doctype html>
<title>Gráfico Cipher</title>
<h1>Gráfico Cipher</h1>
<form method="post">
  Moneda: <input type="text" name="symbol" value="BTCUSDT">
  Intervalo: <input type="text" name="timeframe" value="1h">
  <input type="submit" value="Graficar">
</form>
{% if chart %}
<iframe src="{{ url_for('static', filename='grafico.html') }}" width="100%" height="800"></iframe>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    chart = False
    if request.method == "POST":
        symbol = request.form["symbol"]
        timeframe = request.form["timeframe"]
        limit = 1000

        # Cargar datos
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        df = df[["open", "high", "low", "close", "volume"]]

        # WaveTrend + MFI
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        esa = ta.ema(hlc3, length=9)
        de = ta.ema(abs(hlc3 - esa), length=9)
        ci = (hlc3 - esa) / (0.015 * de)
        wt1 = ta.ema(ci, length=12)
        wt2 = ta.sma(wt1, length=3)
        df['wt1'], df['wt2'] = wt1, wt2

        mfi = ((df['close'] - df['open']) / (df['high'] - df['low'])) * 150
        norm_mfi = ((mfi - mfi.min()) / (mfi.max() - mfi.min()) * 200 - 100) * 10
        df['rsi_mfi'] = ta.sma(norm_mfi, length=60)

        ob_level = 53
        os_level = -53

        df['wt_cross_up'] = (df['wt1'] > df['wt2']) & (df['wt1'].shift(1) <= df['wt2'].shift(1))
        df['wt_cross_down'] = (df['wt1'] < df['wt2']) & (df['wt1'].shift(1) >= df['wt2'].shift(1))
        df['wt_oversold'] = df['wt2'] <= os_level
        df['wt_overbought'] = df['wt2'] >= ob_level
        df['buy_signal'] = df['wt_cross_up'] & df['wt_oversold']
        df['sell_signal'] = df['wt_cross_down'] & df['wt_overbought']
        df['wt1-wt2'] = abs(df['wt1'] - df['wt2'])

        # Features para modelos
        for col in ['wt_cross_up', 'wt_cross_down']:
            df[col + '_only'] = df[col] & ~df['buy_signal'] & ~df['sell_signal']
        df['wt_up_strong'] = df['wt_cross_up_only'] & (df['wt1-wt2'] >= 1)
        df['wt_up_weak'] = df['wt_cross_up_only'] & (df['wt1-wt2'] < 1)
        df['wt_down_strong'] = df['wt_cross_down_only'] & (df['wt1-wt2'] >= 0.5)
        df['wt_down_weak'] = df['wt_cross_down_only'] & (df['wt1-wt2'] < 0.5)

        for col in ['wt_cross_up_only', 'wt_cross_down_only', 'wt_up_strong', 'wt_up_weak', 'wt_down_strong', 'wt_down_weak']:
            df[col] = df[col].astype(int)

        X = df[[ 'wt1', 'wt2', 'rsi_mfi', 'wt1-wt2',
                 'wt_cross_up_only', 'wt_cross_down_only',
                 'wt_up_strong', 'wt_up_weak',
                 'wt_down_strong', 'wt_down_weak' ]].dropna()
        df = df.iloc[-len(X):]

        # Cargar modelos
        model_buy = joblib.load("models/4h-buy.pkl")
        model_sell = joblib.load("models/4h-sell.pkl")
        df["prediction_buy"] = model_buy.predict(X)
        df["prediction_sell"] = model_sell.predict(X)

        # Crear gráfico
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.3, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='Precio'), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['rsi_mfi'], name='RSI+MFI', line=dict(color='white')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['wt1'], name='WT1', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['wt2'], name='WT2', line=dict(color='purple')), row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index[df['buy_signal']], y=df['low'][df['buy_signal']] * 0.98,
                                 mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Compra'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[df['sell_signal']], y=df['high'][df['sell_signal']] * 1.02,
                                 mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Venta'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[df["prediction_sell"] == 1],
                                 y=df['high'][df["prediction_sell"] == 1] * 1.01,
                                 mode='markers', marker=dict(symbol='star', size=12, color='orange'),
                                 name='Predicción Venta'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[df["prediction_buy"] == 1],
                                 y=df['low'][df["prediction_buy"] == 1] * 0.99,
                                 mode='markers', marker=dict(symbol='star', size=12, color='lime'),
                                 name='Predicción Compra'), row=1, col=1)

        fig.update_layout(title=f"Cipher - {symbol} ({timeframe})", height=900, template="plotly_dark", hovermode="x unified")
        fig.update_xaxes(rangeslider_visible=False)

        path = os.path.join("static", "grafico.html")
        fig.write_html(path)
        chart = True

    return render_template_string(HTML_FORM, chart=chart)

if __name__ == "__main__":
    app.run(debug=True)
