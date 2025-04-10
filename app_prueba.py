from flask import Flask, render_template_string, request, redirect, url_for
import plotly.graph_objects as go

app = Flask(__name__)

HTML_FORM = """
<!doctype html>
<title>Gráfico de prueba</title>
<h1>Selecciona y genera gráfico</h1>
<form method="post">
  Moneda: <input type="text" name="symbol" value="BTCUSDT"><br>
  Temporalidad:
  <select name="timeframe">
    <option value="1m">1m</option>
    <option value="15m">15m</option>
    <option value="1h">1h</option>
    <option value="1d">1d</option>
  </select><br>
  <input type="submit" value="Graficar">
</form>
{% if chart %}
  <iframe src="{{ url_for('static', filename='grafico.html') }}" width="100%" height="600"></iframe>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    chart = False
    if request.method == "POST":
        symbol = request.form["symbol"]
        timeframe = request.form["timeframe"]

        # Crear gráfico de prueba
        fig = go.Figure()
        fig.add_scatter(y=[1, 3, 2, 4], name=f"{symbol} {timeframe}")
        try:
            fig.write_html("static/grafico.html")
            chart = True
            print("Gráfico guardado correctamente.")
        except Exception as e:
            print(f"Error al guardar el gráfico: {e}")

    return render_template_string(HTML_FORM, chart=chart)

if __name__ == "__main__":
    app.run(debug=True)
