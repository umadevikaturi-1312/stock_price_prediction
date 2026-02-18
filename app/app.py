from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# TensorFlow/Keras import
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv("data/stock_features_ready.csv", parse_dates=['Date'])
df.set_index('Date', inplace=True)

# ---------------------------
# Load Models & Scalers
# ---------------------------
company_models = {}
scalers_X = joblib.load("scalers_X.pkl")
scalers_y = joblib.load("scalers_y.pkl")
company_features_dict = joblib.load("features.pkl")

for file in os.listdir("models"):
    if file.endswith(".keras"):
        company_name = file.replace(".keras", "")
        company_models[company_name] = load_model(os.path.join("models", file))

# Companies list (for sidebar)
companies = [c.replace("_Close_Next", "") for c in company_models.keys()]

timesteps = 60  # same as training

# ---------------------------
# Routes
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html", companies=companies)

@app.route("/plot", methods=["POST"])
def plot_stock():
    company = request.form.get("company")
    target = company + "_Close_Next"

    if target not in company_models:
        return jsonify({"error": f"Model for {company} not found"})

    model = company_models[target]
    scaler_X = scalers_X[target]
    scaler_y = scalers_y[target]

    company_features = company_features_dict[target]
    X = df[company_features].values

    # Last 60 days sequence
    last_seq = X[-timesteps:]
    last_seq_scaled = scaler_X.transform(last_seq).reshape(1, timesteps, len(company_features))

    # Predict next 7 days
    future_preds = []
    for _ in range(7):
        next_pred_scaled = model.predict(last_seq_scaled, verbose=0)
        next_pred = scaler_y.inverse_transform(next_pred_scaled)[0, 0]
        future_preds.append(float(next_pred))

        # Update sequence with new prediction
        new_features = last_seq_scaled[0, -1, :].copy()
        new_features[-1] = next_pred_scaled[0, 0]
        last_seq_scaled = np.roll(last_seq_scaled, -1, axis=1)
        last_seq_scaled[0, -1, :] = new_features

    # Actual last 30 days
    price_column = f"{company}_Close"
    actual_prices = df[price_column].values[-30:]
    actual_prices = [float(p) for p in actual_prices]

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    future_dates = [str(d.date()) for d in future_dates]
    actual_dates = [str(d.date()) for d in df.index[-30:]]

    # Create Plotly figure
    fig = go.Figure()

    # Actual Price (with area fill)
    fig.add_trace(go.Scatter(
        x=actual_dates,
        y=actual_prices,
        mode='lines+markers',
        name='Actual Price',
        fill='tozeroy',                 # 🔹 Fill area under line
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>Price: ₹%{y:,.2f}<extra></extra>'
    ))

    # Predicted Price (dashed)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_preds,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(dash='dash', color='red', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Predicted</b><br>Date: %{x}<br>Price: ₹%{y:,.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"{company} Close Price: Actual vs Predicted (7-day forecast)",
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        width=800,
        height=450,
        hovermode='x unified',  # 🔹 Show hover for both lines together
        legend=dict(
            x=1,
            y=1,
            bgcolor='rgba(255,255,255,0)',
            bordercolor='rgba(0,0,0,0)'
        )
    )

    return jsonify(fig.to_dict())


# ---------------------------

if __name__ == "__main__":

    app.run(debug=True)
