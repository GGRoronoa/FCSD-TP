import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMAResults
from datetime import timedelta

st.set_page_config(page_title="AgriPredict | Kenya", page_icon="", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #black; border-radius: 10px; padding: 15px; border: 1px solid #eee; }
    .status-box { padding: 20px; border-radius: 10px; background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
    </style>
    """, unsafe_allow_html=True)

#data
@st.cache_resource
def load_assets():
    df = pd.read_csv('data_filtr.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    model_xgb = joblib.load('modell.pkl')
    model_arima = ARIMAResults.load('arima.pkl')
    return df, model_xgb, model_arima

try:
    df, model_xgb, model_arima = load_assets()
except Exception as e:
    st.error(f"Erreur de chargement : Vérifiez que les fichiers .csv et .pkl sont présents. ({e})")
    st.stop()

#sidebar
with st.sidebar:
    st.title("Maizepricepredict")
    st.subheader("Configuration de l'analyse")
    
    county_list = sorted(df['County'].unique())
    selected_county = st.selectbox("Comté cible", county_list)
    
    model_choice = st.radio("Algorithme", ["XGBoost (Recommandé)", "ARIMA"])
    
    st.markdown("---")
    st.warning("**Horizon : Semaine +1**\n\nLe modèle est configuré pour une prédiction à court terme afin de garantir une fiabilité maximale.")


df_county = df[df['County'] == selected_county].sort_values('Date')
last_date = df_county['Date'].max()
next_week = last_date + timedelta(weeks=1)
current_price = df_county['Price'].iloc[-1]

#XGBoost
df_dummy = pd.get_dummies(df, columns=['County'])
feature_names = model_xgb.get_booster().feature_names
X_input = df_dummy[df_dummy[f'County_{selected_county}'] == 1].drop(columns=['Date', 'Price']).iloc[-1:]
X_input = X_input[feature_names]


if "XGBoost" in model_choice:
    prediction = model_xgb.predict(X_input)[0]
    score = "98.2%"
    rmse = "1.68"
else:
    prediction = model_arima.forecast(steps=1)[0]
    score = "89.5%"
    rmse = "2.74"

st.title(f"Marché du Maïs : {selected_county}")
st.write(f"Analyse basée sur les données réelles jusqu'au **{last_date.strftime('%d %B %Y')}**")


c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Dernier Prix Connu", f"{current_price:.2f} KES")
with c2:
    delta = prediction - current_price
    st.metric(f"Prévision pour le {next_week.strftime('%d/%m')}", f"{prediction:.2f} KES", f"{delta:+.2f} KES", delta_color="inverse")
with c3:
    st.metric("Score de Confiance", score, f"RMSE: {rmse}")

st.divider()

#graph
col_graph, col_info = st.columns([2, 1])

with col_graph:
    fig = go.Figure()
    #historique
    df_hist = df_county.tail(12)
    fig.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['Price'], name="Historique", line=dict(color='#2E7D32', width=3)))
    #prediction
    fig.add_trace(go.Scatter(x=[last_date, next_week], y=[current_price, prediction], 
                             name="Prévision", line=dict(color='#FFA000', width=3, dash='dash'),
                             mode='lines+markers', marker=dict(size=10)))
    
    fig.update_layout(height=400, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.markdown(f"""
    <div class="status-box">
        <h4>💡 Note d'analyse</h4>
        Le modèle <b>{model_choice}</b> analyse les cycles de prix saisonniers au Kenya. 
        Pour <b>{selected_county}</b>, la tendance actuelle indique une {'hausse' if delta > 0 else 'baisse'} 
        de <b>{abs(delta):.2f} KES</b>.
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    st.write("**Données utilisées :**")
    st.write("- 4 dernières semaines (Lags)")
    st.write("- Moyenne mobile (Rolling Mean)")
    st.write("- Effet saisonnier du mois")