import pandas as pd
import xgboost as xgb
import joblib
from statsmodels.tsa.arima.model import ARIMA
import schedule
import time
import os

#config
SOURCE_URL = "https://raw.githubusercontent.com/AgriDataKenya/datasets/main/maize_prices.csv"
BACKUP_FILE = "maize_prices_backup.csv"

def run_pipeline():
    print(f"\nCycle de mise à jour lancé : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    #recueillir données
    try:
        print("Téléchargement des données...")
        df = pd.read_csv(SOURCE_URL)
        df.to_csv(BACKUP_FILE, index=False) # On met à jour notre sauvegarde
        print("Nouvelles données récupérées.")
    except Exception as e:
        print(f"Serveur injoignable. Utilisation du backup local. ({e})")
        df = pd.read_csv(BACKUP_FILE)

    df['Date'] = pd.to_datetime(df['Date'])

    #filtrage avec date actuelle - 6 mois
    today = pd.Timestamp.now()
    seuil_fraicheur = today - pd.DateOffset(months=6)
    
    #filtrage des comtés actifs
    df_filtered = df[df['Date'] >= seuil_fraicheur].copy()
    
    if df_filtered.empty:
        print("Alerte:Aucune donnée récente trouvée. Entraînement annulé.")
        return

    #prepa donnees
    df_filtered = df_filtered.sort_values(['County', 'Date'])
    
    #lags
    for i in range(1, 5):
        df_filtered[f'lag_price_{i}'] = df_filtered.groupby('County')['Price'].shift(i)
    
    #rolling mean
    df_filtered['rolling_mean_4'] = df_filtered.groupby('County')['Price'].transform(lambda x: x.rolling(window=4).mean())
    df_filtered['month'] = df_filtered['Date'].dt.month
    
    data_filtr = df_filtered.dropna()

    #retrain xgboost
    print("Apprentissage de l'IA...")
    data_ml = pd.get_dummies(data_filtr, columns=['County'])
    
    X = data_ml.drop(columns=['Date', 'Price'])
    y = data_ml['Price']
    
    model_xgb = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
    model_xgb.fit(X, y)

    #sauvegarde modeles et data
    joblib.dump(model_xgb, 'model.pkl')
    data_filtr.to_csv('data_filtr.csv', index=False)
    
    print(f"Mise à jour terminée")
    print("Mise en veille jusqu'au prochain lundi à 08:00...")


#tâche chaque lundi à 08h00
schedule.every().monday.at("08:00").do(run_pipeline)

run_pipeline()
while True:
    schedule.run_pending()
    time.sleep(60)
