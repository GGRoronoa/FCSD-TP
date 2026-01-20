Notre solution d'intelligence artificielle est dédiée à la prévision des prix du maïs sur les marchés kenyans. Ce projet utilise une approche hybride combinant l'analyse de séries temporelles et l'apprentissage automatique (Machine Learning) pour fournir des outils d'aide à la décision robustes aux acteurs agricoles.



## Points Forts du Projet
- **Modélisation avancée** : Utilisation de **XGBoost** pour capturer les tendances complexes et **ARIMA** pour l'analyse structurelle des séries temporelles.
- **Pipeline MLOps** : Système de ré-entraînement automatisé garantissant que les prédictions restent précises face à l'évolution du marché.
- **Filtrage dynamique** : Algorithme de nettoyage intelligent.
- **Dashboard interactif** : Interface utilisateur fluide permettant de visualiser l'historique et les prévisions futures en temps réel.

## Installation et Configuration

### 1. Prérequis
- Python 3.9 ou version supérieure.
- Gestionnaire de paquets `pip`.

### 2. Installation
```bash
# Cloner le dépôt
git clone [https://github.com/GGRoronoa/FCSD-TP.git]
cd FCSD-TP

# Installer les dépendances nécessaires
pip install -r requirements.txt

#Utilisation
streamlit run app.py
