import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Charger le modèle, le scaler et les colonnes sauvegardées
model = joblib.load('best_xgb.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

# Titre du tableau de bord avec une image de logo
st.image("Images/Logo.jpeg", caption="Prêt à dépenser", use_column_width=True)
st.title("Tableau de Bord de Scoring Crédit")

# Entrée des données client
st.header("Entrée des données client")

# Widgets d'entrée
code_gender = st.selectbox("Genre", ["Homme", "Femme"])
cnt_children = st.number_input("Nombre d'enfants", min_value=0, step=1)
amt_income_total = st.number_input("Revenu total", min_value=0.0, step=100.0)
name_education_type = st.selectbox("Niveau d'éducation", ["Secondaire", "Supérieur", "Diplôme universitaire", "Etudes supérieures"])
days_birth = st.slider("Âge", 18, 100, 30)  # Convertir en DAYS_BIRTH (en jours)
days_employed = st.number_input("Jours employés", min_value=-36500, max_value=0, step=1)
amt_credit = st.number_input("Montant du crédit", min_value=0.0, step=100.0)
amt_annuity = st.number_input("Montant de l'annuité", min_value=0.0, step=100.0)
amt_goods_price = st.number_input("Prix des biens", min_value=0.0, step=100.0)
bureau_total_credit_sum = st.number_input("Somme totale du crédit dans le bureau", min_value=0.0, step=100.0)
bureau_total_credit_debt = st.number_input("Dette totale du crédit dans le bureau", min_value=0.0, step=100.0)
ext_source_1 = st.number_input("Score source externe 1", min_value=0.0, max_value=1.0, step=0.01)
ext_source_2 = st.number_input("Score source externe 2", min_value=0.0, max_value=1.0, step=0.01)
ext_source_3 = st.number_input("Score source externe 3", min_value=0.0, max_value=1.0, step=0.01)
inst_pay_total_amount = st.number_input("Montant total des paiements des mensualités", min_value=0.0, step=100.0)
cc_bal_total_amount = st.number_input("Montant total du solde de la carte de crédit", min_value=0.0, step=100.0)
pos_cash_total_instalments_future = st.number_input("Total des mensualités futures", min_value=0.0, step=100.0)

# Conversion de l'âge en DAYS_BIRTH
days_birth = -days_birth * 365
code_gender = 'M' if code_gender == "Homme" else 'F'

# Préparer les données
donnees_clients = pd.DataFrame({
    'CODE_GENDER': [code_gender],
    'CNT_CHILDREN': [cnt_children],
    'AMT_INCOME_TOTAL': [amt_income_total],
    'NAME_EDUCATION_TYPE': [name_education_type],
    'DAYS_BIRTH': [days_birth],
    'DAYS_EMPLOYED': [days_employed],
    'AMT_CREDIT': [amt_credit],
    'AMT_ANNUITY': [amt_annuity],
    'AMT_GOODS_PRICE': [amt_goods_price],
    'BUREAU_TOTAL_CREDIT_SUM': [bureau_total_credit_sum],
    'BUREAU_TOTAL_CREDIT_DEBT': [bureau_total_credit_debt],
    'EXT_SOURCE_1': [ext_source_1],
    'EXT_SOURCE_2': [ext_source_2],
    'EXT_SOURCE_3': [ext_source_3],
    'INST_PAY_TOTAL_AMOUNT': [inst_pay_total_amount],
    'CC_BAL_TOTAL_AMOUNT': [cc_bal_total_amount],
    'POS_CASH_TOTAL_INSTALMENTS_FUTURE': [pos_cash_total_instalments_future]
})

donnees_clients_encoded = pd.get_dummies(donnees_clients)
for col in model_columns:
    if col not in donnees_clients_encoded.columns:
        donnees_clients_encoded[col] = 0

donnees_clients_encoded = donnees_clients_encoded[model_columns]
donnees_clients_scaled = scaler.transform(donnees_clients_encoded)

# Bouton de prédiction
if st.button("Prédire"):
    prediction = model.predict(donnees_clients_scaled)
    result = "Crédit Refusé" if prediction[0] == 1 else "Crédit Accordé"
    st.subheader(f"Résultat : {result}")
    # Afficher une image selon le résultat
    st.image("Images/Refuse.jpg" if prediction[0] == 1 else "Images/Accorde.jpeg", use_column_width=True)
