import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Lire le fichier CSV
df = pd.read_csv('Financial_inclusion.csv')

X = df[['country', 'year', 'location_type', 'cellphone_access', 'household_size', 'age_of_respondent', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']]
y = df['bank_account']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

st.markdown('''
<center>
<h1>Having a bank account in Africa</h1>
</center>
''', unsafe_allow_html=True)

from PIL import Image
img = Image.open("Customer_in_the_bank.jpg")
st.image(img, width=750)

st.header('''
This Application allows us to predict which individuals are most likely to have or use a bank account.
''')
st.subheader('''Based on the demographic information and what financial services are used :''')

st.sidebar.header("Input parameters")

# Collecte des données dans la barre latérale
with st.sidebar:
    st.write("Country :")
    st.write(" 0 - Rwanda")
    st.write(" 1 - Tanzanie")
    st.write(" 2 - Kenya")
    st.write(" 3 - Ouganda")
    st.write("\n")
    country = st.selectbox("Country", df['country'].unique())
    year = st.number_input("Year", value=2021)
    location_type = st.selectbox("Location Type", ['Rural', 'Urban'])
    cellphone_access = st.selectbox("Cellphone Access", ['Yes', 'No'])
    household_size = st.number_input("Household Size", value=1)
    age_of_respondent = st.number_input("Age of Respondent", value=30)
    gender_of_respondent = st.selectbox("Gender of Respondent", ['Male', 'Female'])
    st.write("Relationship with Head :")
    st.write(" 0 - Head of Household")
    st.write(" 1 - Spouse")
    st.write(" 2 - Child")
    st.write(" 3 - Parent")
    st.write(" 4 - Other relative")
    st.write(" 5 - Other non-relative")
    st.write("\n")
    relationship_with_head = st.selectbox("Relationship with Head", df['relationship_with_head'].unique())
    st.write("Marital Status :")
    st.write(" 0 - Marié(e)/En concubinage")
    st.write(" 1 - Célibataire/Jamais marié(e)")
    st.write(" 2 - Veuf/Veuve")
    st.write(" 3 - Divorcé(e)/Séparé(e)")
    st.write(" 4 - Ne sait pas")
    st.write("\n")
    marital_status = st.selectbox("Marital Status", df['marital_status'].unique())
    st.write("Education Level :")
    st.write(" 0 - Primary education")
    st.write(" 1 - No formal education")
    st.write(" 2 - Secondary education")
    st.write(" 3 - Tertiary education")
    st.write(" 4 - Vocational/Specialized training")
    st.write(" 5 - Other/Don't know/RTA")
    st.write("\n")
    education_level = st.selectbox("Educational Level", df['education_level'].unique())
    st.write("Job Type :")
    st.write(" 0 - Self-employed")
    st.write(" 1 - Informally employed")
    st.write(" 2 - Farming and fishing")
    st.write(" 3 - Remittance dependent")
    st.write(" 4 - Other income")
    st.write(" 5 - Formally employed private sector")
    st.write(" 6 - No income")
    st.write(" 7 - Formally employed public sector")
    st.write(" 8 - Government dependent")
    st.write(" 9 - Don't know/Refuse to answer")
    st.write("\n")
    job_type = st.selectbox("Job Type", df['job_type'].unique())

# Convertir les données en format utilisable pour la prédiction
input_data = {
    'country': country,
    'year': year,
    'location_type': location_type,
    'cellphone_access': cellphone_access,
    'household_size': household_size,
    'age_of_respondent': age_of_respondent,
    'gender_of_respondent': gender_of_respondent,
    'relationship_with_head': relationship_with_head,
    'marital_status': marital_status,
    'education_level': education_level,
    'job_type': job_type
}

# Convertir les valeurs de cellphone_access de 'Yes' et 'No' à 1 et 0 respectivement
input_data['cellphone_access'] = 1 if input_data['cellphone_access'] == 'Yes' else 0

# Convertir les valeurs de gender_of_respondent de 'Male' et 'Female' à 1 et 0 respectivement
input_data['gender_of_respondent'] = 1 if input_data['gender_of_respondent'] == 'Male' else 0

# Convertir les valeurs de location_type de 'Rural' et 'Urban' à 1 et 2 respectivement
input_data['location_type'] = 1 if input_data['location_type'] == 'Rural' else 2



# Convertir input_data en DataFrame
input_df = pd.DataFrame([input_data])

# Effectuer une prédiction
prediction = knn.predict(input_df)

# Ajouter un bouton de validation
if st.sidebar.button("Validate"):

    # Effectuer une prédiction
    prediction = knn.predict(input_df)

    # Afficher la prédiction
    st.subheader(f"{'Yes' if prediction[0] else 'No'} this individual is most likely to have or use a bank account.")

# Ajouter une légende pour expliquer les paramètres
st.caption("Les paramètres affichés en chiffres correspondent à:")
st.caption("1. Country: Rwanda, Tanzania, Kenya, Uganda")
st.caption("2. Location Type: Rural, Urban")
st.caption("3. Cellphone Access: Yes, No")
st.caption("4. Gender of Respondent: Male, Female")
st.caption("5. Les autres paramètres correspondent aux valeurs disponibles dans le dataset.")


