import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# load model
id_model = "1vw9qq0NPiVOdZpRfq26nTvggEv9mQWBs"
model_path = "random_forest_model.joblib"
scaler_path = "scaler.joblib"

if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={id_model}", model_path, quiet=False)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
features = list(scaler.feature_names_in_)

# buat template
X = pd.DataFrame([[0]*len(features)], columns=features)

st.title("Prediksi Hujan Besok")

# input sederhana
year = st.selectbox("Tahun", list(range(2000, 2025)))
month = st.selectbox("Bulan", list(range(1, 13)))
day = st.selectbox("Hari", list(range(1, 32)))

# lokasi dropdown
locs = sorted({f.split("_",1)[1] for f in features if f.startswith("Location_")})
location = st.selectbox("Lokasi", locs)

minT = st.number_input("Min Temp", -5.0, 45.0, step=0.1)
maxT = st.number_input("Max Temp", -5.0, 50.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", 0.0, 370.0, step=0.1)

humidity9 = st.number_input("Humidity 9am (%)", 0.0, 100.0, step=0.1)
humidity3 = st.number_input("Humidity 3pm (%)", 0.0, 100.0, step=0.1)

rainToday = st.selectbox("Rain Today", ["No", "Yes"])

# isi dataframe
X.loc[0,"Year"] = year
X.loc[0,"Month"] = month
X.loc[0,"Day"] = day
X.loc[0,"MinTemp"] = minT
X.loc[0,"MaxTemp"] = maxT
X.loc[0,"Rainfall"] = rainfall
X.loc[0,"Humidity9am"] = humidity9
X.loc[0,"Humidity3pm"] = humidity3
X.loc[0,"RainToday"] = 1 if rainToday=="Yes" else 0

col_loc = f"Location_{location}"
if col_loc in X.columns:
    X.loc[0,col_loc] = 1.0

# prediksi
if st.button("Prediksi"):
    try:
        Xs = scaler.transform(X)
        pred = model.predict(Xs)[0]
        prob = model.predict_proba(Xs)[0][1]

        if pred == 1:
            st.write("Besok kemungkinan Hujan")
        else:
            st.write("Besok kemungkinan Tidak Hujan")

        st.write("Probabilitas hujan:", round(prob*100,2), "%")

    except Exception as e:
        st.error(e)
