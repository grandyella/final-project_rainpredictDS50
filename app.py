import streamlit as st
import pandas as pd
import joblib

# Judul
st.title("Prediksi Hujan Besok 🌧️☀️")

# Load model (pastikan sudah kamu simpan di Colab atau repo)
# contoh: random_forest_model.joblib dan scaler.joblib
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

# Input dari user
st.header("Masukkan Data Cuaca Hari Ini")
min_temp = st.number_input("MinTemp")
max_temp = st.number_input("MaxTemp")
rainfall = st.number_input("Rainfall")
humidity9am = st.number_input("Humidity 9am")
humidity3pm = st.number_input("Humidity 3pm")
wind_speed9am = st.number_input("Wind Speed 9am")
wind_speed3pm = st.number_input("Wind Speed 3pm")

# Prediksi
if st.button("Prediksi"):
    # Buat dataframe dari input
    X_new = pd.DataFrame([[min_temp, max_temp, rainfall,
                           humidity9am, humidity3pm,
                           wind_speed9am, wind_speed3pm]],
                         columns=["MinTemp","MaxTemp","Rainfall",
                                  "Humidity9am","Humidity3pm",
                                  "WindSpeed9am","WindSpeed3pm"])
    # Scaling
    X_new_scaled = scaler.transform(X_new)

    # Prediksi
    pred = model.predict(X_new_scaled)

    hasil = "Hujan" if pred[0]==1 else "Tidak Hujan"
    st.subheader(f"Hasil Prediksi: {hasil}")