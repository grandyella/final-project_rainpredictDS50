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

# template kosong
X = pd.DataFrame([[0]*len(features)], columns=features)

st.title("Prediksi Hujan Besok")

# input lokasi, season, region
locs = sorted({f.split("_",1)[1] for f in features if f.startswith("Location_")})
location = st.selectbox("Lokasi", locs)

season = st.selectbox("Season", [1,2,3,4], format_func=lambda x: ["Summer","Fall","Winter","Spring"][x-1])

regions = sorted({f.split("_",1)[1] for f in features if f.startswith("Region_")})
region = st.selectbox("Region", regions)

# input suhu dan tekanan
temp9am = st.number_input("Temp 9am (°C)", -5.0, 45.0, step=0.1)
temp3pm = st.number_input("Temp 3pm (°C)", -5.0, 45.0, step=0.1)

pressure9am = st.number_input("Pressure 9am (hPa)", 980.0, 1045.0, step=0.1)
pressure3pm = st.number_input("Pressure 3pm (hPa)", 980.0, 1045.0, step=0.1)

# input angin
windDirs = sorted({f.split("_")[1] for f in features if f.startswith("WindDir9am_")})

wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", 0.0, 135.0, step=0.1)
wind_gust_dir = st.selectbox("Wind Gust Dir", windDirs)

wind_dir9am = st.selectbox("Wind Dir 9am", windDirs)
wind_dir3pm = st.selectbox("Wind Dir 3pm", windDirs)

# isi data ke dataframe
X.loc[0,"Season"] = season
X.loc[0,"Temp9am"] = temp9am
X.loc[0,"Temp3pm"] = temp3pm
X.loc[0,"Pressure9am"] = pressure9am
X.loc[0,"Pressure3pm"] = pressure3pm
X.loc[0,"WindGustSpeed"] = wind_gust_speed

# one hot encode kategori
for pref,val in [("Location",location),("Region",region),
                 ("WindGustDir",wind_gust_dir),
                 ("WindDir9am",wind_dir9am),
                 ("WindDir3pm",wind_dir3pm)]:
    col = f"{pref}_{val}"
    if col in X.columns:
        X.loc[0,col] = 1.0

# prediksi
if st.button("Prediksi"):
    try:
        Xs = scaler.transform(X)
        pred = model.predict(Xs)[0]
        prob = model.predict_proba(Xs)[0][1]

        if pred == 1:
            st.write("Besok kemungkinan Hujan ☔")
        else:
            st.write("Besok kemungkinan Tidak Hujan ☀️")

        st.write("Probabilitas hujan:", round(prob*100,2), "%")

    except Exception as e:
        st.error(e)
