import streamlit as st
import pandas as pd
import joblib
import gdown
import os
from datetime import date

st.set_page_config(page_title="Prediksi Hujan Besok", layout="wide")
st.title("☔ Prediksi Hujan Besok (Auto-encode sesuai scaler)")

# ----------------------------
# CONFIG: ganti ID_MODEL dengan ID Google Drive model kamu
# ----------------------------
ID_MODEL = "ISI_ID_MODEL_DRIVE_DI_SINI"   # <- GANTI INI
MODEL_PATH = "random_forest_model.joblib"
SCALER_PATH = "scaler.joblib"  # harus ada di repo (kecil)

# Download model dari Drive jika belum ada
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={ID_MODEL}", MODEL_PATH, quiet=False)

# Load model & scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Ambil feature names yang dipakai scaler (urutan penting)
feature_names = list(getattr(scaler, "feature_names_in_", []))
if not feature_names:
    st.error("Scaler tidak memiliki attribute feature_names_in_. Pastikan scaler disimpan setelah fit dengan fitur nama.")
    st.stop()

# Pisahkan kolom numeric (tidak ada underscore) dan kategori (yang berbentuk one-hot: prefix_value)
numeric_feats = [f for f in feature_names if "_" not in f]
# cari prefix kategori, misal 'Location', 'WindDir9am', dsb
cat_prefixes = sorted({name.split("_", 1)[0] for name in feature_names if "_" in name})

# Pars kategori choices: buat dict prefix -> list of suffixes (choices)
cat_choices = {}
for p in cat_prefixes:
    cat_choices[p] = [name.split("_", 1)[1] for name in feature_names if name.startswith(p + "_")]

# Sidebar (atau main) input
st.sidebar.header("Input (isi value untuk semua field yang muncul)")
today = date.today()
# numeric inputs (buat tampilan rapi, kita kelompokkan beberapa common first)
with st.sidebar.expander("Date & Basic"):
    year = st.number_input("Year", value=today.year, step=1)
    month = st.number_input("Month", value=today.month, min_value=1, max_value=12, step=1)
    day = st.number_input("Day", value=today.day, min_value=1, max_value=31, step=1)
    # season kalkulasi otomatis (tapi masih bisa override)
    season_default = (month % 12 + 3) // 3
    season = st.number_input("Season (1:Summer,2:Fall,3:Winter,4:Spring)", value=season_default, min_value=1, max_value=4, step=1)

with st.sidebar.expander("Temperatures, Rain, Wind, Humidity, Pressure"):
    # Jika scaler mengandung kolom tertentu, tampilkan inputnya; default angka sederhana
    def mk_num_input(name, default=0.0):
        val = st.number_input(name, value=float(default))
        return val

    # Tampilkan numeric fields yang ada di scaler
    numeric_inputs = {}
    for nf in numeric_feats:
        if nf in ["Year", "Month", "Day", "Season"]:
            # sudah diisi di atas, skip here
            numeric_inputs[nf] = {"value": None}
            continue
        # provide sensible defaults for some known features
        default_map = {
            "MinTemp": 15.0, "MaxTemp": 25.0, "Rainfall": 0.0,
            "WindGustSpeed": 30.0, "WindSpeed9am":10.0, "WindSpeed3pm":12.0,
            "Humidity9am":60.0, "Humidity3pm":45.0, "Pressure9am":1017.0,
            "Pressure3pm":1015.0, "Temp9am":16.0, "Temp3pm":21.0
        }
        default = default_map.get(nf, 0.0)
        numeric_inputs[nf] = {"value": mk_num_input(nf, default)}

with st.sidebar.expander("Categorical (pilih salah satu)"):
    cat_inputs = {}
    for p in cat_prefixes:
        # buat label lebih 'friendly'
        choices = cat_choices[p]
        # default: first choice
        sel = st.selectbox(p, choices, index=0)
        cat_inputs[p] = sel

# RainToday special (0/1)
rain_today = st.sidebar.selectbox("RainToday (apakah hari ini hujan?)", ["No", "Yes"])
rain_today_val = 1 if rain_today == "Yes" else 0

# ----------------------------
# Build DataFrame template sesuai feature_names, isi 0
# ----------------------------
X = pd.DataFrame([[0]*len(feature_names)], columns=feature_names, dtype=float)

# set date numeric fields
X.loc[0, "Year"] = float(year)
X.loc[0, "Month"] = float(month)
X.loc[0, "Day"] = float(day)
X.loc[0, "Season"] = float(season)

# set numeric inputs
for nf, info in numeric_inputs.items():
    if info["value"] is None:
        continue
    X.loc[0, nf] = float(info["value"])

# set RainToday if present
if "RainToday" in X.columns:
    X.loc[0, "RainToday"] = float(rain_today_val)

# set categorical one-hot based on user selection
for p, sel in cat_inputs.items():
    colname = f"{p}_{sel}"
    if colname in X.columns:
        X.loc[0, colname] = 1.0
    else:
        # jika kategori tidak ada (jarang), tetap lanjut tapi beri peringatan
        st.warning(f"Kategori '{sel}' untuk '{p}' tidak ditemukan di scaler (kolom '{colname}').")

# show input preview
st.subheader("Preview data (yang akan dipakai model)")
st.dataframe(X.T.rename(columns={0: "value"}))

# Prediksi
if st.button("Prediksi Sekarang"):
    try:
        # transform: prefer DataFrame dengan nama kolom yang sama
        try:
            X_scaled = scaler.transform(X)  # ini menggunakan feature_names_in_
        except Exception:
            # fallback ke array values
            X_scaled = scaler.transform(X.values)

        # prediksi
        pred = model.predict(X_scaled)
        try:
            proba = model.predict_proba(X_scaled)[0]
            prob_yes = proba[1] if len(proba) > 1 else None
        except Exception:
            prob_yes = None

        result_text = "🌧️ Hujan" if int(pred[0]) == 1 else "☀️ Tidak Hujan"
        st.success(f"Hasil Prediksi: **{result_text}**")
        if prob_yes is not None:
            st.info(f"Probabilitas hujan (model): {prob_yes:.2%}")

        # tampilkan row hasil (dengan kolom hasil)
        out_df = X.copy()
        out_df["Prediksi"] = ["Hujan" if int(pred[0])==1 else "Tidak Hujan"]
        st.subheader("Data + Prediksi")
        st.dataframe(out_df)

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
