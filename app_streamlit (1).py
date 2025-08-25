import streamlit as st
import pandas as pd
import os
import joblib

# path model relatif ke file ini
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_mlbb.joblib")

# load model
model = joblib.load(MODEL_PATH)

st.title("MLBB Classification")
kill = st.slider("Jumlah Kill",0,20)
assist = st.slider("Jumlah Assist",0,20)
death = st.slider("Jumlah Death",0,20)
turret = st.slider("Jumlah Turret",0,20)


if st.button("predict"):
    data_baru = pd.DataFrame([[kill, assist,death,turret]],columns=["kill","assist","death","turret"])
    st.success(f"Hasil Prediksi : {model.predict(data_baru)[0]}")
    st.balloons()
# simpan di folder classification
# buka CMD baru 
# jalankan virtual environment lagi
# jalankan dengan perintah: streamlit run app_streamlit.py