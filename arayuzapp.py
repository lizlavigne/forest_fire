import streamlit as st
import pandas as pd
import pickle

with open("orman_yangini_model_pickle.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🌲 Orman Yangını Risk Tahmini")

st.write("""
Meteorolojik veriler (sıcaklık, bağıl nem, rüzgar, yağmur) girerek  
orman yangını riskini tahmin eden uygulama.
""")

# daata giris
temp = st.number_input("🌡️ Sıcaklık (°C)", min_value=-30.0, max_value=60.0, value=20.0)
RH = st.number_input("💧 Bağıl Nem (%)", min_value=0, max_value=100, value=50)
wind = st.number_input("🌬️ Rüzgar Hızı (km/saat)", min_value=0.0, max_value=100.0, value=10.0)
rain = st.number_input("🌧️ Yağmur (mm)", min_value=0.0, max_value=50.0, value=0.0)

if st.button("🔥 Tahmin Et"):
    veri_girisi = pd.DataFrame([[temp, RH, wind, rain]], columns=['temp', 'RH', 'wind', 'rain'])

    try:
        olasilik = model.predict_proba(veri_girisi)[0][1]

        st.write(f"🔎 Yangın çıkma olasılığı: **%{olasilik * 100:.2f}**")

        if olasilik > 0.6:
            st.error("🚨 Yangın riski YÜKSEK! Lütfen dikkatli olun.")
        else:
            st.success("✅ Yangın riski DÜŞÜK. Kontrollü olun.")

    except AttributeError:
        sonuc = model.predict(veri_girisi)[0]
        if sonuc == 1:
            st.error("🚨 Yangın riski YÜKSEK! Lütfen dikkatli olun.")
        else:
            st.success("✅ Yangın riski DÜŞÜK. Kontrollü olun.")