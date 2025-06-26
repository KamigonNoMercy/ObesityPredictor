import streamlit as st
import requests

# ——— CONFIGURASI ———
API_URL = "http://127.0.0.1:8000/predict"
st.set_page_config(page_title="Obesity Risk Predictor", layout="centered")

# ——— HEADER ———
st.markdown("<h1 style='text-align: center; color: #3366cc;'>Obesity Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan data gaya hidup Anda untuk memprediksi tingkat obesitas.<br><small>Powered by <b>Random Forest</b> & FastAPI 🚀</small></p>", unsafe_allow_html=True)
st.markdown("---")

# ——— FORM INPUT ———
with st.form("input_form"):
    st.subheader("🧍‍♂️🧍‍♀️ Data Diri & Gaya Hidup")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🧠 Usia (tahun)", min_value=10, max_value=100, value=25)
        height = st.number_input("✏️ Tinggi (meter)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("⚖️ Berat (kg)", min_value=30.0, max_value=200.0, value=70.0)
        gender = st.selectbox("👫 Gender", ["Male", "Female"])
        fam_hist = st.selectbox("👪 Riwayat Keluarga Overweight", ["Yes", "No"])
        favc = st.selectbox("🍟 Sering Konsumsi Kalori Tinggi", ["Yes", "No"])
        smoke = st.selectbox("🚬 Merokok", ["Yes", "No"])
        scc = st.selectbox("📊 Monitoring Kalori", ["Yes", "No"])

    with col2:
        fcvc = st.selectbox("🥦 Frekuensi Konsumsi Sayur (1–3)", [1, 2, 3])
        ncp = st.selectbox("🍱 Jumlah Makan Utama (1–4)", [1, 2, 3, 4])
        ch2o = st.selectbox("💧 Asupan Air Harian (1–3)", [1, 2, 3])
        faf = st.selectbox("🏃‍♂️ Frekuensi Aktivitas Fisik (0–3)", [0, 1, 2, 3])
        tue = st.selectbox("💻 Waktu Pakai Teknologi (0–3)", [0, 1, 2, 3])
        caec = st.selectbox("🍪 Makan Di Antara Waktu Makan", ["Never", "Sometimes", "Often"])
        calc = st.selectbox("🍷 Konsumsi Alkohol", ["Never", "Sometimes"])
        mtrans = st.selectbox("🚗 Moda Transportasi Utama", ["Public_Transportation", "Automobile", "Other"])

    st.markdown(" ")
    submitted = st.form_submit_button("📍 Prediksi Tingkat Obesitas")

# ——— PROSES PREDIKSI ———
if submitted:
    payload = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "FCVC": fcvc,
        "NCP": ncp,
        "CH2O": ch2o,
        "FAF": faf,
        "TUE": tue,
        "Gender": gender,
        "family_history_with_overweight": fam_hist,
        "FAVC": favc,
        "SMOKE": smoke,
        "SCC": scc,
        "CAEC": caec,
        "CALC": calc,
        "MTRANS": mtrans
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            res = response.json()

            st.success(f"✅ **Prediksi: {res['predicted_class']}**")
            st.markdown("### 📊 Probabilitas Tiap Kategori:")

            max_prob = max(res["probabilities"].values())

            for cls, p in res["probabilities"].items():
                bar = st.progress(p)
                if p == max_prob:
                    st.markdown(f"**<span style='color:lightgreen;'>{cls}</span>**: {p:.2%} 🔵", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{cls}**: {p:.2%}")
        else:
            st.error("❌ Gagal mendapatkan prediksi dari server.")

    except Exception as e:
        st.error(f"⚠️ Server error: {e}")

# ——— FOOTER ———
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.9em;'>© 2025 Obesity Risk App – Model: <b>Random Forest</b>, API: <b>FastAPI</b></p>",
    unsafe_allow_html=True
)
