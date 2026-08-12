import streamlit as st
import pandas as pd
import joblib

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Konfigurasi Alat Tambang",
    layout="centered"
)

# ============================================================
# LOAD MODEL, SCALER, DAN LABEL ENCODER
# Pastikan file .pkl ini ada di folder yang sama dengan app.py
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_konfigurasi_alat.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, scaler, le

model, scaler, le = load_artifacts()

FEATURES = [
    "Quantity_Avg",
    "Velocity_Avg",
    "Temp_Avg",
    "RH_Avg"
]

# ============================================================
# HEADER
# ============================================================
st.title("Konfigurasi ALat Berat Tambang Bawah Tanah")
st.markdown(
    "Aplikasi ini memprediksi rekomendasi **Konfigurasi Alat** "
    "(kombinasi Diesel/Listrik untuk Mine Truck, Wheel Loader, LHD, "
    "dan Jumbo Drill) berdasarkan **kondisi ventilasi bawah tanah** "
    "yang ada di lokasi kerja."
)
st.divider()

# ============================================================
# INPUT USER — parameter ventilasi
# ============================================================
st.subheader("Masukkan Kondisi Ventilasi")

col1, col2 = st.columns(2)

with col1:
    quantity_avg = st.number_input(
        "Quantity Average (m^3/s)",
        min_value=0.0, value=0.0, step=0.1,
        help="Rata-rata debit/volume aliran udara ventilasi"
    )
    velocity_avg = st.number_input(
        "Velocity Average (m/s)",
        min_value=0.0, value=0.0, step=0.1,
        help="Rata-rata kecepatan aliran udara"
    )

with col2:
    temp_avg = st.number_input(
        "Temperature Average (°C)",
        min_value=0.0, value=0.0, step=0.1,
        help="Rata-rata suhu di area kerja bawah tanah"
    )
    rh_avg = st.number_input(
        "Relative Humidity Average (%)",
        min_value=0.0, max_value=100.0, value=0.0, step=0.1,
        help="Rata-rata kelembapan relatif di area kerja"
    )

st.divider()

# ============================================================
# PREDIKSI
# ============================================================
if st.button("Prediksi Konfigurasi Alat", use_container_width=True):

    input_data = pd.DataFrame(
        [[quantity_avg, velocity_avg, temp_avg, rh_avg]],
        columns=FEATURES
    )

    # Scaling menggunakan scaler yang sama seperti saat training
    input_scaled = scaler.transform(input_data)

    # Prediksi
    pred_encoded = model.predict(input_scaled)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    st.success(f"### Rekomendasi Konfigurasi Alat: **{pred_label}**")

    # Tampilkan probabilitas tiap kelas jika model mendukung predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        proba_df = pd.DataFrame({
            "Konfigurasi": le.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)

        st.subheader("Distribusi Probabilitas Tiap Konfigurasi")
        st.dataframe(
            proba_df.style.format({"Probabilitas": "{:.2%}"}),
            use_container_width=True,
            hide_index=True
        )
        st.bar_chart(proba_df.set_index("Konfigurasi")["Probabilitas"])

    st.caption(
        "⚠️ Prediksi ini berdasarkan pola historis dari data ventilasi dan "
        "konfigurasi alat sebelumnya. Gunakan sebagai referensi awal, "
        "tetap pertimbangkan faktor teknis dan keselamatan lain di lapangan."
    )

st.divider()
st.caption("Model klasifikasi konfigurasi alat tambang · Dibuat dengan Streamlit")
