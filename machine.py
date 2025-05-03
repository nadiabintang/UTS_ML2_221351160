import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dari file
scaler = joblib.load("scaler.pkl")

# Load model TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="machine-failure-cleaned.tflite")
interpreter.allocate_tensors()

# Fungsi prediksi
def predict(input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Normalisasi data input jika diperlukan
    input_data_scaled = scaler.transform(input_data)  # Terapkan scaler pada input
    
    # Set tensor input dan lakukan inference
    interpreter.set_tensor(input_details[0]['index'], input_data_scaled)
    interpreter.invoke()

    # Ambil hasil prediksi
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output

# =============== KONFIGURASI HALAMAN ===============
st.set_page_config(page_title="Prediksi Kegagalan Mesin", layout="centered")

st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Arial', sans-serif;
            background-color: #f4f6f7;
        }
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            color: #1a237e;
            margin-bottom: 5px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #555;
            margin-bottom: 25px;
        }
        .result {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
        }
        .stButton>button {
            background-color: #1e88e5;
            color: white;
            border: none;
            padding: 0.6em 1.2em;
            border-radius: 6px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #1565c0;
        }
    </style>
""", unsafe_allow_html=True)

# =============== JUDUL ===============
st.markdown("<div class='title'>🛠️ Prediksi Kegagalan Mesin</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi Kerusakan Mesin dengan Model Neural Network Berdasarkan Data Sensor</div>", unsafe_allow_html=True)

# =============== FORM INPUT ===============
with st.form("machine_input"):
    col1, col2 = st.columns(2)
    with col1:
        rot_speed = st.slider("Rotational Speed [rpm]", 0, 3000, 1500)
        torque = st.slider("Torque [Nm]", 0.0, 100.0, 40.0)
        tool_wear = st.slider("Tool Wear [min]", 0, 300, 10)
    with col2:
        twf = st.radio("Tool Wear Failure (TWF)", [0, 1], horizontal=True)
        hdf = st.radio("Heat Dissipation Failure (HDF)", [0, 1], horizontal=True)
        pwf = st.radio("Power Failure (PWF)", [0, 1], horizontal=True)
        osf = st.radio("Overstrain Failure (OSF)", [0, 1], horizontal=True)

    submit = st.form_submit_button("🔍 Lakukan Prediksi")

# =============== PREDIKSI ===============
if submit:
    # Menyiapkan input data
    input_data = np.array([[rot_speed, torque, tool_wear, twf, hdf, pwf, osf]], dtype=np.float32)
    
    # Melakukan prediksi
    result = predict(input_data)
    
    # Mengambil probabilitas kegagalan dari output model
    prob = result[0][0]
    
    # Tentukan threshold
    threshold = 0.15
    is_failure = prob > threshold

    # Tampilkan hasil prediksi
    if is_failure:
        st.markdown(
            f"<div class='result' style='background-color:#ffebee; color:#c62828;'>❗ Mesin diprediksi  <b>GAGAL</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result' style='background-color:#e8f5e9; color:#2e7d32;'>✅ Mesin diprediksi <b>TIDAK GAGAL</b></div>",
            unsafe_allow_html=True
        )

    # Tampilkan probabilitas
    st.markdown(
        f"<p style='text-align:center; margin-top:10px;'>📊 Probabilitas Kegagalan: <b>{prob:.4f}</b></p>",
        unsafe_allow_html=True
    )
