import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Prediksi Dasar Penipuan Transaksi Keuangan Digital Menggunakan K-Nearest Neighbors (KNN)",
    page_icon="💳",
    layout="wide"
)

# Load model dan scaler
model = joblib.load("model.pkl")

# Fitur yang dibutuhkan
FEATURES = [
    'amount',
    'transaction_type',
    'merchant_category',
    'location',
    'device_used',
    'hour_of_day',
    'day_of_week',
    'is_night_transaction'
]

# Label encoder mapping
label_encoders = {
    'transaction_type': {'withdrawal': 0, 'deposit': 1, 'transfer': 2, 'payment': 3},
    'merchant_category': {'utilities': 0, 'online': 1, 'other': 2, 'entertainment': 3, 'travel': 4, 'grocery': 5, 'retail': 6, 'restaurant': 7},
    'location': {'Tokyo': 0, 'Toronto': 1, 'London': 2, 'Sydney': 3, 'Berlin': 4, 'Dubai': 5, 'New York': 6, 'Singapore': 7},
    'device_used': {'mobile': 0, 'atm': 1, 'pos': 2, 'web': 3}
}


st.title("Prediksi Dasar Penipuan Transaksi Keuangan 💳")

# Input form
with st.form("fraud_form"):
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Nominal Transaksi (USD)", min_value=0.0, step=10.0)
        transaction_type = st.selectbox("Tipe Transaksi", list(label_encoders['transaction_type'].keys()))
        merchant_category = st.selectbox("Kategori Merchant", list(label_encoders['merchant_category'].keys()))
        device_used = st.selectbox("Perangkat", list(label_encoders['device_used'].keys()))
    
    with col2:
        hour_of_day = st.slider("Jam Transaksi", 0, 23)
        location = st.selectbox("Lokasi", list(label_encoders['location'].keys()))
        day_of_week = st.selectbox("Hari Transaksi", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        submit = st.form_submit_button("Prediksi", use_container_width=True, type="primary")
    
    is_night_transaction = 1 if hour_of_day < 6 or hour_of_day > 22 else 0


# Mapping hari
day_map = {
    "Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3,
    "Jumat": 4, "Sabtu": 5, "Minggu": 6
}

if submit:
    # Encoding fitur kategorikal
    input_data = pd.DataFrame([{
        'amount': amount,
        'transaction_type': label_encoders['transaction_type'][transaction_type],
        'merchant_category': label_encoders['merchant_category'][merchant_category],
        'location': label_encoders['location'][location],
        'device_used': label_encoders['device_used'][device_used],
        'hour_of_day': hour_of_day,
        'day_of_week': day_map[day_of_week],
        'is_night_transaction': is_night_transaction
    }])

    prediction = model.predict(input_data.values)
    prob = model.predict_proba(input_data.values)[0][1]

    st.subheader("🔍 Hasil Prediksi")
    if prediction == 1:
        st.error(f"⚠️ Transaksi ini terindikasi **Penipuan**")
    else:
        st.success(f"✅ Transaksi ini **Normal**")


# Footer
st.markdown("""---""")
st.markdown(
    "<div style='text-align: center; color: gray;'>© Fiki Pratama (22.12.2551)</div>",
    unsafe_allow_html=True
)
