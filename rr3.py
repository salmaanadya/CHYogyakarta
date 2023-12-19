import streamlit as st
import requests
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set the title of the Streamlit application
st.title("Prediksi Curah Hujan Wilayah Yogyakarta")

# Baca data dari file CSV
data = pd.read_csv('iklim.csv')

# Mengecek apakah ada nilai NaN dalam DataFrame sebelum dihapus
print("Jumlah nilai NaN sebelum dihapus:")
print(data.isnull().sum())

# Menghapus baris yang mengandung nilai NaN
data_cleaned = data.dropna()
data_cleaned.to_csv('data_cleaned2.csv', index=False)

# Mengecek apakah masih ada nilai NaN setelah penghapusan
print("\nJumlah nilai NaN setelah dihapus:")
print(data_cleaned.isnull().sum())

# Misalkan 'data_cleaned' adalah DataFrame Anda
# Berikut adalah kolom-kolom yang ingin dihapus
kolom_yang_dihapus = ['Tanggal', 'Tn', 'Tx', 'ff_avg', 'ddd_x']

# Menggunakan df.drop() untuk menghapus kolom-kolom tersebut
data_cleaned = data_cleaned.drop(kolom_yang_dihapus, axis=1)

# Menyimpan DataFrame setelah menghapus kolom ke file CSV
data_cleaned.to_csv('data_cleaned2.csv', index=False)

# Menampilkan DataFrame setelah menghapus kolom
print(data_cleaned)

# Mengecek apakah ada baris dengan nilai 8888 dalam DataFrame
has_8888 = (data_cleaned == 8888).any(axis=1).any()

# Jika terdapat baris dengan nilai 8888, hapus baris tersebut
if has_8888:
    # Mencari baris yang memiliki setidaknya satu kolom dengan nilai 8888
    rows_to_remove = data_cleaned[data_cleaned.eq(8888).any(axis=1)].index
    
    # Menghapus baris yang telah ditemukan
    data_cleaned.drop(rows_to_remove, inplace=True)
    print(f"Baris dengan setidaknya satu nilai 8888 dihapus. DataFrame setelah penghapusan:\n{data_cleaned}")
else:
    print("Tidak ada baris dengan nilai 8888 dalam DataFrame.")

# Select features and target
X = data_cleaned[['Tavg', 'RH_avg']]
y = data_cleaned['RR']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file using joblib
joblib.dump(model, 'linear_regression_model.joblib')

# Load the trained model from the file
loaded_model = joblib.load('linear_regression_model.joblib')

# Real-time Weather Data
api_url = "http://api.weatherapi.com/v1/current.json?key=d2337911bdf249d286495359232611&q=Yogyakarta&aqi=no"
rt_data_generator = requests.get(api_url).json()
temp_c = rt_data_generator["current"]["temp_c"]
humidity = rt_data_generator["current"]["humidity"]
local_time = rt_data_generator["location"]["localtime"]


# Format new data for prediction
new_value = [[temp_c, humidity]]

# Prediction
if st.button("Predict"):
    st.write("Prediction:")
    st.write(f"Local Time: {local_time}")
    st.write(f"Temperature: {temp_c}°C, Humidity: {humidity}")
    
    # Make prediction
    new_prediction = loaded_model.predict(new_value)
    predicted_rainfall = max(0, new_prediction[0])  # Memastikan hasil prediksi tidak negatif
    st.write(f"Predicted Rainfall: {predicted_rainfall} mm/hari")

    # Kategorisasi cuaca berdasarkan intensitas hujan
    if predicted_rainfall == 0:
        st.write("Weather Category: Berawan")
    elif 0.5 < predicted_rainfall <= 20:
        st.write("Weather Category: Hujan Ringan")
    elif 21 < predicted_rainfall <= 50:
        st.write("Weather Category: Hujan Sedang")
    elif 51 < predicted_rainfall <= 100:
        st.write("Weather Category: Hujan Lebat")
    elif 101 < predicted_rainfall <= 150:
        st.write("Weather Category: Hujan Sangat Lebat")
    else:
        st.write("Weather Category: Hujan Ekstrem")

# Custom CSS for background color
st.markdown(
    """
    <style>
        body {
            background-color: #A2C8DF;
            background: linear-gradient(to bottom, #C6DBEA, #A2C8DF);
        }
    </style>
    """,
    unsafe_allow_html=True
)