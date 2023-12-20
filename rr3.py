import streamlit as st
import requests
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
import base64
import plotly.express as px
from datetime import datetime

local_time = datetime.now().strftime("%Y-%m-%d %H:%M")


# data = px.data.iris()

# @st.cache_data(ttl=600) 
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# img = get_img_as_base64("image.png")

# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
# background-image: url("https://i.pinimg.com/564x/72/69/31/726931f9a072c76bb560c091f9ec0979.jpg");
# background-size: 100%;
# background-position: top left;
# background-repeat: no-repeat;
# background-attachment: local;
# }}

# [data-testid="stSidebar"] > div:first-child {{
# background-image: url("data:image/png;base64,{img}");
# background-position: center; 
# background-repeat: no-repeat;
# background-attachment: fixed;
# }}

# [data-testid="stHeader"] {{
# background: rgba(0,0,0,0);
# }}

# [data-testid="stToolbar"] {{
# right: 2rem;
# }}
# </style>
# """

# st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
    <style>
        .centered-title {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Menampilkan judul rata tengah
st.markdown('<h1 class="centered-title" style="margin-bottom: 20px;">Prediksi Curah Hujan Wilayah Yogyakarta</h1>', unsafe_allow_html=True)

selected2 = option_menu(None, ["Home", "Prediksi Manual", "Tentang"], 
    icons=['house', 'water', 'cloud'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

with st.container():
    # Home Page
    if selected2 == "Home":
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
        
        new_prediction = loaded_model.predict(new_value)
        predicted_rainfall = max(0, new_prediction[0])  # Memastikan hasil prediksi tidak negatif

        
        # Prediction
        if st.button("Prediksi"):
    
        # Menggunakan dua kolom untuk menampilkan hasil prediksi
            col1, col2 = st.columns(2)

            # Menampilkan informasi di kolom kiri
            with col1:        
                st.markdown(f"<div style='display: flex; flex-direction: row; margin-bottom: 10px;'>"
                f"<div style='flex: 1; margin-right: 5px;'>"
                f"<b>Suhu:</b><br>{temp_c} °C"
                f"</div>"
                f"<div style='flex: 1;'>"
                f"<b>Kelembaban:</b><br>{humidity}%"
                f"</div>"
                f"</div>", unsafe_allow_html=True)
                
                st.markdown(
                    f"<div style='display: flex; flex-direction: row; margin-bottom: 10px;'>"
                    f"<div style='flex: 1; margin-right: 5px;'>"
                    f"<b>Waktu:</b><br>{local_time}"
                    f"</div>"
                    f"<div style='flex: 1;'>"
                    f"<b>Prediksi Curah Hujan:</b><br>{predicted_rainfall:.2f} mm/hari"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Menampilkan hasil prediksi dan kategori cuaca di kolom kanan
            with col2:

                # Make prediction
                new_prediction = loaded_model.predict(new_value)
                predicted_rainfall = max(0, new_prediction[0])  # Memastikan hasil prediksi tidak negatif

                # Kategorisasi cuaca berdasarkan intensitas hujan
                if predicted_rainfall == 0:
                    # Menampilkan gambar jika kategori adalah "Berawan"
                    st.image('awan.png', caption='Berawan', width=170)
                elif 0.5 < predicted_rainfall <= 20:
                    st.image('ringan.png', caption='Hujan Ringan', width=170)
                elif 21 < predicted_rainfall <= 50:
                    st.write("Weather Category: Hujan Sedang")
                elif 51 < predicted_rainfall <= 100:
                    st.write("Weather Category: Hujan Lebat")
                elif 101 < predicted_rainfall <= 150:
                    st.write("Weather Category: Hujan Sangat Lebat")
                else:
                    st.write("Weather Category: Hujan Ekstrem")

    elif selected2 == "Prediksi Manual":
        # Input user untuk suhu dan kelembaban menggunakan kolom input
        user_temp = st.number_input("Temperature (°C)", min_value=-10, max_value=40, value=25)
        user_humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
        loaded_model = joblib.load('linear_regression_model.joblib')

        # Tombol untuk prediksi
        if st.button("Prediksi"):
            # Format data untuk prediksi
            new_value = [[user_temp, user_humidity]]

            # Make prediction
            new_prediction = loaded_model.predict(new_value)
            predicted_rainfall = max(0, new_prediction[0])  # Memastikan hasil prediksi tidak negatif

            # Menggunakan dua kolom untuk menampilkan hasil prediksi
            col1, col2 = st.columns(2)

            # Menampilkan informasi di kolom kiri
            with col1:        
                st.markdown(f"<div style='display: flex; flex-direction: row; margin-bottom: 10px;'>"
                f"<div style='flex: 1; margin-right: 5px;'>"
                f"<b>Suhu:</b><br>{user_temp} °C"
                f"</div>"
                f"<div style='flex: 1;'>"
                f"<b>Kelembaban:</b><br>{user_humidity}%"
                f"</div>"
                f"</div>", unsafe_allow_html=True)
                
                st.markdown(
                    f"<div style='display: flex; flex-direction: row; margin-bottom: 10px;'>"
                    f"<div style='flex: 1; margin-right: 5px;'>"
                    f"<b>Waktu:</b><br>{local_time}"
                    f"</div>"
                    f"<div style='flex: 1;'>"
                    f"<b>Prediksi Curah Hujan:</b><br>{predicted_rainfall:.2f} mm/hari"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Menampilkan hasil prediksi dan kategori cuaca di kolom kanan
            with col2:

                # Make prediction
                new_prediction = loaded_model.predict(new_value)
                predicted_rainfall = max(0, new_prediction[0])  # Memastikan hasil prediksi tidak negatif

                # Kategorisasi cuaca berdasarkan intensitas hujan
                if predicted_rainfall == 0:
                    # Menampilkan gambar jika kategori adalah "Berawan"
                    st.image('awan.png', caption='Berawan', width=170)
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


    elif selected2 == "Tentang":
        
        
        st.title("Tentang Aplikasi")
            
        st.write(
            "Aplikasi ini menyajikan prediksi curah hujan di wilayah Yogyakarta "
            "berdasarkan suhu dan kelembaban. Menggunakan model Regresi Linier yang telah terlatih, "
            "aplikasi memberikan kategori cuaca, mulai dari kondisi berawan hingga hujan ekstrem. "
            "Data cuaca real-time diperoleh dari WeatherAPI untuk memperbarui prediksi dengan akurat. "
            "Aplikasi ini dirancang untuk memberikan informasi prakiraan cuaca yang lebih mudah dipahami "
            "dan dapat diakses secara manual oleh pengguna."
        )

                
        st.title("Langkah-langkah:")
        st.subheader("Pengumpulan Data")
        st.write("Dalam fase pengumpulan data, informasi diperoleh melalui ekstraksi dari data online Badan Meteorologi, Klimatologi, dan Geofisika (BMKG), selama periode yang berlangsung dari 1 Januari 2022 hingga 28 Oktober 2023." 
                 "Sebanyak 668 data meteorologi dikumpulkan untuk mendukung analisis dan pemahaman kondisi terkait curah hujan." 
                 "Data-data ini kemudian diubah ke dalam format file CSV untuk memudahkan proses pengolahan data selanjutnya.")
        st.write("Kemudian, penentuan variabel. Penentuan variabel independen yang digunakan melibatkan pemilihan dua parameter kunci, yaitu suhu (X1) dan kelembaban (X2)."
                 "Dua variabel ini dipilih karena memiliki potensi untuk memberikan kontribusi signifikan terhadap fenomena cuaca yang sedang diteliti."
                 "Sementara itu, variabel dependen dalam penelitian ini adalah curah hujan (Y), sebagai parameter sentral dalam proses analisis prediksi.")
        st.subheader("Metode Analisis")
        st.write(".")
        st.subheader("Data Preprocessing")
        st.write("1. Penghapusan data “NaN” dan 8888"
                 "Jumlah awal total data adalah dari 668 entri data. Setelah tahap penghapusan ini, jumlah data tersedia berkurang menjadi 626.")
        st.write("2. Menghapus kolom variabel yang tidak digunakan"
                 "Setelah kolom variabel yang tidak terpakai terhapus, maka hanya tersisa 3 kolom variabel yang akan digunakan untuk memprediksi curah hujan harian,"
                 "yaitu suhu (Tavg ), kelembaban (RH_avg), dan curah hujan (RR).")
        st.subheader("Model Training")
        st.write("A Linear Regression model is trained using the remaining data after preprocessing. The model is then saved "
                        "to 'linear_regression_model.joblib'.")

        st.subheader("Prediction")
        st.write("The app uses real-time weather data from WeatherAPI to predict rainfall. Click 'Predict' on the Home page "
                        "to see the predicted rainfall based on the trained model.")

        st.subheader("Weather Categories")
        st.write("The predicted rainfall is categorized into different weather conditions, such as 'Berawan', 'Hujan Ringan', "
                        "'Hujan Sedang', 'Hujan Lebat', 'Hujan Sangat Lebat', and 'Hujan Ekstrem'.")
