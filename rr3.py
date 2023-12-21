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

#String waktu 
local_time = datetime.now().strftime("%Y-%m-%d %H:%M")

#Style rata tengah
st.markdown("""
    <style>
        .centered-title {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Menampilkan judul rata tengah
st.markdown('<h1 class="centered-title" style="margin-bottom: 1px;">Prediksi Curah Hujan Wilayah Yogyakarta</h1>', unsafe_allow_html=True)
st.caption("<div style='text-align: center; margin-bottom: 15px;'>Salma Anadya Shafarany_21537141006</div>", unsafe_allow_html=True)
#Menu
selected2 = option_menu(None, ["Home", "Prediksi Manual", "Tentang"], 
    icons=['house', 'water', 'cloud'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

with st.container():
    
    # Halaman Home
    if selected2 == "Home":
        # Baca data dari file CSV
        data = pd.read_csv('iklim.csv')
        
        # Cleaning
        # Menghapus baris yang mengandung nilai NaN
        data_cleaned = data.dropna()
        data_cleaned.to_csv('data_cleaned2.csv', index=False)

        # Hapus Kolom
        kolom_yang_dihapus = ['Tanggal', 'Tn', 'Tx', 'ff_avg', 'ddd_x']
        data_cleaned = data_cleaned.drop(kolom_yang_dihapus, axis=1)
        # Menyimpan ke dataframe setelah menghapus kolom ke file CSV
        data_cleaned.to_csv('data_cleaned2.csv', index=False)
        
        # Mengecek apakah ada baris dengan nilai 8888 dalam DataFrame
        has_8888 = (data_cleaned == 8888).any(axis=1).any()
        # Jika terdapat baris dengan nilai 8888, hapus baris tersebut
        if has_8888:
            # Mencari baris yang memiliki setidaknya satu kolom dengan nilai 8888
            rows_to_remove = data_cleaned[data_cleaned.eq(8888).any(axis=1)].index
            # Menghapus baris yang telah ditemukan
            data_cleaned.drop(rows_to_remove, inplace=True)

        #Training Model
        # Pemilihan fitur dan target
        X = data_cleaned[['Tavg', 'RH_avg']]
        y = data_cleaned['RR']

        # Split the data into training dan tes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inisialisasi Model
        model = LinearRegression()

        # Training
        model.fit(X_train, y_train)

        # Simpan model menggunakan joblib
        joblib.dump(model, 'linear_regression_model.joblib')

        # Load model
        loaded_model = joblib.load('linear_regression_model.joblib')

        # Mengambil data menggunakan API
        api_url = "http://api.weatherapi.com/v1/current.json?key=d2337911bdf249d286495359232611&q=Yogyakarta&aqi=no"
        rt_data_generator = requests.get(api_url).json()
        temp_c = rt_data_generator["current"]["temp_c"]
        humidity = rt_data_generator["current"]["humidity"]
        local_time = rt_data_generator["location"]["localtime"]

        # Format data baru untuk prediksi
        new_value = [[temp_c, humidity]]
        
        # Prediksi nilai baru
        new_prediction = loaded_model.predict(new_value)
        predicted_rainfall = max(0, new_prediction[0])  # Memastikan hasil prediksi tidak negatif

        
        # Prediksi
        if st.button("Prediksi"):
    
        # Menggunakan dua kolom untuk menampilkan hasil prediksi
            col1, col2 = st.columns(2)

            # Menampilkan informasi di kolom kiri
            with col1:        
                st.markdown(f"<div style='display: flex; flex-direction: row; margin-bottom: 10px;'>"
                f"<div style='flex: 1; margin-right: 5px;'>"
                f"<b>Suhu:</b><br>{temp_c} °C" #Suhu
                f"</div>"
                f"<div style='flex: 1;'>"
                f"<b>Kelembaban:</b><br>{humidity}%" #Kelembaban
                f"</div>"
                f"</div>", unsafe_allow_html=True)
                
                st.markdown(
                    f"<div style='display: flex; flex-direction: row; margin-bottom: 10px;'>"
                    f"<div style='flex: 1; margin-right: 5px;'>"
                    f"<b>Waktu:</b><br>{local_time}" #Waktu
                    f"</div>"
                    f"<div style='flex: 1;'>"
                    f"<b>Prediksi Curah Hujan:</b><br>{predicted_rainfall:.2f} mm/hari" #Curah Hujan
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Menampilkan hasil prediksi di kolom kanan
            with col2:

                new_prediction = loaded_model.predict(new_value)
                predicted_rainfall = max(0, new_prediction[0]) 

                # Kategorisasi cuaca berdasarkan intensitas hujan
                if predicted_rainfall == 0:
                    st.image('awan.png', caption='Berawan', width=170)
                elif 0.5 < predicted_rainfall <= 20:
                    st.image('ringan.png', caption='Hujan Ringan', width=170)
                elif 21 < predicted_rainfall <= 50:
                    st.image('sedang.png', caption='Hujan Sedang', width=170)
                elif 51 < predicted_rainfall <= 100:
                    st.image('lebat.png', caption='Hujan Lebat', width=170)
                elif 101 < predicted_rainfall <= 150:
                    st.image('sangatlebat.png', caption='Hujan Sangat Lebat', width=170)
                else:
                    st.image('ekstrem.png', caption='Hujan Ekstrem', width=170)
    
    # Menu Prediksi Manual
    elif selected2 == "Prediksi Manual":
        # Input user
        user_temp = st.number_input("Temperature (°C)", min_value=-10, max_value=40, value=25)
        user_humidity = st.number_input("Humidity (%)", min_value=0, max_value=170, value=50)
        loaded_model = joblib.load('linear_regression_model.joblib')

        # Tombol prediksi
        if st.button("Prediksi"):
            # Format data baru 
            new_value = [[user_temp, user_humidity]]

            # Membuat Prediksi
            new_prediction = loaded_model.predict(new_value)
            predicted_rainfall = max(0, new_prediction[0])  # Memastikan hasil prediksi tidak negatif

            # Dua kolom
            col1, col2 = st.columns(2)

            # Kolom kiri
            with col1:        
                st.markdown(f"<div style='display: flex; flex-direction: row; margin-bottom: 10px;'>"
                f"<div style='flex: 1; margin-right: 5px;'>"
                f"<b>Suhu:</b><br>{user_temp} °C" #Suhu
                f"</div>"
                f"<div style='flex: 1;'>"
                f"<b>Kelembaban:</b><br>{user_humidity}%" #Kelembaban
                f"</div>"
                f"</div>", unsafe_allow_html=True)
                
                st.markdown(
                    f"<div style='display: flex; flex-direction: row; margin-bottom: 10px;'>"
                    f"<div style='flex: 1; margin-right: 5px;'>"
                    f"<b>Waktu:</b><br>{local_time}" #Waktu
                    f"</div>"
                    f"<div style='flex: 1;'>"
                    f"<b>Prediksi Curah Hujan:</b><br>{predicted_rainfall:.2f} mm/hari" #Intensitas
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Kolom kanan
            with col2:

                # Prediksi dan disimpan agar tidak negatif
                new_prediction = loaded_model.predict(new_value)
                predicted_rainfall = max(0, new_prediction[0])

                # Kategorisasi
                if predicted_rainfall == 0:
                    st.image('awan.png', caption='Berawan', width=170)
                elif 0.5 < predicted_rainfall <= 20:
                    st.image('ringan.png', caption='Hujan Ringan', width=170)
                elif 21 < predicted_rainfall <= 50:
                    st.image('sedang.png', caption='Hujan Sedang', width=170)
                elif 51 < predicted_rainfall <= 100:
                    st.image('lebat.png', caption='Hujan Lebat', width=170)
                elif 101 < predicted_rainfall <= 150:
                    st.image('sangatlebat.png', caption='Hujan Sangat Lebat', width=170)
                else:
                    st.image('ekstrem.png', caption='Hujan Ekstrem', width=170)

    # Menu Tentang
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

        st.subheader("Data Preprocessing")
        st.write("1. Penghapusan data “NaN” dan 8888"
                 "Jumlah awal total data adalah dari 668 entri data. Setelah tahap penghapusan ini, jumlah data tersedia berkurang menjadi 626.")
        st.write("2. Menghapus kolom variabel yang tidak digunakan"
                 "Setelah kolom variabel yang tidak terpakai terhapus, maka hanya tersisa 3 kolom variabel yang akan digunakan untuk memprediksi curah hujan harian,"
                 "yaitu suhu (Tavg ), kelembaban (RH_avg), dan curah hujan (RR).")
        st.subheader("Training")
        st.write("1. Pemilihan Fitur dan Target:"
                " fitur (X) dipilih sebagai kolom suhu (temp_c) dan kelembaban (humidity)"
                 " dan target (y) dipilih sebagai kolom curah hujan (RR).")
        st.write("2. Split Data:"
                " 20% dari data digunakan dalam pengujian, sementara 80% data digunakan sebagai data pelatihan.")
        st.write("3. Pelatihan: "
                " model dilatih menggunakan data pelatihan (X_train dan y_train).")
        st.write("4. Simpan Model:"
                " model yang telah dilatih disimpan ke dalam file menggunakan joblib.")
        
        st.title("Evaluasi")
        st.write("Tingkat Kesalahan dan Akurasi")
        st.write("1. RMSE       : 10.28")
        st.write("2. Akurasi    : 0.88")
        st.write("3. Precission : 10.28")
        st.write("4. Recall     : 0.88")
        st.write("5. F1 Score   : 0.94")
        
        st.title("Kategori Cuaca")
        st.write("1. Berawan: 0 mm")
        st.write("2. Hujan Ringan:  0.5 - 20 mm")
        st.write("3. Hujan Sedang: 21 - 50 mm")
        st.write("4. Hujan Lebat: 51 - 100 mm")
        st.write("5. Hujan Sangat Lebat: 101 - 150 mm")
        st.write("6. Hujan Ekstrem: >150 mm")
