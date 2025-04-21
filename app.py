import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os

class BookingStatusPredictor:
    def __init__(self, model_file, enc_info, attribute_list):
        # Inisialisasi kelas dengan memuat model yang sudah disimpan (pickle), informasi encoding, dan daftar atribut yang dipakai saat pelatihan.
        self.model = joblib.load(model_file)
        self.enc_info = enc_info
        self.attribute_list = attribute_list
        self.encoders = {}
    
    def encode_data(self, df_input):
        # Melakukan encoding pada data input sesuai dengan informasi encoding yang diberikan.
        # Menggunakan LabelEncoder untuk kolom yang perlu di-label encoded dan OneHotEncoder untuk kolom yang perlu di-one-hot encode.
        labelEnc = LabelEncoder()
        for column in self.enc_info['Label_Encode']:
            df_input[column] = labelEnc.fit_transform(df_input[column])
        
        
        ohe_list = []
        for column in self.enc_info['One_Hot_Encode']:
            if column not in self.encoders:
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(df_input[[column]])
                self.encoders[column] = enc

            enc = self.encoders[column]
            encoded_data = enc.transform(df_input[[column]])
            cols = enc.get_feature_names_out([column]) 
            ohe_list.append(pd.DataFrame(encoded_data, columns=cols))

        df_input = df_input.reset_index(drop=True)
        df_input = pd.concat([df_input.drop(columns=self.enc_info['One_Hot_Encode'])] + ohe_list, axis=1)

        df_input = df_input.reindex(columns=self.attribute_list, fill_value=0)
        return df_input
    
    def predict(self, df_input):
        # Melakukan prediksi menggunakan model yang sudah dilatih.
        new_data = self.encode_data(df_input)
        return self.model.predict(new_data)

# Membuat design dari app yang akan dimuat
def main():
    st.title("Hotel Booking Status Predictor")
    st.subheader("Memprediksi apakah pemesanan hotel akan dibatalkan atau tidak")
    st.write("Silakan isi rincian di bawah ini:")

    user_input = {
        'no_of_adults': st.number_input('Jumlah orang dewasa di dalam keluarga', 0, 10),
        'no_of_children': st.number_input('Jumlah anak kecil di dalam keluarga', 0, 10),
        'no_of_weekend_nights': st.number_input('Jumlah malam akhir pekan', 0, 7),
        'no_of_week_nights': st.number_input('Jumlah malam dalam seminggu', 0, 20),
        'type_of_meal_plan': st.selectbox('Jenis paket makanan yang dipesan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
        'required_car_parking_space': st.selectbox('Tempat parkir mobil (0 : Tidak Butuh, 1 : Butuh)', [0, 1]),
        'room_type_reserved': st.selectbox('Jenis kamar yang dipesan', [
            'Room Type 1', 'Room Type 2', 'Room Type 3', 'Room Type 4', 
            'Room Type 5', 'Room Type 6', 'Room Type 7'
        ]),
        'lead_time': st.number_input('Lead Time', 0, 1000),
        'arrival_year': st.number_input('Tahun Kedatangan', 2000, 2023, value=2017),
        'arrival_month': st.number_input('Bulan Kedatangan', 1, 12),
        'arrival_date': st.number_input('Tanggal Kedatangan', 1, 31),
        'market_segment_type': st.selectbox('Segmen Pasar', ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online']),
        'repeated_guest': st.selectbox('Tamu Berulang (0 : Pernah Booking, 1 : Tidak Pernah Booking)', [0, 1]),
        'no_of_previous_cancellations': st.number_input('Jumlah pembatalan sebelumnya', 0, 20),
        'no_of_previous_bookings_not_canceled': st.number_input('Jumlah booking berhasil sebelumnya', 0, 100),
        'avg_price_per_room': st.number_input('Harga rata-rata kamar', 0.0, 100000.0),
        'no_of_special_requests': st.number_input('Permintaan khusus', 0, 5)
    }

    if st.button('Prediksi'):
        df_input = pd.DataFrame([user_input])

        model_path = 'random_forest_model.pkl'
        encoded_columns = {
            'Label_Encode': ['arrival_year'],
            'One_Hot_Encode': ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        }
        attribute_list = [
            'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
            'required_car_parking_space', 'lead_time', 'arrival_year', 'arrival_month',
            'arrival_date', 'repeated_guest', 'no_of_previous_cancellations',
            'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests',
            'type_of_meal_plan_Meal Plan 1', 'type_of_meal_plan_Meal Plan 2', 
            'type_of_meal_plan_Meal Plan 3', 'type_of_meal_plan_Not Selected',
            'room_type_reserved_Room_Type 1', 'room_type_reserved_Room_Type 2',
            'room_type_reserved_Room_Type 3', 'room_type_reserved_Room_Type 4',
            'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6',
            'room_type_reserved_Room_Type 7', 'market_segment_type_Aviation',
            'market_segment_type_Complementary', 'market_segment_type_Corporate',
            'market_segment_type_Offline', 'market_segment_type_Online'
        ]


        predictor = BookingStatusPredictor(model_path, encoded_columns, attribute_list)
        hasil_prediksi = predictor.predict(df_input)

        label_prediksi = "Not Canceled" if hasil_prediksi[0] == 1 else "Canceled"
        print(df_input.iloc[0])
        st.success(f"Hasil Prediksi: {label_prediksi}")

if __name__ == '__main__':
    main()