# Import library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pickle
import os
import joblib

# Class untuk menangani seluruh proses data (cleaning, encoding, dsb)
class DataHandler:
    def __init__(self, file_path):
        # Inisialisasi path file dan atribut lainnya
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        # Load data dari file CSV ke dalam dataframe
        self.data = pd.read_csv(self.file_path)
    
    def fill_missing_values(self, df):
        # Menangani nilai yang hilang di beberapa kolom penting
        df = df.copy()

        df['type_of_meal_plan'] = df['type_of_meal_plan'].replace('Not Selected', 'Meal Plan 1')
        modus = df['type_of_meal_plan'].mode()[0]
        df['type_of_meal_plan'].fillna(modus, inplace=True)

        df['required_car_parking_space'].fillna(df['required_car_parking_space'].mode()[0], inplace=True)

        df['avg_price_per_room'].fillna(df['avg_price_per_room'].median(), inplace=True)

        return df
    
    def drop_column(self, column_name):
        # Menghapus kolom yang tidak diperlukan dari dataframe
        if column_name in self.data.columns:
            self.data = self.data.drop(column_name, axis=1)
    
    def label_encode(self, column_name):
        # Melakukan label encoding terhadap kolom kategorikal tertentu 
        label_encoder = preprocessing.LabelEncoder()
        self.data[column_name] = label_encoder.fit_transform(self.data[column_name])
    
    def onehot_encode(self, df):
        # Melakukan one-hot encoding untuk kolom tyoe_of_meal_plan, room_type_reserved, dan market_segment_type
        df = df.copy()
        categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        self.encoder = OneHotEncoder(sparse_output=False, drop=None)
        encoded_array = self.encoder.fit_transform(df[categorical_cols])
        self.ohe_columns = self.encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_array, columns=self.ohe_columns, index=df.index)
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def create_input_output(self, target_column):
        # Memisahkan antara fitur (input) dan target (output)
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

# Class untuk menangani proses model machine learning
class ModelHandler:
    def __init__(self, input_data, output_data):
        # Inisialisasi data input dan output, serta buat model kosong
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
    
    def split_data(self, test_size=0.2, random_state=42):
        # Membagi data menjadi train dan test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def createModel(self):
         # Membuat model Random Forest
         self.RF_class = RandomForestClassifier()

    def train_model(self):
        # Melatih model dengan data training
        self.RF_class.fit(self.x_train, self.y_train)

    def makePrediction(self):
        # Membuat prediksi dari data test
        self.y_predict = self.RF_class.predict(self.x_test) 

    def evaluate_model(self):
        # Mengembalikan skor akurasi dari prediksi
        pred = self.RF_class.predict(self.x_test)
        return accuracy_score(self.y_test, pred)
    
    def createReport(self):
        # Menampilkan classification report secara lengkap
        print('\nClassification Report of Random Forest\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['Canceled','Not_Canceled']))
    
    def save_model_to_file(self, save_path, filename):
        # Menyimpan model ke dalam file pickle menggunakan joblib
        file_path = os.path.join(save_path, filename)
        with open(file_path, 'wb') as file:
            joblib.dump(self.RF_class, file, compress=3)

#Memuat data dan model
file_path = 'D:/Semester 4/Model Deployment/Model Deployment Mid Exam/Dataset_B_hotel.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.data = data_handler.fill_missing_values(data_handler.data)
data_handler.drop_column('Booking_ID')
data_handler.label_encode('arrival_year')
data_handler.label_encode('booking_status')
data_handler.data = data_handler.onehot_encode(data_handler.data)
data_handler.create_input_output('booking_status')

model_handler = ModelHandler(data_handler.input_df, data_handler.output_df)
model_handler.split_data()
model_handler.train_model()
model_handler.makePrediction()
print("Accuracy:", model_handler.evaluate_model())
model_handler.createReport()
model_handler.save_model_to_file(save_path='D:/Semester 4/Model Deployment/Model Deployment Mid Exam', filename='random_forest_model.pkl')