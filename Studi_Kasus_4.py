import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data_properti = {
    'Luas_Tanah_m2': [120, 150, 80, 200, 95, 180, 110, 130, 160, 140],
    'Jumlah_Kamar': [2, 3, 1, 4, 2, 3, 2, 3, 3, 3],
    'Harga_Juta_Rupiah': [600, 750, 400, 1000, 480, 900, 550, 700, 820, 720]
}

df = pd.DataFrame(data_properti)

X = df[['Luas_Tanah_m2', 'Jumlah_Kamar']]
y = df['Harga_Juta_Rupiah']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("--- HASIL EVALUASI MODEL MACHINE LEARNING ---")
print(f"Skor Akurasi R-squared (R^2): {round(r2, 2)}")

rumah_baru = pd.DataFrame([[170, 3]], columns=['Luas_Tanah_m2', 'Jumlah_Kamar'])
prediksi_harga = model.predict(rumah_baru)
print(f"Estimasi Prediksi Harga Rumah Baru: Rp {round(prediksi_harga[0], 2)} Juta")