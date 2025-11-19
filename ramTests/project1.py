# ram_test.py
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler
from memory_profiler import profile
import time

# --- 1. GEREKLİ ARAÇLARI HAZIRLA ---
# Bu script, scaler'ı (dönüştürücüyü) yeniden "eğitmek" zorunda
# (Normalde scaler'ı 'joblib.dump()' ile kaydeder ve 'joblib.load()' ile yüklerdik)

print("Scaler (Dönüştürücü) 'temsili' verilerle yeniden eğitiliyor...")
np.random.seed(42)
dataNum = 100
noise = 0.1
# Orijinal Proje 1'deki X_train verisini (160 satır) temsil etmesi için 
# 200 satırlık tam veri setini (xTotal) yeniden oluşturuyoruz
Duz_x = np.random.normal(0.0, scale=noise, size=dataNum)
Duz_y = np.random.normal(0.0, scale=noise, size=dataNum)
Duz_z = np.random.normal(9.8, scale=noise, size=dataNum)
Ters_x = np.random.normal(0.0, scale=noise, size=dataNum)
Ters_y = np.random.normal(0.0, scale=noise, size=dataNum)
Ters_z = np.random.normal(-9.8, scale=noise, size=dataNum)

xDuz = np.column_stack((Duz_x, Duz_y, Duz_z))
xTers = np.column_stack((Ters_x, Ters_y, Ters_z))
xTotal = np.vstack((xDuz, xTers)) # Bu bizim (200, 3)'lük tam X verimiz

# Scaler'ı bu tam veriyle eğitelim (X_train yerine, hızlı olması için)
scaler = StandardScaler()
scaler.fit(xTotal)
print("Scaler eğitildi.")

# Her iki model için de aynı 'yeni' soruyu hazırlayalım
# Soru: [0, 0, -9.8] (Ters Durum)
yeni_veri_float32 = np.array([[0.0, 0.0, -9.8]], dtype=np.float32)
yeni_veri_scaled_float32 = scaler.transform(yeni_veri_float32)


# --- 2. "ŞİŞMAN" (float32) MODELİ TEST ET ---
print("\n" + "="*30)
print("TEST 1: 'ŞİŞMAN' (float32) MODEL (.h5)")
print("="*30)

# "Usta İşi" hile: @profile, bu fonksiyonun RAM kullanımını satır satır ölçer
@profile
def test_float32_model():
    print("Model (float32) RAM'e yükleniyor...")
    # 1. Yükle (Diskten RAM'e)
    model_h5 = tf.keras.models.load_model('../unoptimizedmodel.h5')
    
    print("Tahmin (float32) yapılıyor...")
    time.sleep(1) # Ölçümün doğru yapılması için 1sn bekle
    
    # 2. Tahmin et (RAM sıçraması burada olur)
    tahmin = model_h5.predict(yeni_veri_scaled_float32)
    print(f"Tahmin (float32): {tahmin[0][0]:.4f}")

# --- 3. "ZAYIF" (int8) MODELİ TEST ET ---
print("\n" + "="*30)
print("TEST 2: 'ZAYIF' (int8) MODEL (.tflite)")
print("="*30)

@profile
def test_int8_model():
    print("Model (int8) RAM'e yükleniyor...")
    # TFLite modeli 'Interpreter' ile yüklenir (farklı bir yol)
    # USTA İŞİ DÜZELTME: Senin kaydettiğin adı kullanıyoruz ("optimizedmodel.h5")
    # (Bunun .tflite olması gerektiğini unutmuyoruz!)
    interpreter = tf.lite.Interpreter(model_path='../optimizedmodel.tflite')
    interpreter.allocate_tensors() # Modeli RAM'e "açar"
    
    # Giriş ve çıkış detaylarını al
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # --- USTA İŞİ int8 GİRİŞ HAZIRLIĞI ---
    # Modelimizin girişi artık float32 DEĞİL, int8 bekliyor!
    # float32 verimizi (yeni_veri_scaled_float32) int8'e kuantize etmeliyiz.
    input_scale, input_zero_point = input_details[0]['quantization']
    yeni_veri_quantized_int8 = (yeni_veri_scaled_float32 / input_scale) + input_zero_point
    yeni_veri_quantized_int8 = yeni_veri_quantized_int8.astype(np.int8) # Tipi int8'e zorla
    # --- BİTTİ ---

    print("Tahmin (int8) yapılıyor...")
    time.sleep(1) # Ölçümün doğru yapılması için 1sn bekle

    # 2. Tahmin et (RAM sıçraması burada olur)
    interpreter.set_tensor(input_details[0]['index'], yeni_veri_quantized_int8)
    interpreter.invoke()
    
    tahmin_quantized = interpreter.get_tensor(output_details[0]['index'])
    
    # Çıkış da int8 olduğu için, onu geri 'de-kuantize' etmeliyiz
    output_scale, output_zero_point = output_details[0]['quantization']
    tahmin_float32 = (tahmin_quantized.astype(np.float32) - output_zero_point) * output_scale
    
    print(f"Tahmin (int8 -> float32): {tahmin_float32[0][0]:.4f}")

# --- 4. TESTLERİ ÇALIŞTIR ---
if __name__ == '__main__':
    test_float32_model()
    test_int8_model()