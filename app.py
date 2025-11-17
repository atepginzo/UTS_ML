import re
import nltk
import joblib
from flask import Flask, request, render_template
from nltk.corpus import stopwords

# ---------------------------------------------------
# INETIALISASI & KONFIGURASI
# ---------------------------------------------------

# Unduh daftar 'stopwords' (hanya perlu sekali)
# Pastikan Anda menggunakan bahasa yang sama dengan saat melatih
# Ganti 'english' jika ulasan Anda dalam Bahasa Inggris
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# ---------------------------------------------------
# MEMUAT MODEL YANG SUDAH DILATIH
# ---------------------------------------------------
# Pastikan file-file ini ada di folder yang sama dengan app.py
try:
    model = joblib.load('model_sentiment.joblib')
    vectorizer = joblib.load('vectorizer_sentiment.joblib')
    print("--- Model dan Vectorizer Berhasil Dimuat ---")
except FileNotFoundError:
    print("KESALAHAN: File model/vectorizer tidak ditemukan!")
    print("Pastikan 'model_sentiment.joblib' dan 'vectorizer_sentiment.joblib' ada di folder proyek.")
    model = None
    vectorizer = None

# ---------------------------------------------------
# FUNGSI PREPROCESSING (HARUS SAMA PERSIS DENGAN DI COLAB)
# ---------------------------------------------------
def clean_text(text):
    """
    Fungsi untuk membersihkan teks input dari pengguna.
    Ini harus MENGGUNAKAN LOGIKA YANG SAMA PERSIS dengan saat Anda melatih model.
    """
    text = str(text).lower()  # 1. Ubah ke huruf kecil
    text = re.sub(r'[^\w\s]', '', text)  # 2. Hapus tanda baca
    
    # 3. Hapus stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# ---------------------------------------------------
# ROUTING APLIKASI WEB
# ---------------------------------------------------

@app.route('/')
def home():
    """
    Render halaman utama (index.html) saat pengguna
    pertama kali membuka aplikasi.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Ambil input dari form, proses, dan kembalikan
    hasil prediksi ke halaman.
    """
    if request.method == 'POST':
        # 1. Ambil teks ulasan dari form di HTML
        ulasan_asli = request.form['ulasan']
        
        if not ulasan_asli.strip():
            # Jika pengguna submit form kosong
            return render_template('index.html', 
                                   hasil_prediksi="Error: Teks ulasan tidak boleh kosong.", 
                                   ulasan_asli="")

        if model and vectorizer:
            # 2. Bersihkan teks input menggunakan fungsi preprocessing
            ulasan_bersih = [clean_text(ulasan_asli)]
            
            # 3. Ubah teks bersih menjadi vektor angka (TF-IDF)
            vektor_ulasan = vectorizer.transform(ulasan_bersih)
            
            # 4. Lakukan prediksi menggunakan model
            prediksi = model.predict(vektor_ulasan)
            
            # Ambil hasil prediksi pertama (karena kita hanya memprediksi 1 teks)
            hasil = prediksi[0]
            
        else:
            hasil = "Error: Model tidak berhasil dimuat."

        # 5. Kembalikan hasil ke halaman index.html
        return render_template('index.html', 
                               hasil_prediksi=hasil, 
                               ulasan_asli=ulasan_asli)

# ---------------------------------------------------
# MENJALANKAN APLIKASI
# ---------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True) # debug=True untuk mode pengembangan