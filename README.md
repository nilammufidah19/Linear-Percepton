# Run Process
```
pip install -r requirment.txt
python main.py
```

# Definisi Class & Function

## Class: `LinearPerceptron`
Implementasi perceptron linear dengan aktivasi sigmoid (logistic regression style). Mendukung training, evaluasi, logging, dan visualisasi grafik loss/accuracy.

`__init__(self, input_dim, learning_rate=0.1, n_iter=1000)`
Parameter:
- `input_dim` → jumlah fitur input.
- `learning_rate` → kecepatan pembelajaran (default = 0.1).
- `n_iter` → jumlah epoch training (default = 1000).

Atribut:
- `weights` → bobot linear (numpy array).
- `bias` → bias scalar.
- `history` → dictionary berisi train/val loss & accuracy.

`sigmoid(self, z)`
- Fungsi aktivasi sigmoid.
- Input: nilai linear kombinasi 
      `𝑧 = 𝑤 ⋅ 𝑥 + 𝑏`
- Output: nilai probabilitas antara 0–1.

`predict_proba(self, X)`
- Menghasilkan probabilitas output kelas.
- Input: data `X` (numpy array).
- Output: array probabilitas (float).

`predict(self, X, threshold=0.5)`
- Menghasilkan prediksi kelas biner (0/1).
- Input: data X, ambang batas probabilitas (threshold).
- Output: array label (0/1).

`compute_loss_accuracy(self, X, y)`
- Menghitung Binary Cross Entropy loss dan accuracy.
- Input: fitur `X`, label `y`.
- Output: `tuple (loss, accuracy)`.

`fit(self, X_train, y_train, X_val=None, y_val=None)`
- Melatih model menggunakan gradient descent.
- Mendukung evaluasi pada validation set di setiap epoch.
- Input:
  - `X_train`, `y_train` → data training.
  - `X_val`, `y_val` → data validasi (opsional).
- Output: training history tersimpan di `self.history`.

`plot_history(self)`
- Menampilkan grafik loss dan accuracy pada train/validation set.
- Membaca data dari `self.history`.
