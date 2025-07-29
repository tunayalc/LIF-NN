
# LIF_MODEL.py - Leaky Integrate-and-Fire (LIF) Sinir Ağı Modeli

Bu proje, biyolojik sinir sistemlerinden ilham alan Spike Tabanlı Sinir Ağı (Spiking Neural Network - SNN) modellemesini amaçlar. Model, **Leaky Integrate-and-Fire (LIF)** tipi nöron davranışını kullanarak sinyalleri zamana bağlı olarak işler. Ayrıca **Spike-Timing Dependent Plasticity (STDP)** ile öğrenme gerçekleştirilir ve çıktılar **Boltzmann** olasılık modeliyle değerlendirilir.

---

## İçerik Özeti

### 1. Kullanılan Kütüphaneler

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from tqdm import tqdm
import warnings
import os
```

- `numpy`: Sayısal işlemler
- `matplotlib`, `seaborn`: Görselleştirme
- `pandas`: Veri yükleme ve düzenleme
- `sklearn`: Performans metrikleri ve ölçekleme
- `tqdm`: İlerleme göstergesi
- `warnings`, `os`: Uyarı ve dosya kontrolü

---

### 2. Sınıf: `BoltzmannSpikingNeuralNetwork`

```python
class BoltzmannSpikingNeuralNetwork:
```

Bu sınıf, LIF tabanlı bir nöral ağ yapısını tanımlar. Parametreler:

- `input_size`: Girdi boyutu (örn. MNIST için 784)
- `hidden_size`: Gizli katman nöron sayısı
- `output_size`: Sınıf sayısı (örn. 10)

Model ayrıca STDP ve Boltzmann güncellemeleri için çeşitli katsayılar ve zaman sabitleri tanımlar.

---

### 3. Ana Metotlar

#### `lif_spike_generation(input_data)`
Girdi verisini zaman boyunca spike (0/1) dizisine çevirir.

#### `train(X_train, y_train, ...)`
Modelin eğitildiği ana döngüdür. Ağırlıklar STDP ve softmax tabanlı Boltzmann çıktılar üzerinden güncellenir.

#### `predict(X_test)`
Test verisiyle ileri besleme yapar ve tahminleri döndürür.

#### `softmax(x)`
Numerik stabil softmax uygulaması (satır bazlı).

#### `load_data(path)`
CSV dosyasından veri yükleyip eğitim/test olarak ayırır.

---

### 4. Ana Çalıştırma Akışı (`main`)

```python
if __name__ == "__main__":
    ...
```

- Veriyi yükler
- Modeli başlatır
- Eğitir
- Test seti üzerinde doğruluk ve sınıflandırma raporu üretir
- Toplam eğitim süresini hesaplar ve istatistiksel özetleri verir

---

## Kullanım

```bash
python LIF_MODEL.py
```

Veri dosyasının `mnist.csv` adında ve aynı dizinde olduğundan emin olun.

---

## Özellikler

- LIF modeline uygun zaman temelli spike üretimi
- STDP kurallarına dayalı öğrenme
- Boltzmann sıcaklık parametresiyle olasılıksal çıktı tahmini
- Zamanla azalan sıcaklık (annealing)
- Softmax tabanlı sınıflandırma çıktısı

---

## Gereksinimler

- Python 3.7+
- Gerekli paketler: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`

---

## Kaynaklar

- Dayanıklı sinirsel hesaplama için Spiking Neural Networks literatürü
- MNIST veri seti (CSV formatında kullanılmalıdır)

---

## Not

Bu proje araştırma ve eğitim amaçlıdır. Gerçek zamanlı donanım simülasyonları için ek uyarlamalar gerekebilir.
