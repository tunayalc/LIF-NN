# LIF-NN

Leaky Integrate-and-Fire tabanlı deneysel sinir ağı çalışması

## Genel Bakış

`LIF-NN`, klasik derin öğrenme akışının dışında, biyolojik esinli hesaplama yaklaşımına yöneldiğim araştırma odaklı bir çalışma. Repo, spiking neural network mantığına yaklaşan bir modelleme denemesini tek dosyalı ama yoğun bir prototip olarak içeriyor.

## Çalışmanın Odağı

Bu proje içinde özellikle şu konular üzerine çalışıldı:

- Leaky Integrate-and-Fire nöron davranışı
- zamansal aktivasyon mantığı
- adaptif eşik yaklaşımı
- STDP benzeri öğrenme fikri
- tekrarlayan bağlantılar
- deneysel sınıflandırma senaryosu

## Kodun Merkezindeki Yapı

Repo büyük ölçüde `LIF_MODEL.py` dosyasında toplanıyor. Bu dosya:

- model tanımını
- veri işleme akışını
- eğitim mantığını
- değerlendirme ve görselleştirme adımlarını

aynı araştırma dosyası içinde bir araya getiriyor.

Bu tarz yapı, üretim sistemi kurgusundan çok araştırma ve deney prototipi mantığına daha yakın.

## Repo Yapısı

```text
LIF-NN/
|-- LIF_MODEL.py
|-- requirements.txt
`-- README.md
```

## Kullanılan Teknolojiler

- Python
- NumPy
- Matplotlib
- deneysel sinir ağı modelleme yaklaşımı
