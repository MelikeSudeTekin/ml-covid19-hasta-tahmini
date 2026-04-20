# 🦠 COVID-19 Yoğun Bakım (ICU) İhtiyacı Tahmin Modeli

Bu proje, Kaggle üzerinden sağlanan kapsamlı bir COVID-19 veri setini kullanarak, hastaların klinik bulgularına ve demografik özelliklerine göre **Yoğun Bakım Ünitesine (ICU)** alınıp alınmayacağını tahmin eden uçtan uca bir makine öğrenmesi boru hattıdır (pipeline).

## 📊 Proje Özeti
Proje kapsamında veri ön işleme, keşifsel veri analizi (EDA), özellik ölçeklendirme (Feature Scaling) ve modelleme adımları uygulanmıştır. Büyük veri seti optimize edilerek model eğitim süreleri iyileştirilmiş ve üç farklı sınıflandırma algoritmasının (Random Forest, Logistic Regression, K-Nearest Neighbors) performansları karşılaştırılmıştır.

## 🛠️ Kullanılan Teknolojiler
* **Programlama Dili:** Python 3
* **Veri Manipülasyonu & Analizi:** Pandas, NumPy
* **Makine Öğrenmesi Kütüphanesi:** Scikit-Learn
* **Veri Görselleştirme:** Matplotlib, Seaborn

## 🧹 Veri Ön İşleme (Data Preprocessing) Adımları
Veri seti üzerinde modellerin sağlıklı çalışabilmesi için aşağıdaki adımlar izlenmiştir:
1. **Hedef Değişken Filtrelemesi:** `ICU` (Yoğun Bakım) sütununda bulunan ve "eksik veri" anlamına gelen 97, 98, 99 gibi değerler temizlenmiş; yalnızca `1` (Evet) ve `2` (Hayır) etiketleri bırakılmıştır.
2. **Tip Dönüşümü & Temizlik:** Algoritmaların matematiksel hesaplamaları yapabilmesi için tarih ve metin formatındaki (`object` tipli) sütunlar veri setinden izole edilmiştir.
3. **Eksik Veri Doldurma:** Kalan nümerik eksik veriler, ilgili sütunun ortalama değeri (`mean`) ile doldurulmuştur.
4. **Örneklem Alma (Sampling):** İşlemci maliyetini düşürmek ve KNN gibi uzaklık tabanlı algoritmaları optimize etmek adına veri setinden rastgele 50.000 satırlık dengeli bir örneklem alınmıştır.
5. **Özellik Ölçeklendirme:** Uzaklık tabanlı ve gradient tabanlı modeller için veriler `StandardScaler` ile standartlaştırılmıştır.

## 🤖 Makine Öğrenmesi Modelleri ve Performans Değerlendirmesi
Veri %80 Eğitim (Train) ve %20 Test olarak ayrılmış ve üç farklı model ile eğitilmiştir. 10.000 test verisi üzerinden elde edilen kesin sonuçlar ve Karmaşıklık Matrisi (Confusion Matrix) verileri şu şekildedir:

### 1. K-Nearest Neighbors (KNN) - 🏆 En Başarılı Model
* **Doğruluk Oranı (Accuracy):** `%91.11`
* **Karmaşıklık Matrisi (Test Verisi: 10.000):**
  * Doğru Pozitif (Yoğun bakıma girmeyenleri bilme): 8851
  * Doğru Negatif (Yoğun bakıma girenleri bilme): 260
  * Yanlış Pozitif: 697
  * Yanlış Negatif: 192
* *Yorum:* Doğruluk oranı en yüksek model olmasına rağmen, verideki sınıf dengesizliğinden dolayı azınlık sınıfı olan "1" (Yoğun bakım evet) durumunu yakalamada (%27 Recall) zorlanmıştır.

### 2. Logistic Regression
* **Doğruluk Oranı (Accuracy):** `%90.71`
* **Karmaşıklık Matrisi:**
  * Doğru Pozitif: 8948
  * Doğru Negatif: 123
  * Yanlış Pozitif: 834
  * Yanlış Negatif: 95
* *Yorum:* Sınıf 2'yi tahmin etmede (%99 Recall) kusursuza yakın çalışmış, ancak azınlık sınıfını en çok kaçıran model olmuştur.

### 3. Random Forest Classifier
* **Doğruluk Oranı (Accuracy):** `%90.26`
* **Karmaşıklık Matrisi:**
  * Doğru Pozitif: 8749
  * Doğru Negatif: 277
  * Yanlış Pozitif: 680
  * Yanlış Negatif: 294
* *Yorum:* Genel doğruluk oranı en düşük model olmasına karşın, yoğun bakıma girecek (Sınıf 1) hastaları doğru tespit etme (Recall: %29) konusunda diğer iki modelden daha dengeli ve agresif bir tutum sergilemiştir.

## 📌 Sonuç ve Gelecek Çalışmalar
Tüm modeller **%90'ın üzerinde genel doğruluk** oranına ulaşarak yüksek bir öngörü başarısı yakalamıştır. Ancak modeller "Yoğun Bakıma Girmeyecek" olan çoğunluk sınıfını öğrenmeye daha yatkınlık göstermiştir. 

**Gelecekteki Geliştirmeler:**
Gelecek aşamalarda bu dengesizliği (imbalanced dataset) çözmek adına azınlık sınıfına **SMOTE (Synthetic Minority Over-sampling Technique)** uygulanarak veya modellerin sınıf ağırlıkları (`class_weight='balanced'`) dengelenerek azınlık sınıfındaki `Recall` değerlerinin artırılması hedeflenmektedir.

## 🚀 Kurulum ve Çalıştırma
Projeyi kendi bilgisayarınızda çalıştırmak için:
1. Repoyu klonlayın.
2. Kaggle'dan ilgili veri setini indirip `covid_veri_setin.csv` olarak ana dizine ekleyin.
3. Gerekli kütüphaneleri yükleyin: `pip install pandas numpy scikit-learn matplotlib seaborn`
4. Kod dosyasını çalıştırın: `python main.py`

---
