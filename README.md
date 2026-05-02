#  COVID-19 ICU (Yoğun Bakım) Tahmin Sistemi

Bu projede, COVID-19 hastalarının klinik ve demografik verilerini kullanarak **yoğun bakım ihtiyacını önceden tahmin eden** bir makine öğrenmesi sistemi geliştirdim.

##  Problem

Pandemi sürecinde hastaneler için en kritik konulardan biri, hangi hastaların yoğun bakıma ihtiyaç duyacağını önceden tahmin edebilmektir.

##  Çözüm

Bu projede, hasta verilerini analiz ederek ICU (Yoğun Bakım) ihtiyacını tahmin eden bir ML pipeline oluşturuldu.

* Veri temizleme
* Özellik mühendisliği
* Model eğitimi
* Performans karşılaştırma

##  Sonuçlar

| Model               | Accuracy |
| ------------------- | -------- |
| KNN                 | %91.11   |
| Logistic Regression | %90.71   |
| Random Forest       | %90.26   |

 **Önemli içgörü:**
Veri dengesiz olduğu için modeller yüksek doğruluk verse de, yoğun bakıma girecek hastaları (kritik sınıf) tahmin etmekte zorlanmaktadır.

##  Öğrenilenler

* Accuracy tek başına yeterli bir metrik değildir
* Imbalanced dataset ciddi bir problemdir
* Recall metriği sağlık projelerinde kritik öneme sahiptir

##  Teknolojiler

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

##  Nasıl Çalıştırılır

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python main.py
```

##  Örnek Çıktı

(Buraya confusion matrix veya grafik ekle)

##  Geliştirme Fikirleri

* SMOTE ile veri dengeleme
* Flask ile web arayüz ekleme
* Modeli API olarak sunma

##  Proje Amacı

Bu proje, gerçek dünya problemlerine makine öğrenmesi uygulayarak hem teknik hem analitik becerilerimi geliştirmek amacıyla yapılmıştır.

