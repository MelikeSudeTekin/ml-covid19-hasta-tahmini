import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("1. Veri seti yükleniyor...")
df = pd.read_csv('covid19_data.csv')

print("2. Veri temizleniyor ve kalibre ediliyor...")
# Hedef değişkeni sadece 1 ve 2 olan (Geçerli) kayıtları alıyoruz
df = df[(df['ICU'] == 1) | (df['ICU'] == 2)]

# Metin ve tarih formatındaki sütunları temizliyoruz!
# Böylece model '30/05/2020' gibi değerlere takılmayacak.
df = df.select_dtypes(exclude=['object'])

# Eksik verileri doldurma
df.fillna(df.mean(numeric_only=True), inplace=True)

# İşlemlerin hızlıca bitmesi için rastgele 50.000 satır alıyoruz.
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)
    print("-> İşlem hızlandırması için veriden 50.000 satırlık örneklem alındı.")

# Bağımlı ve Bağımsız Değişkenleri Ayırma
X = df.drop('ICU', axis=1)
y = df['ICU']

# Eğitim ve Test Olarak Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellik Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model Değerlendirme Modülü
def model_degerlendir(model, model_adi, X_tr, X_te):
    print(f"\n>>> {model_adi} eğitiliyor... Lütfen bekleyin.")
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    print(f"\n{'=' * 40}")
    print(f"Model: {model_adi}")
    print(f"Doğruluk Oranı (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Karmaşıklık Matrisi (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_adi} - Confusion Matrix')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen Değer')
    plt.show()


# --- MODELLERİN ÇALIŞTIRILMASI ---
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_degerlendir(rf_model, "Random Forest", X_train, X_test)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
model_degerlendir(lr_model, "Logistic Regression", X_train_scaled, X_test_scaled)

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
model_degerlendir(knn_model, "K-Nearest Neighbors (KNN)", X_train_scaled, X_test_scaled)

print("\nTüm işlemler başarıyla tamamlandı!")