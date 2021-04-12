import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv("student_scores.csv")

# verisetinin boyutları
dataShape = dataset.shape
print(dataShape)

print(dataset.head(7)) # datasetin başını gösterir. default değer 5'tir.
print(dataset.tail(7)) # datasetin sonunu gösterir. default değer 5'tir.

# Veriseti hakkında ön bilgi için describe() kullanılabilir.
print(dataset.describe())
print(dataset.columns) # verisetindeki kolonların isimlerini verir.


X = dataset.iloc[:, :-1].values #Hours değerleri
y = dataset.iloc[:, 1].values #Scores değerleri

print(X)
print(y)

# Verisetini eğitim ve test olarak parçalar
# test_size=0.2 => verisetinin %80'i eğitim %20'si test için ayrılmasını sağlar
# random_state => fonksiyon her çalışmasında veriyi farklı bir sırayla çekmesinin önüne geçer
#   Böylece eğitim her zaman aynı sırayla yapılmış olur.    
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

# Regresyon modelinin eklenmesi
from sklearn.linear_model import LinearRegression
reg = LinearRegression() # regresyon nesnesi tanımalama

reg.fit(Xtrain,ytrain) # hazırlanan veriler modele verilir ve eğitime başlanır.

print(reg.intercept_) # Çıkan sonucun yaklaşık 2.01816004143 değerde olması gerekiyor.

# x değişkenini bir birim artmasıyla y'de olan değişiklik
print(reg.coef_) # bu örnek için öğrenci bir saat fazla çalışarak çıkan sonuç kadar yüksek skor elde edebilir.

### Tahmin Yapma
yPred = reg.predict(Xtest) ## Xtest değerlerine göre yapılan tahminler
print(yPred)
# yapılan tahminlerin gerçek verilerle kıyaslanabilmesi için dataframe haline getirilir.
df = pd.DataFrame({"Gerçek Değer": ytest, "Tahmin Edilen değer": yPred})
print(df)

## Performans Değerlendirmesi
from sklearn import metrics

# Mean Absolute Error (MAE) (Ortalama Mutlak Hata) metriğine göre puanı
# Mutlak hata, tahmin edilen değerler ile gerçek değerler arasındaki farktır.
# gerçek değer ile tahmin arasındaki farkın mutlak değeridir.
maeScore = metrics.mean_absolute_error(ytest, yPred)
print("Ortalama Mutlak Hata = " + str(maeScore))

# Mean Squared Error (MSE) (Ortalama Kare Hatası) metriğine göre sonucu
# MSE her değer için gerçek değer ile tahmin arasındaki farkın kareleri toplamının 
# aritmetik ortalamasıdır.
mseScore = metrics.mean_squared_error(ytest, yPred)
print("Ortalama Kare Hatası = " + str(mseScore))

# Root Mean Squared Error (RMSE) metriğine göre sonucu
rmseScore = np.sqrt(metrics.mean_squared_error(ytest, yPred))
print("RMSE = "+ str(rmseScore))

# Veri görselleştirme
dataset.plot(x="Hours", y="Scores", style="go")
random_x = [1.1, 5.01, 9.2]
plt.plot(random_x,
         reg.intercept_ + reg.coef_ * random_x,
         color='red',
         label='regresyon grafiği')
plt.title("Saatlere Göre Yüzdelik Skorlar")
plt.xlabel("Çalışma Saatleri")
plt.ylabel("Yüzdelik Skorlar")
plt.savefig("Grafik.jpg")
plt.show()


## Veri seti dışındaki veriler ile yapılan tahminler

testVeri = np.array([0.5, 1.0, 4.2, 6.7, 10.0]).reshape(-1,1)
pred = reg.predict(testVeri)

for i in range(len(testVeri)):
    print(str(testVeri[i]) + "=>" + str(pred[i]) )