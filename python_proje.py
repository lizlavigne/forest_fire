

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px

data = pd.read_csv("./forestfires.csv")

data.head()



data.isnull()

df= pd.DataFrame(data)
df

area_filter= df.groupby(["X","Y"])[["area"]].agg(lambda x: x.mean()).reset_index()

plt.figure(figsize=(15,5))
sns.barplot(x="X", y="area", data=area_filter)
plt.xlabel("X'in sayısı")
plt.ylabel("XY alanlarının toplamı")
plt.title("XY alanları")
plt.show()



df.groupby(["X","Y"])[["area","FFMC","ISI"]].mean()

plt.figure(figsize=(15,5))
sns.countplot(x='month', data=df, order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
plt.title('Aylara Göre Yangın Sayısı')
plt.show()





plt.figure(figsize=(15, 5))
sns.scatterplot(x='temp', y='area', data=df)
plt.xlabel("Sıcaklık (°C)")
plt.ylabel("Yanan Alan")
plt.title("Sıcaklık ve Yanan Alan Arasındaki İlişki")
plt.show()



plt.figure(figsize=(12, 10))


numeric_df = df.select_dtypes(include=['float64', 'int64'])


correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Sayısal Değişkenler Arası Korelasyon Matrisi")
plt.show()







plt.figure(figsize=(8,6))
sns.countplot(x='day', data=df, order=['mon','tue','wed','thu','fri','sat','sun'])
plt.title("Haftanın Günlerine Göre Yangın Sayısı")
plt.xlabel("Gün")
plt.ylabel("Yangın Sayısı")
plt.show()




plt.figure(figsize=(8, 6))
sns.scatterplot(x='temp', y=np.log1p(df['area']), data=df)
plt.title("Sıcaklık ve Yangın Alanı (Log Ölçekli)")
plt.xlabel("Sıcaklık (°C)")
plt.ylabel("Yangın Alanı (log(1 + ha))")
plt.grid(True)
plt.show()




df['yangın_var'] = df['area'].apply(lambda x: 1 if x > 0 else 0)
df = df.drop(columns=['area'])
print(df.head())
print(df['yangın_var'].value_counts())



X = df[['temp', 'RH', 'wind', 'rain']]
y = df['yangın_var']


X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Eğitim seti örnek sayısı: {X_egitim.shape[0]}")
print(f"Test seti örnek sayısı: {X_test.shape[0]}")


model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_egitim, y_egitim)


y_tahmin = model.predict(X_test)

print("Model Doğruluk Skoru:", accuracy_score(y_test, y_tahmin))
print("\nDetaylı Performans Raporu:\n", classification_report(y_test, y_tahmin))


with open("orman_yangini_model_pickle.pkl", "wb") as file:
    pickle.dump(model, file)



