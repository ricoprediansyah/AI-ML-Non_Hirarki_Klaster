# ML-6 Non-Hierarchical Clustering
# Import Library

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  #standarisasi feature
from sklearn.cluster import KMeans                #algoritma kmeans
from sklearn.metrics import silhouette_score

# Menginstal library machine learning visualization: yellowbrick.
# !pip install -U yellowbrick
# Import library KElbowVisualizer.
from yellowbrick.cluster import KElbowVisualizer #metode elbow

# Load Dataset
# Download dataset 
# Pada hands-on ini, kita masih menggunakan dataset Mall_Customers.csv.
# Berbeda dengan materi sebelumnya, kali ini, kita akan lakukan Customer Segmentation menggunakan algoritma K-Means.
data = pd.read_csv('Mall_Customers (1).csv',index_col='CustomerID')
# Memeriksa kelengkapan dataset.
data.info()
# Kita hanya memilih dua variable saja, yakni AnnualIncome dan SpendingScore.
# Note: Jika kita memilih variable Gender, maka kita perlu melakukan preprocessing terlebih dahulu.
X = data[['AnnualIncome', 'SpendingScore']].values
# Metode Elbow
# Sebelum men-training model K-Means, sebaiknya kita mencari nilai K yang paling baik/optimal terlebih dahulu.
# Nilai ini bisa kita cari menggunakan Metode Elbow atau KElbowVisualizer pada Python.
tes_model = KMeans(random_state=42)
visualizer = KElbowVisualizer(tes_model, k=(2,10))
visualizer.fit(X)
visualizer.show()
# Berdasarkan hasil Metode Elbow, Titik Elbow berada pada K=5.
# Maka nilai K terbaik/optimal adalah 5.

# Selanjutnya kita lakukan training model menggunakan algoritma K-Means dengan:
# cluster bejumlah 5 (K=5)
# random state bernilai 42
# Random state berguna untuk mengontrol ke-random-an inisiasi centroid.

# Hyperparameter tuning
jumlah_cluster = 5
randomizer = 42
model_kmeans = KMeans(n_clusters=jumlah_cluster, random_state=randomizer)

#Training model
model_kmeans.fit(X)

# hasil clustering
labels_kmeans = model_kmeans.labels_
labels_kmeans

# Karena kita menentukan K=5 (cluster berjumlah 5), maka kita akan mendapat 5 centroid.
# Informasi (letak) tiap centroid bisa kita akses menggunakan method cluster_centers_.
# Ada beberapa metode yang bisa kita gunakan untuk mengevaluasi model clustering, antara lain:

# Melihat dendrogram
# Metode elbow
# Visualisasi data
# Silhouette Coefficient (SC)
# Dendrogram digunakan pada model AHC. Sedangkan, metode elbow sudah kita gunakan untuk menentukan nilai K terbaik.

# Selanjutnya, kita akan mengevaluasi model clustering menggunakan visualisasi data dan SC.
# Visualisasi Hasil Clustering (Visualisasi Data)
# Kita bisa memvisualisasikan hasil clustering dan centroid secara bersamaan menggunaan scatter plot.
# Visualisasi hasil clustering
plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='rainbow')
# visualisasi centroid
plt.scatter(model_kmeans.cluster_centers_[:, 0], model_kmeans.cluster_centers_[:,1], color='black')
plt.show()
# Terlihat hasil clustering sudah cukup bagus.

# Note:
# Kita bisa memvisualisasikan dataset secara 2D karena dataset tersebut hanya terdiri dari dua variable/feature, yakni AnnualIncome dan SpendingScore.
# Jika dataset kita terdiri dari tiga variable, maka kita masih bisa memvisualisasikan dataset secara 3D.
# Namun, kita tidak akan bisa memvisualisasikan dataset dengan jumlah variable lebih dari tiga, karena visualisasinya lebih dari 3D.
# Salah satu solusi visualisasi dataset dengan jumlah variable >3 adalah reduksi dimensi (dimensionality reduction).
# Materi dimensionality reduction akan kita pelajari pada domain Data Science.

# Kita bisa menggunakan silhouette_score untuk mendapat Silhouette Score model yang telah kita training.
SC_kmeans = silhouette_score(X, labels_kmeans, metric='euclidean')
print('Silhouette Score model K-means=', SC_kmeans)

# Semakin Silhouette Score mendekati 1, maka model clustering akan semakin bagus.
# Berdasarkan Silhouette Score, model K-Means yang telah kita training sudah cukup bagus.
# Alangkah baiknya jika kita men-training lebih dari satu model clustering dengan berbagai algoritma dan tuning hyperparameter.
# Kemudian, kita bandingkan Silhouette Score tiap model untuk mengetahui mana model terbagus.

# Simpan hasil clustering ke dalam dataset.
data['Hasil_Clustering'] = labels_kmeans
data

# Contoh:
# Menghitung rata-rata tiap cluster untuk menarik kesimpulan.
for i in range(jumlah_cluster):
  print(f'Cluster ke-{i}')
  print('Rata-rata pemasukan customer   : ', data[data['Hasil_Clustering']==i]['AnnualIncome'].mean())
  print('Rata-rata pengeluaran customer : ', data[data['Hasil_Clustering']==i]['SpendingScore'].mean())
  print()

#   Bisa kita simpulkan bahwa :

# Cluster 0 adalah kelompok customer menengah ke bawah yang hemat (pengeluarannya sangat kecil)
# Cluster 1 adalah kelompok customer menengah ke bawah yang sangat boros (pengeluaran lebih besar dari pemasukan)
# Cluster 2 adalah kelompok customer tingkat atas yang hemat (pengeluarannya sangat kecil)
# Cluster 3 adalah kelompok customer menengah ke atas yang sangat boros (pengeluaran lebih besar dari pemasukan)
# Cluster 4 adalah kelompok customer menengah ke bawah yang boros (pengeluaran hampir sama dengan pemasukan)