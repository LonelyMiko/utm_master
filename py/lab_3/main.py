import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsne

from py.scripts.plot import plot2D_PX

RESOURCES_PATH = "py/resources/"
df_calls = pd.read_csv(RESOURCES_PATH + "calls.csv")
df_contract = pd.read_csv(RESOURCES_PATH + "contract.csv")

df_joined = pd.merge(df_calls, df_contract, how='inner', on='Phone')

types = ['int64', 'float64']
df_filtered = df_joined.select_dtypes(include=types)
cols = df_filtered.columns
min_max_scale = MinMaxScaler()
min_max_scale.fit(df_filtered)
df_normalized = min_max_scale.transform(df_filtered)
df_normalized = pd.DataFrame(data=df_normalized, columns=cols)
df_normalized_k_means = np.array(
    [(df_normalized.values[i, 10], df_normalized.values[i, 11]) for i in range(0, len(df_normalized.values[:, 0]))])

kmeans = KMeans(n_clusters=51)
kmeans.fit(df_normalized_k_means)
y_km = kmeans.predict(df_normalized_k_means)
plt.scatter(df_normalized_k_means[:, 0], df_normalized_k_means[:, 1], c=y_km, s=50, cmap='nipy_spectral')
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=60, alpha=0.5)
plt.show()

df_normalized_k_means = np.array(
    [(df_normalized.values[i, 6], df_normalized.values[i, 10], df_normalized.values[i, 12]) for i in
     range(0, len(df_normalized.values[:, 0]))])

pca = PCA(n_components=1)
df_joined_reduced = pca.fit(df_normalized_k_means, ["DailyCalls", "NightCalls", "IntCalls"]).transform(
    df_normalized_k_means)
df_joined_reduced = np.array(
    [(df_joined_reduced[i, 0], df_normalized.values[i, 11]) for i in range(0, len(df_normalized.values[:, 0]))])

min_max_scale = MinMaxScaler()
min_max_scale.fit(df_joined_reduced)
df_joined_reduced = min_max_scale.transform(df_joined_reduced)

inertias = []
for i in range(1, 51):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_joined_reduced)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 51), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=51)
kmeans.fit(df_joined_reduced)
y_km = kmeans.predict(df_joined_reduced)
plot2D_PX(df_joined_reduced, y_km, 'KMeans N_Clusters = 51').show()

cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=60, alpha=0.5)
plt.show()
calls_numbers_format = df_calls.select_dtypes(include=types)
min_max_scale.fit(calls_numbers_format)
calls_numbers_format_normailzed = min_max_scale.transform(calls_numbers_format)

calls_numbers_format_tsne = tsne(n_components=2).fit_transform(calls_numbers_format_normailzed)

min_max_scale.fit(calls_numbers_format_tsne)
calls_numbers_format_tsne = min_max_scale.transform(calls_numbers_format_tsne)

kmeans = KMeans(n_clusters=51)
kmeans.fit(calls_numbers_format_tsne)
y_km = kmeans.predict(calls_numbers_format_tsne)
cluster_centers = kmeans.cluster_centers_

plot2D_PX(calls_numbers_format_tsne, y_km, 'TSNE')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=60, alpha=0.5)
plt.show()

contract_numbers_format = df_contract.select_dtypes(include=types)
min_max_scale.fit(contract_numbers_format)
contract_numbers_format_normalized = min_max_scale.transform(contract_numbers_format)
contract_numbers_format_tsne = tsne(n_components=2).fit_transform(contract_numbers_format_normalized)
min_max_scale.fit(contract_numbers_format_tsne)
contract_numbers_format_tsne = min_max_scale.transform(contract_numbers_format_tsne)
kmeans.fit(contract_numbers_format_tsne)
y_km = kmeans.predict(contract_numbers_format_tsne)

plot2D_PX(contract_numbers_format_tsne, y_km, 'Contract TSNE').show()
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=60, alpha=0.5)
plt.show()

df_joined_numeric_format = df_contract.select_dtypes(include=types)
min_max_scale.fit(df_joined_numeric_format)
df_joined_numeric_format_normalized = min_max_scale.transform(df_joined_numeric_format)
df_joined_numeric_format_tsne = tsne(n_components=2).fit_transform(df_joined_numeric_format_normalized)

min_max_scale.fit(df_joined_numeric_format_tsne)
df_joined_numeric_format_tsne = min_max_scale.transform(df_joined_numeric_format_tsne)

kmeans.fit(df_joined_numeric_format_tsne)
y_km = kmeans.predict(df_joined_numeric_format_tsne)
plot2D_PX(df_joined_numeric_format_tsne, y_km, 'Joined TSNE').show()
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=60, alpha=0.5)
plt.show()
