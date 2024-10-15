import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

data = pd.read_csv('eth_cleaned.csv', encoding='utf-8')
data = shuffle(data)
data['flag'] = data['flag'].astype(str)

exchange_data = data[data['flag'] == 'exchange'].sample(n=11180, random_state=42)
non_exchange_data = data[data['flag'] != 'exchange']
data = pd.concat([exchange_data, non_exchange_data])

data.fillna(0, inplace=True)

cluster_map = {0: 'crime', 1: 'exchange', 2: 'normal'}
label_encoder = LabelEncoder()
data['flag'] = data['flag'].map({'crime': 0, 'exchange': 1, 'normal': 2})

X = data.drop(columns=['address', 'flag'])
y = data['flag']

scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(scaled_X)

kmeans = KMeans(n_clusters=3, random_state=42)

for _ in tqdm(range(1), desc="Clustering", colour='green'):
    clusters = kmeans.fit_predict(scaled_X)

mapped_clusters = [cluster_map[c] for c in clusters]

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=mapped_clusters, palette="deep", legend="full")
plt.title('ETH Onchain KMeans Clustering')
plt.legend(title="Flag")
plt.show()
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
tsne_3d = TSNE(n_components=3, random_state=42)
X_tsne_3d = tsne_3d.fit_transform(scaled_X)

scatter = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=clusters, cmap='plasma')

legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.to_rgba(i), markersize=10) for i in range(3)]
ax.legend(legend_labels, ['crime', 'exchange', 'normal'], title="Flag")
ax.set_title('ETH Onchain KMeans Clustering')
plt.show()