import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Fake dataset: study hours vs exam scores
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Score": [20,25,35,40,50,60,65,78,85,95]
}
df = pd.DataFrame(data)

# Features
X = df[['Hours', 'Score']]

# Apply KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

print(df)

# Plot clusters
plt.scatter(df['Hours'], df['Score'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            color='red', marker='X', s=200, label='Centroids')
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Student Clustering (K-Means)")
plt.legend()
plt.show()
