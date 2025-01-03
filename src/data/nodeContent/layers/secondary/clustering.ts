import { NodeContent } from '../../../../types/content';

export const clusteringContent: NodeContent = {
  title: 'Clustering',
  description: 'Unsupervised learning techniques that group similar data points together based on their features and characteristics.',
  concepts: [
    'K-means Clustering',
    'Hierarchical Clustering',
    'DBSCAN',
    'Cluster Validation',
    'Distance Metrics'
  ],
  examples: [
    {
      language: 'python',
      description: 'K-means clustering implementation',
      code: `from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create and fit model
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(data)

# Evaluate clustering
silhouette_avg = silhouette_score(data, clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Get cluster centers
centers = kmeans.cluster_centers_`
    }
  ],
  resources: [
    {
      title: 'Clustering Guide',
      description: 'Comprehensive guide to clustering algorithms',
      url: 'https://scikit-learn.org/stable/modules/clustering.html'
    },
    {
      title: 'Clustering Visualization',
      description: 'Interactive clustering visualization',
      url: 'https://www.naftaliharris.com/blog/visualizing-k-means-clustering/'
    }
  ],
  prerequisites: ['Statistics', 'Linear Algebra', 'Unsupervised Learning'],
  relatedTopics: ['Dimensionality Reduction', 'Pattern Recognition', 'Data Mining']
};