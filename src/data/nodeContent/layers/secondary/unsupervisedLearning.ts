import { NodeContent } from '../../../../types/content';

export const unsupervisedLearningContent: NodeContent = {
  title: 'Unsupervised Learning',
  description: 'Machine learning techniques that find hidden patterns or structures in unlabeled data without pre-existing outputs to train on.',
  concepts: [
    'Clustering',
    'Dimensionality Reduction',
    'Anomaly Detection',
    'Pattern Recognition',
    'Feature Learning'
  ],
  examples: [
    {
      language: 'python',
      description: 'K-means clustering example',
      code: `from sklearn.cluster import KMeans
import numpy as np

# Create and fit model
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Analyze cluster centers
centers = kmeans.cluster_centers_

# Get cluster assignments
labels = kmeans.labels_`
    }
  ],
  resources: [
    {
      title: 'Unsupervised Learning Guide',
      description: 'Comprehensive guide to unsupervised learning',
      url: 'https://www.deeplearning.ai/courses/unsupervised-learning/'
    },
    {
      title: 'Clustering Algorithms',
      description: 'Overview of different clustering techniques',
      url: 'https://scikit-learn.org/stable/modules/clustering.html'
    }
  ],
  prerequisites: ['Machine Learning Basics', 'Linear Algebra', 'Statistics'],
  relatedTopics: ['Clustering', 'Dimensionality Reduction', 'Pattern Recognition']
};