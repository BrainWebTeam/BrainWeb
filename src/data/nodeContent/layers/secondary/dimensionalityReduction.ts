import { NodeContent } from '../../../../types/content';

export const dimensionalityReductionContent: NodeContent = {
  title: 'Dimensionality Reduction',
  description: 'Techniques to reduce the number of features in high-dimensional data while preserving important patterns and relationships.',
  concepts: [
    'Principal Component Analysis (PCA)',
    't-SNE',
    'UMAP',
    'Feature Selection',
    'Manifold Learning'
  ],
  examples: [
    {
      language: 'python',
      description: 'PCA implementation',
      code: `from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data_scaled)

# Explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)`
    }
  ],
  resources: [
    {
      title: 'Dimensionality Reduction Guide',
      description: 'Overview of reduction techniques',
      url: 'https://scikit-learn.org/stable/modules/manifold.html'
    },
    {
      title: 'How to Use t-SNE Effectively',
      description: 'Visual guide to t-SNE',
      url: 'https://distill.pub/2016/misread-tsne/'
    }
  ],
  prerequisites: ['Linear Algebra', 'Statistics', 'Machine Learning Basics'],
  relatedTopics: ['Feature Selection', 'Data Visualization', 'Manifold Learning']
};