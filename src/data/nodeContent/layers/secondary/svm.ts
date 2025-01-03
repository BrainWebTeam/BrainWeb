import { NodeContent } from '../../../../types/content';

export const svmContent: NodeContent = {
  title: 'Support Vector Machines',
  description: 'A powerful supervised learning algorithm that finds the optimal hyperplane to separate classes in high-dimensional space.',
  concepts: [
    'Kernel Functions',
    'Margin Maximization',
    'Support Vectors',
    'Kernel Trick',
    'Soft Margin Classification'
  ],
  examples: [
    {
      language: 'python',
      description: 'SVM classifier with different kernels',
      code: `from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM with different kernels
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_scaled, y)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_scaled, y)

# Make predictions
predictions = svm_rbf.predict(X_test_scaled)`
    }
  ],
  resources: [
    {
      title: 'SVM Tutorial',
      description: 'Comprehensive guide to SVMs',
      url: 'https://scikit-learn.org/stable/modules/svm.html'
    },
    {
      title: 'SVM Visualization',
      description: 'Interactive SVM visualization',
      url: 'https://www.cs.princeton.edu/~karthik/teaching/svm-visualization/'
    }
  ],
  prerequisites: ['Linear Algebra', 'Optimization Theory', 'Machine Learning Basics'],
  relatedTopics: ['Kernel Methods', 'Classification', 'Feature Space']
};