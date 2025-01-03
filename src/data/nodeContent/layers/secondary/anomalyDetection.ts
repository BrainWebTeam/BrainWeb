import { NodeContent } from '../../../../types/content';

export const anomalyDetectionContent: NodeContent = {
  title: 'Anomaly Detection',
  description: 'Techniques for identifying rare items, events or observations that deviate significantly from the majority of the data and raise suspicions.',
  concepts: [
    'Statistical Methods',
    'Isolation Forest',
    'One-Class SVM',
    'Autoencoders for Anomaly Detection',
    'Time Series Anomalies'
  ],
  examples: [
    {
      language: 'python',
      description: 'Isolation Forest implementation',
      code: `from sklearn.ensemble import IsolationForest

# Create and train model
iso_forest = IsolationForest(
    contamination=0.1,
    random_state=42
)

# Fit and predict
predictions = iso_forest.fit_predict(data)

# Anomalies are labeled as -1
anomalies = data[predictions == -1]`
    }
  ],
  resources: [
    {
      title: 'Anomaly Detection Guide',
      description: 'Comprehensive guide to anomaly detection',
      url: 'https://scikit-learn.org/stable/modules/outlier_detection.html'
    },
    {
      title: 'Time Series Anomaly Detection',
      description: 'Microsoft\'s guide to time series anomaly detection',
      url: 'https://github.com/microsoft/anomalydetector'
    }
  ],
  prerequisites: ['Statistics', 'Machine Learning', 'Data Analysis'],
  relatedTopics: ['Outlier Detection', 'Fraud Detection', 'Monitoring']
};