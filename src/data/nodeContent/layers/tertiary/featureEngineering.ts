import { NodeContent } from '../../../../types/content';

export const featureEngineeringContent: NodeContent = {
  title: 'Feature Engineering',
  description: 'The process of using domain knowledge to extract and create meaningful features from raw data to improve machine learning model performance.',
  concepts: [
    'Feature Selection',
    'Feature Transformation',
    'Feature Scaling',
    'Encoding Techniques',
    'Dimensionality Reduction'
  ],
  examples: [
    {
      language: 'python',
      description: 'Feature engineering techniques',
      code: `import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y)

# Feature creation
df['price_to_sqft'] = df['price'] / df['sqft']
df['age'] = df['year_sold'] - df['year_built']`
    }
  ],
  resources: [
    {
      title: 'Feature Engineering Book',
      description: 'Practical guide to feature engineering',
      url: 'https://www.featureengineering.com/'
    },
    {
      title: 'Kaggle Feature Engineering',
      description: 'Real-world feature engineering examples',
      url: 'https://www.kaggle.com/learn/feature-engineering'
    }
  ],
  prerequisites: ['Statistics', 'Data Analysis', 'Domain Knowledge'],
  relatedTopics: ['Feature Selection', 'Data Preprocessing', 'Model Performance']
};