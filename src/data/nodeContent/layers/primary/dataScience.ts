import { NodeContent } from '../../../../types/content';

export const dataScienceContent: NodeContent = {
  title: 'Data Science',
  description: 'The interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.',
  concepts: [
    'Data Collection and Cleaning',
    'Exploratory Data Analysis',
    'Statistical Analysis',
    'Data Visualization',
    'Feature Engineering'
  ],
  examples: [
    {
      language: 'python',
      description: 'Data analysis with pandas',
      code: `import pandas as pd
import seaborn as sns

# Load and analyze data
df = pd.read_csv('data.csv')
df.describe()

# Create visualization
sns.boxplot(x='category', y='value', data=df)
plt.title('Distribution by Category')

# Feature engineering
df['new_feature'] = df['a'] / df['b']
df['log_feature'] = np.log(df['value'])`
    }
  ],
  resources: [
    {
      title: 'Data Science Handbook',
      description: 'Comprehensive guide to data science',
      url: 'https://jakevdp.github.io/PythonDataScienceHandbook/'
    },
    {
      title: 'Kaggle Courses',
      description: 'Free data science courses',
      url: 'https://www.kaggle.com/learn'
    }
  ],
  prerequisites: ['Statistics', 'Programming', 'Mathematics'],
  relatedTopics: ['Machine Learning', 'Big Data', 'Data Engineering']
};