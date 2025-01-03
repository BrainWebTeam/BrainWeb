import { NodeContent } from '../../../../types/content';

export const supervisedLearningContent: NodeContent = {
  title: 'Supervised Learning',
  description: 'A machine learning approach where models learn from labeled training data to make predictions or classifications on new, unseen data.',
  concepts: [
    'Classification',
    'Regression',
    'Training and Testing',
    'Model Evaluation',
    'Cross-validation'
  ],
  examples: [
    {
      language: 'python',
      description: 'Basic classification with scikit-learn',
      code: `from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)`
    }
  ],
  resources: [
    {
      title: 'Scikit-learn Tutorials',
      description: 'Official tutorials for supervised learning',
      url: 'https://scikit-learn.org/stable/supervised_learning.html'
    },
    {
      title: 'Machine Learning Mastery',
      description: 'Practical guides to supervised learning',
      url: 'https://machinelearningmastery.com/supervised-learning/'
    }
  ],
  prerequisites: ['Machine Learning Basics', 'Statistics', 'Python Programming'],
  relatedTopics: ['Classification', 'Regression', 'Model Evaluation']
};