import { NodeContent } from '../../../../types/content';

export const modelVersioningContent: NodeContent = {
  title: 'Model Versioning',
  description: 'Techniques and practices for tracking different versions of machine learning models, including their code, data, and hyperparameters.',
  concepts: [
    'Version Control',
    'Reproducibility',
    'Model Lineage',
    'Dependency Management',
    'Configuration Tracking'
  ],
  examples: [
    {
      language: 'python',
      description: 'DVC model versioning',
      code: `import dvc.api
from dvc.repo import Repo

# Initialize DVC repository
repo = Repo.init()

# Track model file
repo.add('models/model.pkl')

# Add model parameters
params = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}
repo.config.save_params('params.yaml', params)

# Create a version
repo.commit('Add initial model version')
repo.push()

# Load specific version
with dvc.api.open(
    'models/model.pkl',
    rev='v1.0'
) as f:
    model_v1 = pickle.load(f)`
    }
  ],
  resources: [
    {
      title: 'DVC Documentation',
      description: 'Data Version Control for ML projects',
      url: 'https://dvc.org/doc'
    },
    {
      title: 'ML Versioning',
      description: 'Best practices for ML versioning',
      url: 'https://neptune.ai/blog/version-control-for-ml-projects'
    }
  ],
  prerequisites: ['Version Control', 'MLOps', 'Software Engineering'],
  relatedTopics: ['Model Registry', 'Reproducibility', 'Model Management']
};