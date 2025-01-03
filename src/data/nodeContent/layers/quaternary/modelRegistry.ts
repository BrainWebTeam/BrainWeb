import { NodeContent } from '../../../../types/content';

export const modelRegistryContent: NodeContent = {
  title: 'Model Registry',
  description: 'A centralized repository for managing machine learning models throughout their lifecycle, including versioning, staging, and deployment.',
  concepts: [
    'Model Versioning',
    'Model Metadata',
    'Artifact Management',
    'Stage Transitions',
    'Deployment Tracking'
  ],
  examples: [
    {
      language: 'python',
      description: 'MLflow model registry usage',
      code: `import mlflow
from mlflow.tracking import MlflowClient

# Initialize client
client = MlflowClient()

# Register model
model_name = "recommendation_model"
model_version = mlflow.register_model(
    "runs:/d16076a3ec534311817565e6527539c0/model",
    model_name
)

# Transition to staging
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging"
)

# Add model description
client.update_model_version(
    name=model_name,
    version=model_version.version,
    description="Collaborative filtering model v2"
)`
    }
  ],
  resources: [
    {
      title: 'MLflow Model Registry',
      description: 'Guide to MLflow model registry',
      url: 'https://www.mlflow.org/docs/latest/model-registry.html'
    },
    {
      title: 'Model Management',
      description: 'Best practices for model management',
      url: 'https://neptune.ai/blog/ml-model-management'
    }
  ],
  prerequisites: ['MLOps', 'Version Control', 'Model Lifecycle'],
  relatedTopics: ['Model Versioning', 'Model Deployment', 'MLOps']
};