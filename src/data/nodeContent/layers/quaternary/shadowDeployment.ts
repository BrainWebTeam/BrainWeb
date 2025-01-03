import { NodeContent } from '../../../../types/content';

export const shadowDeploymentContent: NodeContent = {
  title: 'Shadow Deployment',
  description: 'A deployment strategy where a new version of a model runs in parallel with the production model, receiving the same input but without affecting the actual output.',
  concepts: [
    'Traffic Mirroring',
    'Performance Comparison',
    'Risk Mitigation',
    'Validation Strategy',
    'Production Testing'
  ],
  examples: [
    {
      language: 'python',
      description: 'Shadow deployment implementation',
      code: `import asyncio
from typing import Dict

class ShadowDeployment:
    def __init__(self, prod_model, shadow_model):
        self.prod_model = prod_model
        self.shadow_model = shadow_model
        self.metrics: Dict[str, list] = {
            'prod': [],
            'shadow': []
        }
    
    async def predict(self, input_data):
        # Run both models concurrently
        prod_future = asyncio.create_task(
            self.prod_model.predict(input_data)
        )
        shadow_future = asyncio.create_task(
            self.shadow_model.predict(input_data)
        )
        
        # Get production result
        prod_result = await prod_future
        
        # Log shadow result without affecting response
        try:
            shadow_result = await shadow_future
            self.log_comparison(prod_result, shadow_result)
        except Exception as e:
            self.log_error(str(e))
        
        return prod_result`
    }
  ],
  resources: [
    {
      title: 'Shadow Deployment Guide',
      description: 'Best practices for shadow deployments',
      url: 'https://martinfowler.com/bliki/ShadowDeployment.html'
    },
    {
      title: 'ML Model Deployment',
      description: 'Advanced deployment strategies',
      url: 'https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines'
    }
  ],
  prerequisites: ['MLOps', 'System Architecture', 'Production ML'],
  relatedTopics: ['Model Deployment', 'Testing Strategies', 'Risk Management']
};