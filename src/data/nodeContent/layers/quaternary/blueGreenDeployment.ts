import { NodeContent } from '../../../../types/content';

export const blueGreenDeploymentContent: NodeContent = {
  title: 'Blue-Green Deployment',
  description: 'A deployment strategy that maintains two identical production environments, allowing instant rollback and zero-downtime deployments.',
  concepts: [
    'Environment Switching',
    'Zero Downtime',
    'Instant Rollback',
    'Traffic Routing',
    'State Management'
  ],
  examples: [
    {
      language: 'python',
      description: 'Blue-green deployment implementation',
      code: `from enum import Enum
from typing import Dict, Optional

class Environment(Enum):
    BLUE = 'blue'
    GREEN = 'green'

class BlueGreenDeployment:
    def __init__(self):
        self.environments: Dict[Environment, Optional[Model]] = {
            Environment.BLUE: None,
            Environment.GREEN: None
        }
        self.active = Environment.BLUE
    
    def deploy_new_version(self, new_model, environment: Environment):
        # Deploy to inactive environment
        self.environments[environment] = new_model
        
    def switch_environment(self):
        # Validate inactive environment
        new_env = Environment.GREEN if self.active == Environment.BLUE else Environment.BLUE
        if not self.validate_environment(new_env):
            raise ValueError("New environment validation failed")
        
        # Switch active environment
        self.active = new_env
        return f"Switched to {self.active.value} environment"
    
    def rollback(self):
        # Instant rollback to previous environment
        self.active = Environment.BLUE if self.active == Environment.GREEN else Environment.GREEN
        return f"Rolled back to {self.active.value} environment"`
    }
  ],
  resources: [
    {
      title: 'Blue-Green Deployment',
      description: 'Comprehensive guide to blue-green deployments',
      url: 'https://martinfowler.com/bliki/BlueGreenDeployment.html'
    },
    {
      title: 'Zero Downtime Deployment',
      description: 'Implementing zero-downtime deployments',
      url: 'https://docs.aws.amazon.com/whitepapers/latest/blue-green-deployments/welcome.html'
    }
  ],
  prerequisites: ['MLOps', 'System Architecture', 'Deployment Strategies'],
  relatedTopics: ['Deployment Patterns', 'High Availability', 'Risk Management']
};