import { NodeContent } from '../../../../types/content';

export const canaryReleaseContent: NodeContent = {
  title: 'Canary Release',
  description: 'A deployment strategy where a new model version is gradually rolled out to a small subset of users before full deployment.',
  concepts: [
    'Traffic Routing',
    'Gradual Rollout',
    'Risk Assessment',
    'Monitoring Strategy',
    'Rollback Planning'
  ],
  examples: [
    {
      language: 'python',
      description: 'Canary deployment implementation',
      code: `class CanaryDeployment:
    def __init__(
        self,
        old_model,
        new_model,
        initial_percent=5,
        increment=5,
        threshold=0.95
    ):
        self.old_model = old_model
        self.new_model = new_model
        self.percent = initial_percent
        self.increment = increment
        self.threshold = threshold
        self.metrics = []
    
    def route_request(self, request_id, input_data):
        # Determine routing based on request ID
        use_new_model = hash(request_id) % 100 < self.percent
        
        if use_new_model:
            result = self.new_model.predict(input_data)
            self.monitor_performance(result)
            
            # Increase traffic if performance is good
            if self.check_performance():
                self.percent = min(100, self.percent + self.increment)
        else:
            result = self.old_model.predict(input_data)
        
        return result`
    }
  ],
  resources: [
    {
      title: 'Canary Deployments',
      description: 'Guide to canary deployment strategy',
      url: 'https://martinfowler.com/bliki/CanaryRelease.html'
    },
    {
      title: 'Progressive Delivery',
      description: 'Advanced deployment patterns',
      url: 'https://www.split.io/blog/progressive-delivery-canary-deployment/'
    }
  ],
  prerequisites: ['MLOps', 'Deployment Strategies', 'Monitoring'],
  relatedTopics: ['Deployment Patterns', 'Risk Management', 'Model Monitoring']
};