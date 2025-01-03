import { NodeContent } from '../../../../types/content';

export const modelDeploymentContent: NodeContent = {
  title: 'Model Deployment',
  description: 'The process of making machine learning models available in production environments where they can be used to make predictions on new data.',
  concepts: [
    'Model Serving',
    'API Development',
    'Containerization',
    'Model Monitoring',
    'Version Control'
  ],
  examples: [
    {
      language: 'python',
      description: 'FastAPI model deployment',
      code: `from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionInput(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(input: PredictionInput):
    prediction = model.predict([input.features])
    return {"prediction": prediction[0]}`
    }
  ],
  resources: [
    {
      title: 'MLOps Guide',
      description: 'Best practices for ML deployment',
      url: 'https://ml-ops.org/'
    },
    {
      title: 'Model Deployment Course',
      description: 'Full Stack Deep Learning deployment course',
      url: 'https://fullstackdeeplearning.com/'
    }
  ],
  prerequisites: ['Machine Learning', 'Software Engineering', 'DevOps'],
  relatedTopics: ['MLOps', 'Containerization', 'Model Serving']
};