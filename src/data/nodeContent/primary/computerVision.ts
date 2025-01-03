import { NodeContent } from '../../../types/content';

export const computerVisionContent: NodeContent = {
  title: 'Computer Vision',
  description: 'The field of AI that enables computers to understand and process visual information from the world, including images and videos.',
  concepts: [
    'Image Processing',
    'Object Detection',
    'Image Segmentation',
    'Feature Extraction',
    'Pattern Recognition'
  ],
  examples: [
    {
      language: 'python',
      description: 'Basic image processing with OpenCV',
      code: `import cv2
import numpy as np

# Load and process image
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 100, 200)

# Detect objects
objects = cv2.findContours(
    edges, 
    cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE
)`
    }
  ],
  resources: [
    {
      title: 'PyImageSearch',
      description: 'Practical computer vision tutorials',
      url: 'https://pyimagesearch.com/'
    },
    {
      title: 'OpenCV Documentation',
      description: 'Comprehensive guide to OpenCV',
      url: 'https://docs.opencv.org/'
    }
  ],
  prerequisites: ['Python Programming', 'Linear Algebra', 'Image Processing Basics'],
  relatedTopics: ['Deep Learning', 'CNNs', 'Object Detection']
};