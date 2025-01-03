import React from 'react';
import TransformerTools from './TransformerTools';
import VectorDBTools from './VectorDBTools';
import TrainingTools from './TrainingTools';
import InferenceTools from './InferenceTools';

function MLTools() {
  return (
    <section id="ml-tools">
      <h2>Machine Learning Tools</h2>
      <p>
        Advanced tools and utilities for machine learning tasks, including transformer models,
        vector databases, model training, and inference optimization.
      </p>

      <div className="space-y-12">
        <TransformerTools />
        <VectorDBTools />
        <TrainingTools />
        <InferenceTools />
      </div>
    </section>
  );
}

export default MLTools;