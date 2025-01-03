import React from 'react';
import ModelOptimization from './ModelOptimization';
import PerformanceOptimization from './PerformanceOptimization';
import MemoryOptimization from './MemoryOptimization';
import InferenceOptimization from './InferenceOptimization';

function OptimizationTools() {
  return (
    <section id="optimization-tools">
      <h2>Optimization Tools</h2>
      <p>
        Advanced optimization tools for model compression, performance tuning,
        memory management, and inference acceleration.
      </p>

      <div className="space-y-12">
        <ModelOptimization />
        <PerformanceOptimization />
        <MemoryOptimization />
        <InferenceOptimization />
      </div>
    </section>
  );
}

export default OptimizationTools;