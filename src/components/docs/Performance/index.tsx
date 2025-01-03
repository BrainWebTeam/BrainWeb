import React from 'react';
import CodeOptimization from './CodeOptimization';
import RenderingOptimization from './RenderingOptimization';
import MemoryManagement from './MemoryManagement';
import NetworkOptimization from './NetworkOptimization';

function Performance() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1>Performance</h1>
      <p>
        Comprehensive performance optimization strategies for the AI Learning Network, 
        focusing on code efficiency, rendering, memory management, and network optimization.
      </p>
      
      <CodeOptimization />
      <RenderingOptimization />
      <MemoryManagement />
      <NetworkOptimization />
    </div>
  );
}

export default Performance;