import React from 'react';
import CodeBlock from '../../../CodeBlock';

function TrackingTools() {
  return (
    <section id="tracking-tools">
      <h2>Tracking Tools</h2>
      <p>
        Tools for tracking user interactions, system events, and performance metrics.
      </p>

      <CodeBlock
        language="typescript"
        code={`interface TrackingConfig {
  events: {
    interactions: boolean;
    performance: boolean;
    errors: boolean;
    custom: string[];
  };
  sampling: {
    enabled: boolean;
    rate: number;
    rules: SamplingRule[];
  };
  storage: {
    backend: 'clickhouse' | 'bigquery' | 'elasticsearch';
    retention: number;
    partitioning: string[];
  };
  batching: {
    enabled: boolean;
    maxSize: number;
    maxWait: number;
  };
}`}
      />
    </section>
  );
}

export default TrackingTools;