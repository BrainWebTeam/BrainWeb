import React from 'react';
import CodeBlock from '../../CodeBlock';

function CodeOptimization() {
  return (
    <section>
      <h2>Code Optimization</h2>
      <p>
        Efficient code patterns and optimization techniques to maximize performance:
      </p>

      <CodeBlock
        language="typescript"
        code={`// Memoization for expensive calculations
const memoizedCalculation = useMemo(() => {
  return expensiveOperation(data);
}, [data]);

// Efficient event handlers
const debouncedHandler = useCallback(
  debounce((value: string) => {
    processInput(value);
  }, 300),
  []
);

// Web Worker for heavy computations
const worker = new Worker(new URL(
  '../workers/compute.worker.ts',
  import.meta.url
));

worker.postMessage({ data: complexData });
worker.onmessage = (e) => {
  setResult(e.data);
};`}
      />
    </section>
  );
}

export default CodeOptimization;