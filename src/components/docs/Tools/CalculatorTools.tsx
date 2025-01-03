import React from 'react';
import CodeBlock from '../../CodeBlock';

function CalculatorTools() {
  return (
    <section id="calculator-tools">
      <h2>Calculator Tools</h2>
      <p>
        Mathematical and statistical calculation tools for data analysis.
      </p>

      <h3>Class Methods</h3>

      <h4>calculateStatistics()</h4>
      <p>
        Performs statistical analysis on numerical datasets.
      </p>

      <CodeBlock
        language="typescript"
        code={`// Calculate statistics
const stats = await CalculatorTools.calculateStatistics({
  data: [1, 2, 3, 4, 5],
  metrics: ["mean", "median", "stdDev", "variance"]
});`}
      />

      <h4>performRegression()</h4>
      <p>
        Performs regression analysis on data points.
      </p>

      <CodeBlock
        language="typescript"
        code={`// Perform regression analysis
const regression = await CalculatorTools.performRegression({
  x: [1, 2, 3, 4, 5],
  y: [2.1, 3.8, 5.2, 7.1, 8.9],
  type: "linear"
});`}
      />
    </section>
  );
}

export default CalculatorTools;