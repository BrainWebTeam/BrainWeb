import React from 'react';
import CodeBlock from '../../../CodeBlock';

function VisualizationTools() {
  return (
    <section id="visualization-tools">
      <h2>Visualization Tools</h2>
      <p>
        Tools for creating interactive data visualizations and dashboards.
      </p>

      <CodeBlock
        language="typescript"
        code={`interface VisualizationConfig {
  chart: {
    type: 'line' | 'bar' | 'pie' | 'scatter';
    dimensions: string[];
    metrics: string[];
    options: ChartOptions;
  };
  interactivity: {
    zoom: boolean;
    drill: boolean;
    filters: boolean;
    tooltips: TooltipConfig;
  };
  styling: {
    theme: string;
    colors: string[];
    fonts: FontConfig;
    responsive: boolean;
  };
}`}
      />
    </section>
  );
}

export default VisualizationTools;