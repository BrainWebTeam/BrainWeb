import React from 'react';
import TrackingTools from './TrackingTools';
import ReportingTools from './ReportingTools';
import VisualizationTools from './VisualizationTools';
import ExportTools from './ExportTools';

function AnalyticsTools() {
  return (
    <section id="analytics-tools">
      <h2>Analytics Tools</h2>
      <p>
        Advanced analytics tools for tracking, reporting, visualization, and data export.
        These tools provide comprehensive insights into system usage and performance.
      </p>

      <div className="space-y-12">
        <TrackingTools />
        <ReportingTools />
        <VisualizationTools />
        <ExportTools />
      </div>
    </section>
  );
}

export default AnalyticsTools;