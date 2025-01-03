import React from 'react';
import DataProcessing from './DataProcessing';
import DataValidation from './DataValidation';
import DataAugmentation from './DataAugmentation';
import DataPipelines from './DataPipelines';

function DataTools() {
  return (
    <section id="data-tools">
      <h2>Data Tools</h2>
      <p>
        Advanced tools for data processing, validation, augmentation, and pipeline management.
        These tools provide enterprise-grade functionality for handling large-scale data operations.
      </p>

      <div className="space-y-12">
        <DataProcessing />
        <DataValidation />
        <DataAugmentation />
        <DataPipelines />
      </div>
    </section>
  );
}

export default DataTools;