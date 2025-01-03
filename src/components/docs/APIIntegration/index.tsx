import React from 'react';
import RESTEndpoints from './RESTEndpoints';
import Authentication from './Authentication';
import ErrorHandling from './ErrorHandling';
import DataValidation from './DataValidation';

function APIIntegration() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1>API Integration</h1>
      <p>
        Comprehensive guide to integrating with the AI Learning Network API, 
        including authentication, endpoints, and best practices.
      </p>
      
      <RESTEndpoints />
      <Authentication />
      <ErrorHandling />
      <DataValidation />
    </div>
  );
}

export default APIIntegration;