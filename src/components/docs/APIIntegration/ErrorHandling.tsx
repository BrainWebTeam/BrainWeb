import React from 'react';
import CodeBlock from '../../CodeBlock';

function ErrorHandling() {
  return (
    <section>
      <h2>Error Handling</h2>
      <p>
        Robust error handling and recovery strategies:
      </p>

      <CodeBlock
        language="typescript"
        code={`// API error types
interface APIError {
  code: string;
  message: string;
  details?: Record<string, any>;
}

// Error handler
const handleAPIError = (error: unknown): APIError => {
  if (axios.isAxiosError(error)) {
    return {
      code: error.response?.data?.code || 'UNKNOWN_ERROR',
      message: error.response?.data?.message || 'An error occurred',
      details: error.response?.data?.details
    };
  }
  
  return {
    code: 'NETWORK_ERROR',
    message: 'Network connection error'
  };
};

// Retry mechanism
const withRetry = async <T,>(
  operation: () => Promise<T>,
  retries = 3,
  delay = 1000
): Promise<T> => {
  try {
    return await operation();
  } catch (error) {
    if (retries === 0) throw error;
    
    await new Promise(resolve => 
      setTimeout(resolve, delay)
    );
    
    return withRetry(
      operation,
      retries - 1,
      delay * 2
    );
  }
};`}
      />
    </section>
  );
}

export default ErrorHandling;