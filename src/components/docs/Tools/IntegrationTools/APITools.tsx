import React from 'react';
import CodeBlock from '../../../CodeBlock';

function APITools() {
  return (
    <section id="api-tools">
      <h2>API Tools</h2>
      <p>
        Tools for building and consuming APIs with support for multiple protocols
        and data formats.
      </p>

      <CodeBlock
        language="typescript"
        code={`interface APIConfig {
  client: {
    baseURL: string;
    timeout: number;
    retries: number;
    rateLimit: {
      enabled: boolean;
      maxRequests: number;
      window: number;
    };
  };
  auth: {
    type: 'basic' | 'bearer' | 'oauth2';
    credentials: AuthCredentials;
    refreshToken?: boolean;
  };
}`}
      />
    </section>
  );
}

export default APITools;