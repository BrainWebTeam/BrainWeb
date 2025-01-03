import React from 'react';
import CodeBlock from '../../../CodeBlock';

function MessagingTools() {
  return (
    <section id="messaging-tools">
      <h2>Messaging Tools</h2>
      <p>
        Tools for integrating with message queues and event streaming platforms.
      </p>

      <CodeBlock
        language="typescript"
        code={`interface MessagingConfig {
  broker: {
    type: 'rabbitmq' | 'kafka' | 'redis';
    url: string;
    options: BrokerOptions;
  };
  queues: {
    durable: boolean;
    deadLetter: boolean;
    maxRetries: number;
    ttl: number;
  };
}`}
      />
    </section>
  );
}

export default MessagingTools;