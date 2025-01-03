import React from 'react';
import APITools from './APITools';
import DatabaseTools from './DatabaseTools';
import MessagingTools from './MessagingTools';
import WebhookTools from './WebhookTools';

function IntegrationTools() {
  return (
    <section id="integration-tools">
      <h2>Integration Tools</h2>
      <p>
        Advanced tools for integrating with external systems, APIs, databases,
        and messaging systems. These tools provide enterprise-grade functionality
        for building robust integrations.
      </p>

      <div className="space-y-12">
        <APITools />
        <DatabaseTools />
        <MessagingTools />
        <WebhookTools />
      </div>
    </section>
  );
}

export default IntegrationTools;