import React from 'react';
import CodeBlock from '../../CodeBlock';

function DeploymentTools() {
  return (
    <section>
      <h2>Deployment Tools</h2>
      <p>
        The DeploymentTools class provides methods for deploying and managing applications across different environments.
        It handles deployment configuration, status tracking, and environment management.
      </p>

      <h3>Class Methods</h3>
      
      <h4>getDeploymentStatus()</h4>
      <p>
        This method retrieves the current deployment status, including build progress, deployment URL,
        and any error information. It's used to monitor ongoing deployments and verify successful completion.
      </p>

      <CodeBlock
        language="typescript"
        code={`// Check deployment status
const status = await DeploymentTools.getDeploymentStatus({
  id: "deployment-123"
});

// Status response
{
  status: "success", // or "building", "error"
  url: "https://my-app.netlify.app",
  error?: string,
  progress?: number
}`}
      />

      <div className="bg-gray-800 rounded-lg p-6 my-8">
        <h4 className="text-lg font-semibold mb-4">Usage Notes</h4>
        <ul className="space-y-2">
          <li>Deployment status checks should be polled at reasonable intervals (e.g., every 5 seconds)</li>
          <li>Handle potential deployment failures gracefully with user feedback</li>
          <li>Store deployment IDs for future status checks and management</li>
          <li>Monitor deployment progress to provide accurate user feedback</li>
        </ul>
      </div>

      <h4>Example Usage</h4>
      <CodeBlock
        language="typescript"
        code={`// Create deployment manager
const deploymentManager = {
  async checkStatus(deployId: string) {
    try {
      const status = await DeploymentTools.getDeploymentStatus({
        id: deployId
      });

      if (status.status === 'success') {
        console.log(\`Deployment live at: \${status.url}\`);
      } else if (status.status === 'error') {
        console.error(\`Deployment failed: \${status.error}\`);
      } else {
        console.log(\`Building: \${status.progress}%\`);
      }
    } catch (error) {
      console.error('Failed to check deployment status:', error);
    }
  }
};

// Monitor deployment
const monitorDeployment = async (deployId: string) => {
  const checkInterval = setInterval(async () => {
    const status = await deploymentManager.checkStatus(deployId);
    
    if (status.status !== 'building') {
      clearInterval(checkInterval);
    }
  }, 5000);
};`}
      />

      <div className="bg-gray-800 rounded-lg p-6 my-8">
        <h4 className="text-lg font-semibold mb-4">Error Handling</h4>
        <p className="mb-4">The deployment tools include comprehensive error handling:</p>
        <ul className="space-y-2">
          <li>Network connectivity issues</li>
          <li>Invalid deployment configurations</li>
          <li>Build failures and timeouts</li>
          <li>Environment-specific deployment errors</li>
        </ul>
      </div>
    </section>
  );
}

export default DeploymentTools;