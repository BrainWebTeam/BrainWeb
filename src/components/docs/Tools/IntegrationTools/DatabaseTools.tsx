import React from 'react';
import CodeBlock from '../../../CodeBlock';

function DatabaseTools() {
  return (
    <section id="database-tools">
      <h2>Database Tools</h2>
      <p>
        Tools for database integration and management with support for multiple
        database types and operations.
      </p>

      <CodeBlock
        language="typescript"
        code={`interface DatabaseConfig {
  connection: {
    type: 'postgres' | 'mysql' | 'mongodb' | 'redis';
    url: string;
    pool: {
      min: number;
      max: number;
      idle: number;
    };
  };
  migrations: {
    enabled: boolean;
    directory: string;
    tableName: string;
  };
}`}
      />
    </section>
  );
}

export default DatabaseTools;