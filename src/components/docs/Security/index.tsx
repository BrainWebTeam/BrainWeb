import React from 'react';
import Authentication from './Authentication';
import DataProtection from './DataProtection';
import NetworkSecurity from './NetworkSecurity';
import SecurityBestPractices from './SecurityBestPractices';

function Security() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1>Security & Privacy</h1>
      <p>
        Comprehensive security measures and privacy controls implemented throughout 
        the AI Learning Network to protect user data and system integrity.
      </p>
      
      <Authentication />
      <DataProtection />
      <NetworkSecurity />
      <SecurityBestPractices />
    </div>
  );
}

export default Security;