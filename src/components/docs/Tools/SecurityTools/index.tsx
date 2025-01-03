import React from 'react';
import AuthenticationTools from './AuthenticationTools';
import EncryptionTools from './EncryptionTools';
import AccessControlTools from './AccessControlTools';
import AuditingTools from './AuditingTools';

function SecurityTools() {
  return (
    <section id="security-tools">
      <h2>Security Tools</h2>
      <p>
        Enterprise-grade security tools for authentication, encryption, access control,
        and security auditing.
      </p>

      <div className="space-y-12">
        <AuthenticationTools />
        <EncryptionTools />
        <AccessControlTools />
        <AuditingTools />
      </div>
    </section>
  );
}

export default SecurityTools;