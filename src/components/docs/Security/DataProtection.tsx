import React from 'react';
import CodeBlock from '../../CodeBlock';

function DataProtection() {
  return (
    <section>
      <h2>Data Protection</h2>
      <p>
        Measures for securing sensitive data at rest and in transit:
      </p>

      <CodeBlock
        language="typescript"
        code={`// Data encryption utilities
import { createCipheriv, createDecipheriv } from 'crypto';

class DataEncryption {
  private readonly algorithm = 'aes-256-gcm';
  private readonly keyLength = 32;

  async encrypt(data: string, key: Buffer): Promise<string> {
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const cipher = createCipheriv(this.algorithm, key, iv);
    
    const encrypted = Buffer.concat([
      cipher.update(data, 'utf8'),
      cipher.final()
    ]);

    const tag = cipher.getAuthTag();
    
    return JSON.stringify({
      iv: iv.toString('base64'),
      data: encrypted.toString('base64'),
      tag: tag.toString('base64')
    });
  }

  async decrypt(
    encryptedData: string, 
    key: Buffer
  ): Promise<string> {
    const { iv, data, tag } = JSON.parse(encryptedData);
    const decipher = createDecipheriv(
      this.algorithm,
      key,
      Buffer.from(iv, 'base64')
    );
    
    decipher.setAuthTag(Buffer.from(tag, 'base64'));
    
    return decipher.update(
      Buffer.from(data, 'base64')
    ).toString('utf8') + decipher.final('utf8');
  }
}`}
      />
    </section>
  );
}

export default DataProtection;