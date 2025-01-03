import React from 'react';
import CodeBlock from '../../CodeBlock';

function SecurityBestPractices() {
  return (
    <section>
      <h2>Security Best Practices</h2>
      <p>
        Implementation of security best practices and vulnerability prevention:
      </p>

      <CodeBlock
        language="typescript"
        code={`// Content Security Policy
const csp = {
  directives: {
    defaultSrc: ["'self'"],
    scriptSrc: ["'self'", "'unsafe-inline'"],
    styleSrc: ["'self'", "'unsafe-inline'"],
    imgSrc: ["'self'", "data:", "https:"],
    connectSrc: ["'self'", "https://api.example.com"],
    frameSrc: ["'none'"],
    objectSrc: ["'none'"],
    baseUri: ["'self'"],
    formAction: ["'self'"],
    frameAncestors: ["'none'"]
  }
};

// Rate limiting
const rateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests, please try again later',
  headers: true,
  handler: (req, res) => {
    res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: res.getHeader('Retry-After')
    });
  }
});`}
      />
    </section>
  );
}

export default SecurityBestPractices;