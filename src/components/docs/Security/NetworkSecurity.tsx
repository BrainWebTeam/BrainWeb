import React from 'react';
import CodeBlock from '../../CodeBlock';

function NetworkSecurity() {
  return (
    <section>
      <h2>Network Security</h2>
      <p>
        Implementation of secure communication protocols and request validation:
      </p>

      <CodeBlock
        language="typescript"
        code={`// Request validation middleware
const validateRequest = (schema: Schema) => {
  return async (
    req: Request,
    res: Response,
    next: NextFunction
  ) => {
    try {
      const validated = await schema.validate(req.body);
      req.body = validated;
      next();
    } catch (error) {
      res.status(400).json({
        error: 'Invalid request data',
        details: error.errors
      });
    }
  };
};

// CSRF protection
const csrfProtection = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const token = req.headers['x-csrf-token'];
  const storedToken = req.session.csrfToken;
  
  if (!token || token !== storedToken) {
    return res.status(403).json({
      error: 'Invalid CSRF token'
    });
  }
  
  next();
};`}
      />
    </section>
  );
}

export default NetworkSecurity;