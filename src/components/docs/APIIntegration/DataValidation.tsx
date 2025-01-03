import React from 'react';
import CodeBlock from '../../CodeBlock';

function DataValidation() {
  return (
    <section>
      <h2>Data Validation</h2>
      <p>
        Request and response data validation:
      </p>

      <CodeBlock
        language="typescript"
        code={`// Request validation schema
const progressSchema = z.object({
  nodeId: z.string().uuid(),
  status: z.enum(['started', 'completed']),
  score: z.number().min(0).max(100).optional(),
  timestamp: z.string().datetime()
});

// Response validation
const validateResponse = <T,>(
  data: unknown,
  schema: z.ZodSchema<T>
): T => {
  try {
    return schema.parse(data);
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw new APIError({
        code: 'INVALID_RESPONSE',
        message: 'Invalid response data',
        details: error.errors
      });
    }
    throw error;
  }
};

// Middleware setup
const validateRequest = (schema: z.ZodSchema) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      req.body = schema.parse(req.body);
      next();
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({
          code: 'VALIDATION_ERROR',
          message: 'Invalid request data',
          details: error.errors
        });
        return;
      }
      next(error);
    }
  };
};`}
      />
    </section>
  );
}

export default DataValidation;