import React from 'react';
import CodeBlock from '../../CodeBlock';

function Authentication() {
  return (
    <section>
      <h2>Authentication & Authorization</h2>
      <p>
        Secure user authentication and role-based access control implementation:
      </p>

      <CodeBlock
        language="typescript"
        code={`// Role-based access control
interface UserRole {
  id: string;
  permissions: string[];
}

const checkPermission = (
  user: User,
  requiredPermission: string
): boolean => {
  return user.roles.some(role => 
    role.permissions.includes(requiredPermission)
  );
};

// Protected route wrapper
function ProtectedRoute({ 
  children, 
  requiredPermission 
}: ProtectedRouteProps) {
  const { user } = useAuth();
  
  if (!user || !checkPermission(user, requiredPermission)) {
    return <Navigate to="/login" />;
  }
  
  return <>{children}</>;
}`}
      />
    </section>
  );
}

export default Authentication;