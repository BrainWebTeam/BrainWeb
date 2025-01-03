import React from 'react';
import CodeBlock from '../../CodeBlock';

function Authentication() {
  return (
    <section>
      <h2>API Authentication</h2>
      <p>
        Secure API authentication and token management:
      </p>

      <CodeBlock
        language="typescript"
        code={`// Auth interceptor
const authInterceptor = (config: AxiosRequestConfig) => {
  const token = getAuthToken();
  if (token) {
    config.headers.Authorization = \`Bearer \${token}\`;
  }
  return config;
};

// Token refresh
const refreshAuthToken = async () => {
  const refreshToken = getRefreshToken();
  if (!refreshToken) throw new Error('No refresh token');

  const { data } = await api.post('/auth/refresh', {
    refreshToken
  });

  setAuthToken(data.accessToken);
  return data.accessToken;
};

// Auto refresh setup
api.interceptors.response.use(
  response => response,
  async error => {
    const originalRequest = error.config;
    
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      const token = await refreshAuthToken();
      originalRequest.headers.Authorization = \`Bearer \${token}\`;
      return api(originalRequest);
    }
    
    return Promise.reject(error);
  }
);`}
      />
    </section>
  );
}

export default Authentication;