import React from 'react';
import CodeBlock from '../../CodeBlock';

function NetworkOptimization() {
  return (
    <section>
      <h2>Network Optimization</h2>
      <p>
        Techniques for optimizing data loading and network requests:
      </p>

      <CodeBlock
        language="typescript"
        code={`// Request deduplication and caching
const cache = new Map<string, Promise<any>>();

async function fetchWithCache(url: string) {
  if (cache.has(url)) {
    return cache.get(url);
  }

  const promise = fetch(url)
    .then(res => res.json())
    .finally(() => {
      setTimeout(() => cache.delete(url), 60000);
    });

  cache.set(url, promise);
  return promise;
}

// Prefetching critical resources
const prefetchLinks = () => {
  const links = document.querySelectorAll('a');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const link = entry.target as HTMLAnchorElement;
        const href = link.href;
        
        if (href && !prefetched.has(href)) {
          const prefetchLink = document.createElement('link');
          prefetchLink.rel = 'prefetch';
          prefetchLink.href = href;
          document.head.appendChild(prefetchLink);
          prefetched.add(href);
        }
      }
    });
  });

  links.forEach(link => observer.observe(link));
  return () => observer.disconnect();
};`}
      />
    </section>
  );
}

export default NetworkOptimization;