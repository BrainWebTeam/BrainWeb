import React from 'react';
import CodeBlock from '../../CodeBlock';

function RenderingOptimization() {
  return (
    <section>
      <h2>Rendering Optimization</h2>
      <p>
        Techniques to optimize React rendering and minimize unnecessary updates:
      </p>

      <CodeBlock
        language="typescript"
        code={`// Virtual list implementation
function VirtualList({ items, rowHeight, visibleRows }) {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const startIndex = Math.floor(scrollTop / rowHeight);
  const endIndex = Math.min(
    startIndex + visibleRows,
    items.length
  );

  const visibleItems = items.slice(startIndex, endIndex);
  const offsetY = startIndex * rowHeight;

  return (
    <div 
      ref={containerRef}
      style={{ height: items.length * rowHeight }}
      onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
    >
      <div style={{ transform: \`translateY(\${offsetY}px)\` }}>
        {visibleItems.map(item => (
          <ListItem key={item.id} data={item} />
        ))}
      </div>
    </div>
  );
}`}
      />
    </section>
  );
}

export default RenderingOptimization;