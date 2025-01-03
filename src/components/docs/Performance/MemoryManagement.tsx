import React from 'react';
import CodeBlock from '../../CodeBlock';

function MemoryManagement() {
  return (
    <section>
      <h2>Memory Management</h2>
      <p>
        Strategies for efficient memory usage and leak prevention:
      </p>

      <CodeBlock
        language="typescript"
        code={`// Resource cleanup with useEffect
function ResourceManager() {
  useEffect(() => {
    const cache = new Map();
    const worker = new Worker('worker.js');
    const observer = new IntersectionObserver(callback);

    return () => {
      cache.clear();
      worker.terminate();
      observer.disconnect();
    };
  }, []);
}

// Memory-efficient data structures
class CircularBuffer<T> {
  private buffer: T[];
  private head = 0;
  private tail = 0;
  private size = 0;

  constructor(private capacity: number) {
    this.buffer = new Array(capacity);
  }

  push(item: T): void {
    this.buffer[this.tail] = item;
    this.tail = (this.tail + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
    if (this.size === this.capacity) {
      this.head = (this.head + 1) % this.capacity;
    }
  }

  get items(): T[] {
    return [...this.buffer.slice(this.head, this.size)];
  }
}`}
      />
    </section>
  );
}

export default MemoryManagement;