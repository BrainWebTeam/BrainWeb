import React, { useEffect, useRef, useState } from 'react';
import Node from './Node';
import Connection from './Connection';

interface Point {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

const NetworkVisual: React.FC = () => {
  const [points, setPoints] = useState<Point[]>([]);
  const animationRef = useRef<number>();
  const lastUpdateRef = useRef<number>(0);

  useEffect(() => {
    // Initialize points with random positions and velocities
    const initialPoints = Array.from({ length: 15 }, () => ({
      x: Math.random() * 80 + 10, // Keep points away from edges
      y: Math.random() * 80 + 10,
      vx: (Math.random() - 0.5) * 0.1,
      vy: (Math.random() - 0.5) * 0.1,
    }));
    setPoints(initialPoints);

    const animate = (timestamp: number) => {
      if (!lastUpdateRef.current) lastUpdateRef.current = timestamp;
      const delta = timestamp - lastUpdateRef.current;

      if (delta > 16) { // Cap at ~60fps
        setPoints(prevPoints => 
          prevPoints.map(point => {
            // Update position based on velocity
            let newX = point.x + point.vx * delta;
            let newY = point.y + point.vy * delta;
            let newVx = point.vx;
            let newVy = point.vy;

            // Bounce off edges
            if (newX < 0 || newX > 100) newVx = -newVx;
            if (newY < 0 || newY > 100) newVy = -newVy;

            // Keep points within bounds
            newX = Math.max(0, Math.min(100, newX));
            newY = Math.max(0, Math.min(100, newY));

            return {
              x: newX,
              y: newY,
              vx: newVx,
              vy: newVy,
            };
          })
        );
        lastUpdateRef.current = timestamp;
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Calculate connections between nearby points
  const connections = points.flatMap((point, i) => 
    points.slice(i + 1).map(otherPoint => {
      const distance = Math.sqrt(
        Math.pow(otherPoint.x - point.x, 2) + 
        Math.pow(otherPoint.y - point.y, 2)
      );
      
      // Only connect points within a certain distance
      if (distance < 30) {
        return {
          startX: point.x,
          startY: point.y,
          endX: otherPoint.x,
          endY: otherPoint.y,
          opacity: Math.max(0.1, 1 - distance / 30),
        };
      }
      return null;
    }).filter(Boolean)
  );

  return (
    <div className="relative w-full h-full">
      {/* Render connections */}
      {connections.map((connection, i) => (
        <Connection key={`connection-${i}`} {...connection} />
      ))}
      
      {/* Render nodes */}
      {points.map((point, i) => (
        <Node 
          key={`node-${i}`}
          x={point.x}
          y={point.y}
          size={i === 0 ? 32 : 4}
          pulse={i === 0}
        />
      ))}
    </div>
  );
};

export default NetworkVisual;