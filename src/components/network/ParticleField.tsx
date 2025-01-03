import React, { useMemo } from 'react';
import { generateRandomPoints } from '../../utils/geometry';

interface ParticleFieldProps {
  count: number;
  bounds: { width: number; height: number };
  time: number;
}

const ParticleField: React.FC<ParticleFieldProps> = ({ count, bounds, time }) => {
  const basePoints = useMemo(() => 
    generateRandomPoints(count, bounds), [count, bounds]);

  const points = basePoints.map((point, i) => ({
    x: point.x + Math.sin(time + point.x * 0.01 + i * 0.5) * 20,
    y: point.y + Math.cos(time + point.y * 0.01 + i * 0.3) * 20,
    size: 1 + Math.sin(time * 2 + i) * 0.5,
    opacity: 0.2 + Math.sin(time + i * 0.5) * 0.1
  }));

  return (
    <g className="particle-field">
      {points.map((point, i) => (
        <circle
          key={i}
          cx={point.x}
          cy={point.y}
          r={point.size}
          className="fill-gray-200"
          style={{ opacity: point.opacity }}
        />
      ))}
    </g>
  );
};

export default ParticleField;