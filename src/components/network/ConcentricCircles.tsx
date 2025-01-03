import React from 'react';

interface ConcentricCirclesProps {
  center: { x: number; y: number };
  radiuses: number[];
  time: number;
}

const ConcentricCircles: React.FC<ConcentricCirclesProps> = ({ 
  center, 
  radiuses,
  time 
}) => {
  return (
    <g className="concentric-circles">
      {radiuses.map((radius, i) => {
        const wobble = Math.sin(time + i * 0.5) * 5;
        const rotation = time * (i % 2 ? 1 : -1) * 0.2;
        const dashArray = i % 2 ? "4 4" : i % 3 ? "8 4" : "2 4";
        
        return (
          <g key={i} transform={`rotate(${rotation}, ${center.x}, ${center.y})`}>
            <circle
              cx={center.x}
              cy={center.y}
              r={radius + wobble}
              className="fill-none stroke-gray-100/20"
              strokeDasharray={dashArray}
              strokeWidth={0.5}
            />
            {i % 2 === 0 && (
              <circle
                cx={center.x}
                cy={center.y}
                r={radius - 2 + wobble}
                className="fill-none stroke-gray-100/10"
                strokeDasharray="1 6"
                strokeWidth={0.25}
              />
            )}
          </g>
        );
      })}
    </g>
  );
};

export default ConcentricCircles;