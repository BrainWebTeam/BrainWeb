import React from 'react';

interface ConcentricCircleProps {
  radius: number;
  dashed?: boolean;
}

const ConcentricCircle: React.FC<ConcentricCircleProps> = ({ radius, dashed }) => {
  return (
    <circle 
      cx="500" 
      cy="500" 
      r={radius} 
      className={`stroke-gray-100 fill-none ${dashed ? 'stroke-dashed' : ''}`}
    />
  );
};

export default ConcentricCircle;