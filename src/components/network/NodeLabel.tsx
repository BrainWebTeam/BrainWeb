import React from 'react';
import { motion } from 'framer-motion';

interface NodeLabelProps {
  x: number;
  y: number;
  label: string;
  activated?: boolean;
  available?: boolean;
}

const NodeLabel: React.FC<NodeLabelProps> = ({
  x, y, label,
  activated = false,
  available = false
}) => {
  // Only render if node is active or available
  if (!activated && !available) {
    return null;
  }

  return (
    <motion.text
      x={x}
      y={y + 20}
      className={`
        text-xs text-center select-none pointer-events-none font-medium
        ${activated 
          ? 'fill-blue-400' 
          : 'fill-violet-400/80'}
      `}
      textAnchor="middle"
      initial={{ opacity: 0 }}
      animate={{
        opacity: activated ? 1 : 0.8,
        scale: activated ? 1.1 : 1
      }}
      transition={{
        duration: 0.3,
        ease: "easeOut"
      }}
    >
      {label}
    </motion.text>
  );
};

export default NodeLabel;