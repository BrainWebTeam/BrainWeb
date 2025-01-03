import React from 'react';
import { motion } from 'framer-motion';
import { Node } from '../../types/network';
import { isValidPosition } from '../../utils/validation';

interface OuterNodeEffectsProps {
  node: Node;
  time: number;
}

const OuterNodeEffects: React.FC<OuterNodeEffectsProps> = ({ node, time }) => {
  const isOuter = node.type === 'tertiary' || node.type === 'quaternary';
  if (!isOuter || !node.available || !isValidPosition({ x: node.x, y: node.y })) return null;

  const baseSize = node.type === 'tertiary' ? 6 : 4;
  const wobbleAmount = node.type === 'tertiary' ? 3 : 2;
  const wobbleSpeed = node.type === 'tertiary' ? 1 : 1.5;

  // Calculate dynamic position with orbital motion and ensure valid numbers
  const nodeId = parseInt(node.id.split('-')[1] || '0');
  const wobbleX = Math.cos(time * wobbleSpeed + nodeId * 0.5) * wobbleAmount;
  const wobbleY = Math.sin(time * wobbleSpeed + nodeId * 0.5) * wobbleAmount;

  // Ensure coordinates are valid numbers
  const x = Number.isFinite(node.x + wobbleX) ? node.x + wobbleX : node.x;
  const y = Number.isFinite(node.y + wobbleY) ? node.y + wobbleY : node.y;

  return (
    <>
      {/* Knowledge flow effect */}
      <motion.circle
        cx={x}
        cy={y}
        r={baseSize * 1.5}
        className="fill-none stroke-black/10"
        initial={false}
        animate={{
          scale: [1, 1.5, 1],
          opacity: [0.2, 0.4, 0.2],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Particle effect */}
      {node.type === 'quaternary' && (
        <g>
          {[...Array(3)].map((_, i) => {
            const angle = (time * 2 + i * (Math.PI * 2 / 3));
            const radius = baseSize * 2;
            const px = node.x + Math.cos(angle) * radius;
            const py = node.y + Math.sin(angle) * radius;

            // Skip invalid coordinates
            if (!isValidPosition({ x: px, y: py })) return null;

            return (
              <motion.circle
                key={i}
                cx={px}
                cy={py}
                r={0.5}
                className="fill-black/20"
                initial={false}
                animate={{
                  opacity: [0.4, 0.1, 0.4],
                }}
                transition={{
                  duration: 2,
                  delay: i * 0.3,
                  repeat: Infinity,
                }}
              />
            );
          })}
        </g>
      )}

      {/* Connection ripple */}
      {node.available && (
        <motion.circle
          cx={node.x}
          cy={node.y}
          r={baseSize * 2}
          className="fill-none stroke-black/5"
          initial={{ scale: 0.8, opacity: 0.5 }}
          animate={{
            scale: [1, 2],
            opacity: [0.3, 0],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeOut",
          }}
        />
      )}
    </>
  );
};

export default OuterNodeEffects;