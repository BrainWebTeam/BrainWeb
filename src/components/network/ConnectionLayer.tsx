import React, { useMemo } from 'react';
import Connection from './Connection';
import { Node, Connection as ConnectionType } from '../../types/network';

interface ConnectionLayerProps {
  connections: ConnectionType[];
  nodes: Node[];
  scale: number;
}

const ConnectionLayer: React.FC<ConnectionLayerProps> = ({ connections, nodes, scale }) => {
  // Memoize active connections
  const activeConnections = useMemo(() => {
    return connections.filter(conn => {
      const source = nodes.find(n => n.id === conn.sourceId);
      const target = nodes.find(n => n.id === conn.targetId);
      return source && target && (
        source.activated || target.activated ||
        source.available || target.available
      );
    });
  }, [connections, nodes]);

  // Sort connections by layer (memoized)
  const sortedConnections = useMemo(() => {
    return [...activeConnections].sort((a, b) => {
      const aFromCenter = nodes.find(n => n.id === a.sourceId)?.type === 'center';
      const bFromCenter = nodes.find(n => n.id === b.sourceId)?.type === 'center';
      if (aFromCenter && !bFromCenter) return -1;
      if (!aFromCenter && bFromCenter) return 1;
      return 0;
    });
  }, [activeConnections, nodes]);

  return (
    <g className="connections-layer">
      {sortedConnections.map((conn, index) => {
        const source = nodes.find(n => n.id === conn.sourceId);
        const target = nodes.find(n => n.id === conn.targetId);
        
        if (!source || !target) return null;

        const baseDelay = source.type === 'center' ? 0 : 
                         source.type === 'primary' ? 0.2 :
                         source.type === 'secondary' ? 0.4 :
                         source.type === 'tertiary' ? 0.6 : 0.8;
        const delay = baseDelay + (index % 4) * 0.05;

        return (
          <Connection
            key={conn.id}
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            strength={conn.strength}
            scale={scale}
            sourceActivated={source.activated || false}
            targetActivated={target.activated || false}
            targetAvailable={target.available || false}
            delay={delay}
          />
        );
      })}
    </g>
  );
};

export default React.memo(ConnectionLayer);