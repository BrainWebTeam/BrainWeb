import React from 'react';
import CodeBlock from '../CodeBlock';

function InteractionSystem() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1>Interaction System</h1>

      <h2>User Interactions</h2>
      <p>
        The network provides rich interaction capabilities through various input methods:
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Mouse Interactions</h3>
          <ul>
            <li>Click to activate nodes</li>
            <li>Drag to pan view</li>
            <li>Wheel to zoom</li>
            <li>Hover for previews</li>
            <li>Double-click to reset</li>
          </ul>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Touch Interactions</h3>
          <ul>
            <li>Tap to activate</li>
            <li>Pinch to zoom</li>
            <li>Pan to move</li>
            <li>Long press for info</li>
            <li>Double tap to reset</li>
          </ul>
        </div>
      </div>

      <h2>Event Handling</h2>
      <CodeBlock
        language="typescript"
        code={`const useInteractions = () => {
  const handleNodeClick = useCallback((nodeId: string) => {
    if (isNodeAvailable(nodeId)) {
      activateNode(nodeId);
      showNodeContent(nodeId);
    }
  }, []);

  const handleNodeHover = useCallback((nodeId: string) => {
    highlightConnections(nodeId);
    showPreview(nodeId);
  }, []);

  const handlePan = useCallback((e: MouseEvent) => {
    const newPosition = calculatePanPosition(e);
    updateViewPosition(newPosition);
  }, []);

  const handleZoom = useCallback((e: WheelEvent) => {
    const newScale = calculateZoomScale(e);
    updateViewScale(newScale);
  }, []);

  return {
    handleNodeClick,
    handleNodeHover,
    handlePan,
    handleZoom
  };
};`}
      />

      <h2>Animation System</h2>
      <p>
        Smooth animations enhance the user experience and provide visual feedback:
      </p>

      <CodeBlock
        language="typescript"
        code={`const useAnimations = () => {
  const [time, setTime] = useState(0);

  useAnimationFrame((deltaTime) => {
    setTime(prev => prev + deltaTime * 0.001);
  });

  const calculateNodeEffects = useCallback((node: Node) => {
    const wobble = Math.sin(time + node.id) * 0.3;
    const pulse = (1 + Math.sin(time * 2)) * 0.2;
    
    return {
      scale: node.activated ? 1 + pulse : 1,
      translate: node.available ? wobble : 0
    };
  }, [time]);

  return { calculateNodeEffects };
};`}
      />

      <h2>Visual Feedback</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Node States</h3>
          <ul>
            <li>Active: Solid fill</li>
            <li>Available: Pulsing outline</li>
            <li>Locked: Faded appearance</li>
            <li>Hover: Highlight effect</li>
            <li>Selected: Glow effect</li>
          </ul>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Connection States</h3>
          <ul>
            <li>Active: Solid line</li>
            <li>Available: Dashed line</li>
            <li>Highlighted: Glowing line</li>
            <li>Inactive: Faded line</li>
            <li>Loading: Animated dash</li>
          </ul>
        </div>
      </div>

      <h2>Accessibility</h2>
      <p>
        The network implements various accessibility features:
      </p>

      <ul>
        <li>Keyboard navigation</li>
        <li>Screen reader support</li>
        <li>ARIA labels and roles</li>
        <li>Focus management</li>
        <li>High contrast mode</li>
      </ul>

      <div className="bg-gray-800 rounded-lg p-6 my-8">
        <h3>Keyboard Controls</h3>
        <ul>
          <li><kbd>Tab</kbd> - Navigate between nodes</li>
          <li><kbd>Enter</kbd> - Activate node</li>
          <li><kbd>Space</kbd> - Show node info</li>
          <li><kbd>Arrow keys</kbd> - Pan view</li>
          <li><kbd>+/-</kbd> - Zoom in/out</li>
        </ul>
      </div>
    </div>
  );
}

export default InteractionSystem;