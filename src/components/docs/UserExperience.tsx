import React from 'react';
import CodeBlock from '../CodeBlock';

function UserExperience() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1>User Experience</h1>

      <h2>Interaction Design</h2>
      <p>
        The network provides intuitive interactions through carefully designed gesture and event handling:
      </p>

      <CodeBlock
        language="typescript"
        code={`const useGestureHandling = () => {
  const [gesture, setGesture] = useState<GestureState>({
    isDragging: false,
    isPinching: false,
    startPoint: null,
    scale: 1
  });

  const handleTouchStart = useCallback((e: TouchEvent) => {
    if (e.touches.length === 2) {
      const distance = getTouchDistance(e.touches);
      setGesture(prev => ({
        ...prev,
        isPinching: true,
        initialDistance: distance
      }));
    }
  }, []);

  const handleTouchMove = useCallback((e: TouchEvent) => {
    if (gesture.isPinching) {
      const newDistance = getTouchDistance(e.touches);
      const scale = newDistance / gesture.initialDistance;
      updateScale(scale);
    }
  }, [gesture]);

  return {
    handleTouchStart,
    handleTouchMove,
    handleTouchEnd: () => setGesture(initialState)
  };
};`}
      />

      <h2>Visual Feedback</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Interactive States</h3>
          <ul>
            <li>Hover effects</li>
            <li>Active states</li>
            <li>Loading indicators</li>
            <li>Progress feedback</li>
            <li>Error states</li>
          </ul>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Motion Design</h3>
          <ul>
            <li>Smooth transitions</li>
            <li>Micro-animations</li>
            <li>Loading animations</li>
            <li>State transitions</li>
            <li>Gesture feedback</li>
          </ul>
        </div>
      </div>

      <h2>Accessibility</h2>
      <p>
        The network implements comprehensive accessibility features:
      </p>

      <CodeBlock
        language="typescript"
        code={`const useAccessibility = () => {
  const [focusedNode, setFocusedNode] = useState<string | null>(null);
  const [announcements, setAnnouncements] = useState<string[]>([]);

  const handleKeyNavigation = useCallback((e: KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowRight':
        focusNextNode('horizontal');
        break;
      case 'ArrowDown':
        focusNextNode('vertical');
        break;
      case 'Enter':
        activateNode(focusedNode);
        break;
    }
  }, [focusedNode]);

  const announce = useCallback((message: string) => {
    setAnnouncements(prev => [...prev, message]);
  }, []);

  return {
    focusedNode,
    handleKeyNavigation,
    announcements,
    announce
  };
};`}
      />

      <h2>Responsive Design</h2>
      <p>
        The interface adapts seamlessly to different screen sizes and devices:
      </p>

      <CodeBlock
        language="typescript"
        code={`const useResponsiveLayout = () => {
  const [viewport, setViewport] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
    deviceType: getDeviceType()
  });

  const calculateLayout = useCallback(() => {
    const baseRadius = Math.min(viewport.width, viewport.height) * 0.4;
    
    return {
      centerPoint: {
        x: viewport.width / 2,
        y: viewport.height / 2
      },
      layerRadii: {
        primary: baseRadius * 0.5,
        secondary: baseRadius * 0.75,
        tertiary: baseRadius
      }
    };
  }, [viewport]);

  return {
    viewport,
    layout: calculateLayout(),
    isCompact: viewport.width < 768
  };
};`}
      />

      <h2>Performance Optimization</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Rendering Optimization</h3>
          <ul>
            <li>Component memoization</li>
            <li>Virtual scrolling</li>
            <li>Lazy loading</li>
            <li>Debounced updates</li>
            <li>RAF scheduling</li>
          </ul>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Animation Performance</h3>
          <ul>
            <li>Hardware acceleration</li>
            <li>Transform compositing</li>
            <li>Animation throttling</li>
            <li>Batch updates</li>
            <li>GPU utilization</li>
          </ul>
        </div>
      </div>

      <h2>Error Handling</h2>
      <p>
        The system provides graceful error handling and recovery:
      </p>

      <CodeBlock
        language="typescript"
        code={`const useErrorHandling = () => {
  const [errors, setErrors] = useState<Map<string, Error>>(new Map());

  const handleError = useCallback((id: string, error: Error) => {
    setErrors(prev => new Map(prev).set(id, error));
    
    // Show user-friendly error message
    showNotification({
      type: 'error',
      message: getErrorMessage(error),
      action: {
        label: 'Retry',
        handler: () => retryOperation(id)
      }
    });
  }, []);

  const clearError = useCallback((id: string) => {
    setErrors(prev => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);

  return {
    errors,
    handleError,
    clearError,
    hasErrors: errors.size > 0
  };
};`}
      />

      <h2>Loading States</h2>
      <p>
        The network provides engaging loading states and transitions:
      </p>

      <CodeBlock
        language="typescript"
        code={`const useLoadingStates = () => {
  const [loading, setLoading] = useState<LoadingState>({
    type: 'initial',
    progress: 0,
    message: 'Loading network...'
  });

  const updateProgress = useCallback((progress: number) => {
    setLoading(prev => ({
      ...prev,
      progress,
      message: getProgressMessage(progress)
    }));
  }, []);

  const transitionState = useCallback((type: LoadingType) => {
    setLoading(prev => ({
      ...prev,
      type,
      message: getStateMessage(type)
    }));
  }, []);

  return {
    loading,
    updateProgress,
    transitionState
  };
};`}
      />

      <h2>User Preferences</h2>
      <p>
        The system respects and adapts to user preferences:
      </p>

      <div className="bg-gray-800 rounded-lg p-6 my-8">
        <h3>Customization Options</h3>
        <ul>
          <li>Color schemes and themes</li>
          <li>Animation preferences</li>
          <li>Interaction modes</li>
          <li>Layout options</li>
          <li>Accessibility settings</li>
        </ul>
      </div>
    </div>
  );
}

export default UserExperience;