export const PHYSICS = {
  SPRING: {
    STIFFNESS: 0.6,     // Reduced for more rope-like behavior
    DAMPING: 0.65,      // Increased for more natural movement
    REST_LENGTH: 100,
    MAX_LENGTH: 200,
  },
  
  FORCE: {
    ATTRACTION: 1.5,    // Adjusted for better balance
    REPULSION: 2500,    // Adjusted for spacing
    DRAG: 0.95,        // Increased for smoother movement
    RETURN: 0.12,      // Reduced for more flexibility
  },
  
  VELOCITY: {
    MIN: 0.001,
    MAX: 15,           // Reduced for more controlled movement
    DAMPING: 0.92,     // Adjusted damping
  },
  
  DISTANCE: {
    INFLUENCE_MIN: 1,
    INFLUENCE_MAX: 350,
    CONNECTION: 150,
  },
  
  ANIMATION: {
    FRAME_CAP: 16.67,
    MAX_DELTA: 32,
    SUBSTEPS: 3,
  },
  
  INTERACTION: {
    DRAG_STRENGTH: 0.8,
    ELASTIC_STRENGTH: 0.5,
    NEIGHBOR_INFLUENCE: 0.7,
    STABILITY_THRESHOLD: 0.01,
  }
} as const;