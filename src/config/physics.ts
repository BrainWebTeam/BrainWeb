export const PHYSICS_CONFIG = {
  // Spring physics
  SPRING_DAMPING: 0.92, // Smoother damping
  SPRING_STIFFNESS: 0.03, // Softer springs for more elasticity
  
  // Node connections
  CONNECTION_STRENGTH: 0.3, // Stronger node influence
  CONNECTION_RADIUS: 250, // Larger influence radius
  
  // Movement
  RETURN_FORCE: 0.04, // Gentler return force
  RETURN_DAMPING: 0.95, // Smoother movement
  
  // Animation
  MIN_VELOCITY: 0.01,
  MAX_VELOCITY: 20, // Higher max velocity for snappier movement
  MAX_DISPLACEMENT: 400,
  
  // Interaction
  DRAG_INFLUENCE: 0.7, // Stronger dragging effect
  WOBBLE_AMOUNT: 0.3, // More noticeable wobble
  WOBBLE_SPEED: 1.5, // Slower wobble for more natural movement
  
  // Constraints
  MIN_DISTANCE: 1,
  POSITION_THRESHOLD: 0.1
};