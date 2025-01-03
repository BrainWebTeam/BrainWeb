interface Point {
  x: number;
  y: number;
}

export const generateCircularPoints = (
  center: Point,
  radius: number,
  count: number
): Point[] => {
  return Array.from({ length: count }, (_, i) => {
    const angle = (i * 2 * Math.PI) / count;
    return {
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius
    };
  });
};

export const generateOrbitPoints = (
  center: Point,
  radius: number,
  count: number,
  time: number
): Point[] => {
  return Array.from({ length: count }, (_, i) => {
    const angle = (i * 2 * Math.PI) / count + time;
    return {
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius
    };
  });
};

export const generateSpiral = (
  center: Point,
  startRadius: number,
  endRadius: number,
  revolutions: number,
  time: number
): Point[] => {
  const points: Point[] = [];
  const steps = 50;
  
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1);
    const angle = 2 * Math.PI * revolutions * t + time;
    const radius = startRadius + (endRadius - startRadius) * t;
    points.push({
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius
    });
  }
  
  return points;
};

export const generateRandomPoints = (
  count: number,
  bounds: { width: number; height: number }
): Point[] => {
  return Array.from({ length: count }, () => ({
    x: Math.random() * bounds.width,
    y: Math.random() * bounds.height
  }));
};