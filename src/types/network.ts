export interface Node {
  id: string;
  x: number;
  y: number;
  type: 'center' | 'primary' | 'secondary' | 'tertiary' | 'quaternary';
  activated?: boolean;
  available?: boolean;
}

export interface Connection {
  id: string;
  sourceId: string;
  targetId: string;
  strength: number;
}