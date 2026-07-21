export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // w, x, y, z

export interface Rigid3d {
  rotation: Quat;
  translation: Vec3;
}

export interface Camera {
  id: number;
  modelId: number;
  width: number;
  height: number;
  params: number[];
}

export interface Point2D {
  xy: Vec2;
  point3DId: bigint | null;
}

export interface ImageRecord {
  id: number;
  cameraId: number;
  frameId: number;
  rigId: number;
  name: string;
  camFromWorld: Rigid3d;
  points2D: Point2D[];
}

export interface TrackElement {
  imageId: number;
  point2DIdx: number;
}

export interface Point3D {
  id: bigint;
  xyz: Vec3;
  color: [number, number, number];
  error: number;
  track: TrackElement[];
}

export interface Reconstruction {
  cameras: Map<number, Camera>;
  images: Map<number, ImageRecord>;
  points3D: Map<bigint, Point3D>;
  modernRigFormat: boolean;
}

export interface LocalFile {
  path: string;
  file: File;
}

export interface SparseModelCandidate {
  path: string;
  files: Map<string, File>;
}
