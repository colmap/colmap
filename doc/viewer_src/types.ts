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

export const INVALID_POINT3D_ID = 0xffffffffffffffffn;

// Observations are flattened so their buffers can be transferred from the parser worker.
export interface Point2DData {
  // [x0, y0, x1, y1, ...]
  xy: Float64Array;
  point3DIds: BigUint64Array;
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
  points2D: Point2DData;
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

export interface Point3DData {
  ids: BigUint64Array;
  xyz: Float64Array;
  colors: Uint8Array;
  errors: Float32Array;
  // Offsets delimit each point's range in the two parallel track arrays.
  trackOffsets: Uint32Array;
  trackImageIds: Uint32Array;
  trackPoint2DIdxs: Uint32Array;
}

export interface Reconstruction {
  cameras: Map<number, Camera>;
  images: Map<number, ImageRecord>;
  points3D: Point3DData;
  modernRigFormat: boolean;
}

export function point2DCount(image: ImageRecord): number {
  return image.points2D.point3DIds.length;
}

export function point2DAt(image: ImageRecord, index: number): Point2D | undefined {
  if (index < 0 || index >= point2DCount(image)) return undefined;
  const point3DId = image.points2D.point3DIds[index]!;
  return {
    xy: [image.points2D.xy[index * 2]!, image.points2D.xy[index * 2 + 1]!],
    point3DId: point3DId === INVALID_POINT3D_ID ? null : point3DId,
  };
}

export function point3DCount(points: Point3DData): number {
  return points.ids.length;
}

export function point3DTrackLength(points: Point3DData, index: number): number {
  return points.trackOffsets[index + 1]! - points.trackOffsets[index]!;
}

export function point3DAt(points: Point3DData, index: number): Point3D {
  if (index < 0 || index >= point3DCount(points)) throw new RangeError(`Invalid 3D point index ${index}`);
  const track = Array.from(
    {length: point3DTrackLength(points, index)},
    (_, offset) => {
      const trackIndex = points.trackOffsets[index]! + offset;
      return {
        imageId: points.trackImageIds[trackIndex]!,
        point2DIdx: points.trackPoint2DIdxs[trackIndex]!,
      };
    },
  );
  return {
    id: points.ids[index]!,
    xyz: [points.xyz[index * 3]!, points.xyz[index * 3 + 1]!, points.xyz[index * 3 + 2]!],
    color: [points.colors[index * 3]!, points.colors[index * 3 + 1]!, points.colors[index * 3 + 2]!],
    error: points.errors[index]!,
    track,
  };
}

export function findPoint3DIndex(points: Point3DData, id: bigint): number {
  let lower = 0;
  let upper = points.ids.length;
  while (lower < upper) {
    const middle = lower + Math.floor((upper - lower) / 2);
    const middleId = points.ids[middle]!;
    if (middleId < id) lower = middle + 1;
    else upper = middle;
  }
  return lower < points.ids.length && points.ids[lower] === id ? lower : -1;
}

export interface LocalFile {
  path: string;
  file: File;
}

export interface SparseModelCandidate {
  path: string;
  files: Map<string, File>;
}
