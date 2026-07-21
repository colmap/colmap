import {CAMERA_MODEL_PARAM_COUNTS} from "./camera_models";
import {composeRigid} from "./math";
import type {
  Camera,
  ImageRecord,
  LocalFile,
  Point3D,
  Quat,
  Reconstruction,
  Rigid3d,
  SparseModelCandidate,
  Vec3,
} from "./types";

const INVALID_POINT3D_ID = 0xffffffffffffffffn;

class BinaryReader {
  private readonly view: DataView;
  private offset = 0;

  constructor(buffer: ArrayBuffer, private readonly label: string) {
    this.view = new DataView(buffer);
  }

  private require(bytes: number): void {
    if (bytes < 0 || this.offset + bytes > this.view.byteLength) {
      throw new Error(`${this.label} is truncated at byte ${this.offset}`);
    }
  }

  u8(): number { this.require(1); return this.view.getUint8(this.offset++); }
  i32(): number { this.require(4); const value = this.view.getInt32(this.offset, true); this.offset += 4; return value; }
  u32(): number { this.require(4); const value = this.view.getUint32(this.offset, true); this.offset += 4; return value; }
  u64(): bigint { this.require(8); const value = this.view.getBigUint64(this.offset, true); this.offset += 8; return value; }
  f64(): number { this.require(8); const value = this.view.getFloat64(this.offset, true); this.offset += 8; return value; }

  count(name: string, minimumBytesPerItem = 1): number {
    const value = this.u64();
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${this.label} has an invalid ${name}`);
    const count = Number(value);
    if (count > Math.floor(this.remaining / minimumBytesPerItem)) throw new Error(`${this.label} has an invalid ${name}`);
    return count;
  }

  string(): string {
    const bytes: number[] = [];
    while (true) {
      const byte = this.u8();
      if (byte === 0) break;
      bytes.push(byte);
    }
    return new TextDecoder("utf-8", {fatal: true}).decode(new Uint8Array(bytes));
  }

  get remaining(): number { return this.view.byteLength - this.offset; }
}

interface Rig {
  id: number;
  refSensor: string;
  sensors: Map<string, Rigid3d | null>;
}

interface Frame {
  id: number;
  rigId: number;
  rigFromWorld: Rigid3d;
  data: Array<{sensor: string; dataId: bigint}>;
}

function sensorKey(type: number, id: number): string { return `${type}:${id}`; }
function readQuat(reader: BinaryReader): Quat { return [reader.f64(), reader.f64(), reader.f64(), reader.f64()]; }
function readVec3(reader: BinaryReader): Vec3 { return [reader.f64(), reader.f64(), reader.f64()]; }
function readRigid(reader: BinaryReader): Rigid3d {
  const transform = {rotation: readQuat(reader), translation: readVec3(reader)};
  if (![...transform.rotation, ...transform.translation].every(Number.isFinite)) throw new Error("Reconstruction contains a non-finite pose");
  return transform;
}

function parseCameras(buffer: ArrayBuffer): Map<number, Camera> {
  const reader = new BinaryReader(buffer, "cameras.bin");
  const count = reader.count("camera count", 24);
  const cameras = new Map<number, Camera>();
  for (let i = 0; i < count; ++i) {
    const id = reader.u32();
    const modelId = reader.i32();
    const paramCount = CAMERA_MODEL_PARAM_COUNTS[modelId];
    if (paramCount === undefined) throw new Error(`Unsupported camera model id ${modelId}`);
    const widthBig = reader.u64();
    const heightBig = reader.u64();
    if (widthBig > BigInt(Number.MAX_SAFE_INTEGER) || heightBig > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`Camera ${id} has invalid dimensions`);
    const params = Array.from({length: paramCount}, () => reader.f64());
    if (widthBig === 0n || heightBig === 0n || !params.every(Number.isFinite)) throw new Error(`Camera ${id} has invalid parameters`);
    cameras.set(id, {id, modelId, width: Number(widthBig), height: Number(heightBig), params});
  }
  if (reader.remaining !== 0) throw new Error("cameras.bin contains trailing data");
  return cameras;
}

function parseRigs(buffer: ArrayBuffer): Map<number, Rig> {
  const reader = new BinaryReader(buffer, "rigs.bin");
  const count = reader.count("rig count", 8);
  const rigs = new Map<number, Rig>();
  for (let i = 0; i < count; ++i) {
    const id = reader.u32();
    const sensorCount = reader.u32();
    if (sensorCount > reader.remaining / 8) throw new Error("rigs.bin has an invalid sensor count");
    let refSensor = "";
    const sensors = new Map<string, Rigid3d | null>();
    if (sensorCount > 0) {
      refSensor = sensorKey(reader.i32(), reader.u32());
      sensors.set(refSensor, {rotation: [1, 0, 0, 0], translation: [0, 0, 0]});
    }
    for (let j = 1; j < sensorCount; ++j) {
      const sensor = sensorKey(reader.i32(), reader.u32());
      sensors.set(sensor, reader.u8() ? readRigid(reader) : null);
    }
    rigs.set(id, {id, refSensor, sensors});
  }
  if (reader.remaining !== 0) throw new Error("rigs.bin contains trailing data");
  return rigs;
}

function parseFrames(buffer: ArrayBuffer): Map<number, Frame> {
  const reader = new BinaryReader(buffer, "frames.bin");
  const count = reader.count("frame count", 68);
  const frames = new Map<number, Frame>();
  for (let i = 0; i < count; ++i) {
    const id = reader.u32();
    const rigId = reader.u32();
    const rigFromWorld = readRigid(reader);
    const dataCount = reader.u32();
    if (dataCount > reader.remaining / 16) throw new Error("frames.bin has an invalid data count");
    const data = Array.from({length: dataCount}, () => ({
      sensor: sensorKey(reader.i32(), reader.u32()),
      dataId: reader.u64(),
    }));
    frames.set(id, {id, rigId, rigFromWorld, data});
  }
  if (reader.remaining !== 0) throw new Error("frames.bin contains trailing data");
  return frames;
}

function parseImages(
  buffer: ArrayBuffer,
  rigs: Map<number, Rig> | null,
  frames: Map<number, Frame> | null,
): Map<number, ImageRecord> {
  const reader = new BinaryReader(buffer, "images.bin");
  const count = reader.count("image count", 72);
  const images = new Map<number, ImageRecord>();
  const imageToFrame = new Map<bigint, Frame>();
  if (frames) for (const frame of frames.values()) for (const data of frame.data) if (data.sensor.startsWith("0:")) imageToFrame.set(data.dataId, frame);

  for (let i = 0; i < count; ++i) {
    const id = reader.u32();
    const serializedPose = readRigid(reader);
    const cameraId = reader.u32();
    const name = reader.string();
    const pointCount = reader.count("point2D count", 24);
    const points2D = Array.from({length: pointCount}, () => {
      const xy: [number, number] = [reader.f64(), reader.f64()];
      if (!xy.every(Number.isFinite)) throw new Error(`Image ${id} has a non-finite observation`);
      const point3DId = reader.u64();
      return {xy, point3DId: point3DId === INVALID_POINT3D_ID ? null : point3DId};
    });

    let camFromWorld = serializedPose;
    let frameId = id;
    let rigId = cameraId;
    if (rigs && frames) {
      const frame = imageToFrame.get(BigInt(id));
      if (!frame) throw new Error(`No frame contains image ${id}`);
      const rig = rigs.get(frame.rigId);
      if (!rig) throw new Error(`Frame ${frame.id} references missing rig ${frame.rigId}`);
      const sensorFromRig = rig.sensors.get(sensorKey(0, cameraId));
      if (sensorFromRig === undefined) throw new Error(`Rig ${rig.id} does not contain camera ${cameraId}`);
      if (sensorFromRig === null) throw new Error(`Rig ${rig.id} has no pose for camera ${cameraId}`);
      camFromWorld = composeRigid(sensorFromRig, frame.rigFromWorld);
      frameId = frame.id;
      rigId = rig.id;
    }
    images.set(id, {id, cameraId, frameId, rigId, name, camFromWorld, points2D});
  }
  if (reader.remaining !== 0) throw new Error("images.bin contains trailing data");
  return images;
}

function parsePoints3D(buffer: ArrayBuffer): Map<bigint, Point3D> {
  const reader = new BinaryReader(buffer, "points3D.bin");
  const count = reader.count("point3D count", 51);
  const points = new Map<bigint, Point3D>();
  for (let i = 0; i < count; ++i) {
    const id = reader.u64();
    const xyz = readVec3(reader);
    const color: [number, number, number] = [reader.u8(), reader.u8(), reader.u8()];
    const error = reader.f64();
    if (![...xyz, error].every(Number.isFinite)) throw new Error(`Point ${id} contains non-finite values`);
    const trackLength = reader.count("track length", 8);
    const track = Array.from({length: trackLength}, () => ({imageId: reader.u32(), point2DIdx: reader.u32()}));
    points.set(id, {id, xyz, color, error, track});
  }
  if (reader.remaining !== 0) throw new Error("points3D.bin contains trailing data");
  return points;
}

export function normalizePath(path: string): string {
  return path.replaceAll("\\", "/").replace(/^\.\//, "").replace(/\/+/g, "/");
}

export function discoverSparseModels(entries: LocalFile[]): SparseModelCandidate[] {
  const groups = new Map<string, Map<string, File>>();
  for (const entry of entries) {
    const path = normalizePath(entry.path);
    const slash = path.lastIndexOf("/");
    const directory = slash < 0 ? "." : path.slice(0, slash);
    const basename = path.slice(slash + 1);
    const files = groups.get(directory) ?? new Map<string, File>();
    files.set(basename, entry.file);
    groups.set(directory, files);
  }
  return [...groups]
    .filter(([, files]) => ["cameras.bin", "images.bin", "points3D.bin"].every((name) => files.has(name)))
    .map(([path, files]) => ({path, files}))
    .sort((a, b) => a.path.localeCompare(b.path));
}

export async function parseReconstruction(files: Map<string, File>): Promise<Reconstruction> {
  const required = ["cameras.bin", "images.bin", "points3D.bin"];
  for (const name of required) if (!files.has(name)) throw new Error(`Missing ${name}`);
  const hasRigs = files.has("rigs.bin");
  const hasFrames = files.has("frames.bin");
  if (hasRigs !== hasFrames) throw new Error("Modern reconstructions require both rigs.bin and frames.bin");
  const buffers = await Promise.all([
    files.get("cameras.bin")!.arrayBuffer(),
    files.get("images.bin")!.arrayBuffer(),
    files.get("points3D.bin")!.arrayBuffer(),
    hasRigs ? files.get("rigs.bin")!.arrayBuffer() : Promise.resolve(null),
    hasFrames ? files.get("frames.bin")!.arrayBuffer() : Promise.resolve(null),
  ]);
  const parsedRigs = buffers[3] ? parseRigs(buffers[3]) : null;
  const parsedFrames = buffers[4] ? parseFrames(buffers[4]) : null;
  const modernRigFormat = Boolean(parsedRigs && parsedFrames && (parsedRigs.size > 0 || parsedFrames.size > 0));
  const rigs = modernRigFormat ? parsedRigs : null;
  const frames = modernRigFormat ? parsedFrames : null;
  const cameras = parseCameras(buffers[0]);
  const images = parseImages(buffers[1], rigs, frames);
  for (const image of images.values()) if (!cameras.has(image.cameraId)) throw new Error(`Image ${image.id} references missing camera ${image.cameraId}`);
  return {cameras, images, points3D: parsePoints3D(buffers[2]), modernRigFormat};
}
