import type {Quat, Rigid3d, Vec3} from "./types";

export function quatMultiply(a: Quat, b: Quat): Quat {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

export function quatRotate(q: Quat, v: Vec3): Vec3 {
  const [w, x, y, z] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + y * tz - z * ty,
    v[1] + w * ty + z * tx - x * tz,
    v[2] + w * tz + x * ty - y * tx,
  ];
}

export function composeRigid(a: Rigid3d, b: Rigid3d): Rigid3d {
  const t = quatRotate(a.rotation, b.translation);
  return {
    rotation: quatMultiply(a.rotation, b.rotation),
    translation: [
      t[0] + a.translation[0],
      t[1] + a.translation[1],
      t[2] + a.translation[2],
    ],
  };
}

export function transformPoint(transform: Rigid3d, point: Vec3): Vec3 {
  const rotated = quatRotate(transform.rotation, point);
  return [
    rotated[0] + transform.translation[0],
    rotated[1] + transform.translation[1],
    rotated[2] + transform.translation[2],
  ];
}

export function projectionCenter(camFromWorld: Rigid3d): Vec3 {
  const [w, x, y, z] = camFromWorld.rotation;
  const rotated = quatRotate(
    [w, -x, -y, -z],
    camFromWorld.translation,
  );
  return [-rotated[0], -rotated[1], -rotated[2]];
}

export function percentile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, q * (sorted.length - 1)));
  const left = Math.floor(index);
  const right = Math.ceil(index);
  if (left === right) return sorted[left] ?? 0;
  return (sorted[left] ?? 0) * (right - index) + (sorted[right] ?? 0) * (index - left);
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? (sorted[middle] ?? 0)
    : ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}
