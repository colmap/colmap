import type {Camera, Vec2, Vec3} from "./types";

export const CAMERA_MODEL_NAMES = [
  "SIMPLE_PINHOLE",
  "PINHOLE",
  "SIMPLE_RADIAL",
  "RADIAL",
  "OPENCV",
  "OPENCV_FISHEYE",
  "FULL_OPENCV",
  "FOV",
  "SIMPLE_RADIAL_FISHEYE",
  "RADIAL_FISHEYE",
  "THIN_PRISM_FISHEYE",
  "RAD_TAN_THIN_PRISM_FISHEYE",
  "SIMPLE_DIVISION",
  "DIVISION",
  "SIMPLE_FISHEYE",
  "FISHEYE",
  "EUCM",
  "EQUIRECTANGULAR",
] as const;

export const CAMERA_MODEL_PARAM_COUNTS = [
  3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12, 16, 4, 5, 3, 4, 6, 2,
] as const;

function fisheyeFromNormal(u: number, v: number): Vec2 {
  const radius = Math.hypot(u, v);
  if (radius <= Number.EPSILON) return [u, v];
  const scale = Math.atan(radius) / radius;
  return [u * scale, v * scale];
}

function radial(u: number, v: number, coeffs: number[]): Vec2 {
  const r2 = u * u + v * v;
  let power = r2;
  let factor = 1;
  for (const coefficient of coeffs) {
    factor += coefficient * power;
    power *= r2;
  }
  return [u * factor, v * factor];
}

function pinhole(params: number[], uv: Vec2, sharedFocal: boolean): Vec2 {
  if (sharedFocal) return [params[0]! * uv[0] + params[1]!, params[0]! * uv[1] + params[2]!];
  return [params[0]! * uv[0] + params[2]!, params[1]! * uv[1] + params[3]!];
}

function opencvDistortion(u: number, v: number, p: number[]): Vec2 {
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0] = p;
  const u2 = u * u;
  const v2 = v * v;
  const uv = u * v;
  const r2 = u2 + v2;
  const factor = 1 + k1 * r2 + k2 * r2 * r2;
  return [
    u * factor + 2 * p1 * uv + p2 * (r2 + 2 * u2),
    v * factor + 2 * p2 * uv + p1 * (r2 + 2 * v2),
  ];
}

function fullOpenCVDistortion(u: number, v: number, p: number[]): Vec2 {
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0] = p;
  const u2 = u * u;
  const v2 = v * v;
  const uv = u * v;
  const r2 = u2 + v2;
  const r4 = r2 * r2;
  const r6 = r4 * r2;
  const factor = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6);
  return [
    u * factor + 2 * p1 * uv + p2 * (r2 + 2 * u2),
    v * factor + 2 * p2 * uv + p1 * (r2 + 2 * v2),
  ];
}

function thinPrismDistortion(u: number, v: number, p: number[]): Vec2 {
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0, k4 = 0, sx = 0, sy = 0] = p;
  const u2 = u * u;
  const v2 = v * v;
  const uv = u * v;
  const r2 = u2 + v2;
  const r4 = r2 * r2;
  const factor = 1 + k1 * r2 + k2 * r4 + k3 * r4 * r2 + k4 * r4 * r4;
  return [
    u * factor + 2 * p1 * uv + p2 * (r2 + 2 * u2) + sx * r2,
    v * factor + 2 * p2 * uv + p1 * (r2 + 2 * v2) + sy * r2,
  ];
}

function radTanThinPrism(u: number, v: number, p: number[]): Vec2 {
  const r2Theta = u * u + v * v;
  let thetaPower = r2Theta;
  let thetaFactor = 1;
  for (let i = 0; i < 6; ++i) {
    thetaFactor += (p[i] ?? 0) * thetaPower;
    thetaPower *= r2Theta;
  }
  const x = u * thetaFactor;
  const y = v * thetaFactor;
  const x2 = x * x;
  const y2 = y * y;
  const xy = x * y;
  const r2 = x2 + y2;
  const r4 = r2 * r2;
  const p0 = p[6] ?? 0;
  const p1 = p[7] ?? 0;
  return [
    x + 2 * p1 * xy + p0 * (r2 + 2 * x2) + (p[8] ?? 0) * r2 + (p[9] ?? 0) * r4,
    y + 2 * p0 * xy + p1 * (r2 + 2 * y2) + (p[10] ?? 0) * r2 + (p[11] ?? 0) * r4,
  ];
}

export function project(camera: Camera, pointInCamera: Vec3): Vec2 | null {
  const [u, v, w] = pointInCamera;
  const p = camera.params;
  const model = camera.modelId;

  if (model === 17) {
    const horizontal = Math.hypot(u, w);
    if (horizontal + Math.abs(v) < Number.EPSILON) return null;
    const theta = Math.atan2(u, w);
    const phi = Math.atan2(-v, horizontal);
    return [(theta / (2 * Math.PI) + 0.5) * p[0]!, (0.5 - phi / Math.PI) * p[1]!];
  }
  if (w < Number.EPSILON) return null;

  if (model === 12 || model === 13) {
    const k = p[model === 12 ? 3 : 4]!;
    const discriminant = w * w - 4 * (u * u + v * v) * k;
    if (discriminant < 0) return null;
    const scale = 2 / (w + Math.sqrt(discriminant));
    return model === 12
      ? [p[0]! * scale * u + p[1]!, p[0]! * scale * v + p[2]!]
      : [p[0]! * scale * u + p[2]!, p[1]! * scale * v + p[3]!];
  }

  if (model === 16) {
    const rho2 = p[5]! * (u * u + v * v) + w * w;
    if (rho2 < 0) return null;
    const denominator = p[4]! * Math.sqrt(rho2) + (1 - p[4]!) * w;
    if (denominator < Number.EPSILON) return null;
    return [p[0]! * u / denominator + p[2]!, p[1]! * v / denominator + p[3]!];
  }

  if (model === 0) return pinhole(p, [u / w, v / w], true);
  if (model === 1) return pinhole(p, [u / w, v / w], false);

  if (model === 7) {
    const uu = u / w;
    const vv = v / w;
    const radius2 = uu * uu + vv * vv;
    const omega = p[4]!;
    const omega2 = omega * omega;
    let factor: number;
    if (omega2 < 1e-4) factor = omega2 * radius2 / 3 - omega2 / 12 + 1;
    else if (radius2 < 1e-4) {
      const tangent = Math.tan(omega / 2);
      factor = -2 * tangent * (4 * radius2 * tangent * tangent - 3) / (3 * omega);
    } else factor = Math.atan(Math.sqrt(radius2) * 2 * Math.tan(omega / 2)) / (Math.sqrt(radius2) * omega);
    return pinhole(p, [factor * uu, factor * vv], false);
  }

  const fisheye = [5, 8, 9, 10, 11, 14, 15].includes(model);
  let normalized: Vec2 = fisheye ? fisheyeFromNormal(u / w, v / w) : [u / w, v / w];
  switch (model) {
    case 2: normalized = radial(normalized[0], normalized[1], [p[3]!]); break;
    case 3: normalized = radial(normalized[0], normalized[1], p.slice(3, 5)); break;
    case 4: normalized = opencvDistortion(normalized[0], normalized[1], p.slice(4)); break;
    case 5: normalized = radial(normalized[0], normalized[1], p.slice(4, 8)); break;
    case 6: normalized = fullOpenCVDistortion(normalized[0], normalized[1], p.slice(4)); break;
    case 8: normalized = radial(normalized[0], normalized[1], [p[3]!]); break;
    case 9: normalized = radial(normalized[0], normalized[1], p.slice(3, 5)); break;
    case 10: normalized = thinPrismDistortion(normalized[0], normalized[1], p.slice(4)); break;
    case 11: normalized = radTanThinPrism(normalized[0], normalized[1], p.slice(4)); break;
  }
  return [0, 2, 3, 8, 9, 14].includes(model)
    ? pinhole(p, normalized, true)
    : pinhole(p, normalized, false);
}
