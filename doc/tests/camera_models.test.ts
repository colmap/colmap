import {describe, expect, test} from "vitest";

import {CAMERA_MODEL_PARAM_COUNTS, project} from "../viewer_src/camera_models";
import type {Camera} from "../viewer_src/types";

function camera(modelId: number, params: number[]): Camera {
  return {id: 1, modelId, width: 640, height: 480, params};
}

describe("camera projections", () => {
  test("matches hand-computed pinhole and radial projections", () => {
    expect(project(camera(0, [100, 320, 240]), [0.1, -0.2, 1])).toEqual([330, 220]);
    expect(project(camera(1, [100, 120, 320, 240]), [0.1, -0.2, 1])).toEqual([330, 216]);
    const radial = project(camera(2, [100, 320, 240, 0.1]), [0.1, -0.2, 1]);
    expect(radial?.[0]).toBeCloseTo(330.05, 10);
    expect(radial?.[1]).toBeCloseTo(219.9, 10);
  });

  test("projects division, EUCM, and full-sphere equirectangular cameras", () => {
    expect(project(camera(12, [100, 320, 240, 0]), [0.1, -0.2, 1])).toEqual([330, 220]);
    expect(project(camera(16, [100, 100, 320, 240, 0, 1]), [0.1, -0.2, 1])).toEqual([330, 220]);
    expect(project(camera(17, [640, 320]), [0, 0, -1])).toEqual([640, 160]);
  });

  test("has a valid projection implementation for every current model", () => {
    const params = [
      [100, 320, 240],
      [100, 100, 320, 240],
      [100, 320, 240, 0],
      [100, 320, 240, 0, 0],
      [100, 100, 320, 240, 0, 0, 0, 0],
      [100, 100, 320, 240, 0, 0, 0, 0],
      [100, 100, 320, 240, 0, 0, 0, 0, 0, 0, 0, 0],
      [100, 100, 320, 240, 0.5],
      [100, 320, 240, 0],
      [100, 320, 240, 0, 0],
      [100, 100, 320, 240, 0, 0, 0, 0, 0, 0, 0, 0],
      [100, 100, 320, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [100, 320, 240, 0],
      [100, 100, 320, 240, 0],
      [100, 320, 240],
      [100, 100, 320, 240],
      [100, 100, 320, 240, 0, 1],
      [640, 320],
    ];
    expect(params.map((value) => value.length)).toEqual([...CAMERA_MODEL_PARAM_COUNTS]);
    for (let modelId = 0; modelId < params.length; ++modelId) {
      const result = project(camera(modelId, params[modelId]!), [0.1, -0.2, 1]);
      expect(result, `model ${modelId}`).not.toBeNull();
      expect(result?.every(Number.isFinite), `model ${modelId}`).toBe(true);
    }
  });
});
