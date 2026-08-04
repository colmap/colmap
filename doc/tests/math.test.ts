import {expect, test} from "vitest";

import {median, percentile} from "../viewer_src/math";

test("matches COLMAP percentile interpolation and median", () => {
  expect(percentile([0, 100], 0.01)).toBe(1);
  expect(percentile([0, 1, 2, 3], 0.34)).toBeCloseTo(1.02);
  expect(median([4, 1, 3, 2])).toBe(2.5);
});
