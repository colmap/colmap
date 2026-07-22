import * as THREE from "three";
import {expect, test} from "vitest";

import {CAMERA_FRUSTUM_COLORS, COORDINATE_COLORS, photometricPointColor, scaleFromNativeWheel} from "../viewer_src/viewer";

test("matches native COLMAP camera frustum colors", () => {
  expect(CAMERA_FRUSTUM_COLORS).toEqual({
    frame: [0.8, 0.1, 0, 1],
    plane: [1, 0.1, 0, 0.6],
    selectedFrame: [0.8, 0, 0.8, 1],
    selectedPlane: [1, 0, 1, 0.6],
    sameFrame: [0.6, 0, 0.6, 179 / 255],
    sameFramePlane: [0.8, 0, 0.8, 77 / 255],
  });
});

test("matches native COLMAP coordinate overlay colors", () => {
  expect(COORDINATE_COLORS).toEqual({
    grid: [51 / 255, 51 / 255, 51 / 255, 153 / 255],
    x: [230 / 255, 0, 0, 128 / 255],
    y: [0, 230 / 255, 0, 128 / 255],
    z: [0, 0, 230 / 255, 128 / 255],
  });
});

test("matches native modifier-wheel scaling and limits", () => {
  expect(scaleFromNativeWheel(1, -120, 0, 0.5, 100)).toBeCloseTo(1.12);
  expect(scaleFromNativeWheel(1, 120, 0, 0.5, 100)).toBeCloseTo(0.88);
  expect(scaleFromNativeWheel(0.5, 1000, 0, 0.5, 100)).toBe(0.5);
  expect(scaleFromNativeWheel(100, -1000, 0, 0.5, 100)).toBe(100);
});

test("preserves COLMAP photometric RGB values through Three.js color management", () => {
  expect(photometricPointColor([115, 121, 122]).getHex(THREE.SRGBColorSpace)).toBe(0x73797a);
});
