import {describe, expect, test} from "vitest";

import {discoverSparseModels, parseReconstruction} from "../viewer_src/parser";
import {cameraFile, imageFile, modernRigFiles, pointFile} from "./binary_fixture";

function modelFiles(cameraId = 1): Map<string, File> {
  return new Map([
    ["cameras.bin", cameraFile(cameraId)],
    ["images.bin", imageFile(2, cameraId)],
    ["points3D.bin", pointFile()],
  ]);
}

describe("binary reconstruction parser", () => {
  test("parses a legacy model without losing uint64 point ids", async () => {
    const pointId = 9007199254740993n;
    const files = modelFiles();
    files.set("images.bin", imageFile(2, 1, pointId));
    files.set("points3D.bin", pointFile(2, pointId));
    const reconstruction = await parseReconstruction(files);
    expect(reconstruction.modernRigFormat).toBe(false);
    expect(reconstruction.cameras.get(1)?.params).toEqual([500, 320, 240]);
    expect(reconstruction.images.get(2)?.camFromWorld.translation).toEqual([4, 5, 6]);
    expect(reconstruction.images.get(2)?.points2D[0]?.point3DId).toBe(pointId);
    expect(reconstruction.points3D.get(pointId)?.track).toEqual([{imageId: 2, point2DIdx: 0}]);
  });

  test("composes modern sensor and rig poses", async () => {
    const files = modelFiles(2);
    const [rigs, frames] = modernRigFiles();
    files.set("rigs.bin", rigs);
    files.set("frames.bin", frames);
    const reconstruction = await parseReconstruction(files);
    expect(reconstruction.modernRigFormat).toBe(true);
    expect(reconstruction.images.get(2)?.camFromWorld.translation).toEqual([1, 2, 0]);
    expect(reconstruction.images.get(2)?.frameId).toBe(9);
    expect(reconstruction.images.get(2)?.rigId).toBe(7);
  });

  test("rejects incomplete and truncated models", async () => {
    const files = modelFiles();
    files.set("rigs.bin", new File([new Uint8Array(8)], "rigs.bin"));
    await expect(parseReconstruction(files)).rejects.toThrow("both rigs.bin and frames.bin");
    files.delete("rigs.bin");
    files.set("cameras.bin", new File([new Uint8Array([1])], "cameras.bin"));
    await expect(parseReconstruction(files)).rejects.toThrow("truncated");
  });
});

test("discovers nested models and ignores unrelated files", () => {
  const files = [cameraFile(), imageFile(), pointFile()];
  const entries = files.map((file) => ({path: `workspace/sparse/0/${file.name}`, file}));
  entries.push({path: "workspace/images/frame.jpg", file: new File([], "frame.jpg")});
  const candidates = discoverSparseModels(entries);
  expect(candidates).toHaveLength(1);
  expect(candidates[0]?.path).toBe("workspace/sparse/0");
});
