import {describe, expect, test} from "vitest";

import {discoverSparseModels, parseReconstruction, reconstructionTransferables} from "../viewer_src/parser";
import {findPoint3DIndex, point2DAt, point3DAt} from "../viewer_src/types";
import {BinaryWriter, cameraFile, imageFile, modernRigFiles, pointFile} from "./binary_fixture";

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
    expect(point2DAt(reconstruction.images.get(2)!, 0)?.point3DId).toBe(pointId);
    const pointIndex = findPoint3DIndex(reconstruction.points3D, pointId);
    expect(point3DAt(reconstruction.points3D, pointIndex).track).toEqual([{imageId: 2, point2DIdx: 0}]);
    expect(reconstruction.points3D.xyz).toBeInstanceOf(Float64Array);
    expect(reconstruction.points3D.colors).toBeInstanceOf(Uint8Array);
    expect(reconstruction.points3D.trackImageIds).toBeInstanceOf(Uint32Array);
    const transfer = reconstructionTransferables(reconstruction);
    expect(transfer).toHaveLength(9);
    const transferred = structuredClone(reconstruction, {transfer});
    expect(reconstruction.points3D.ids.byteLength).toBe(0);
    expect(transferred.points3D.ids[0]).toBe(pointId);
  });

  test("sorts point ids for compact binary-search lookup", async () => {
    const points = new BinaryWriter().u64(2);
    for (const [id, x] of [[9, 90], [3, 30]] as const) {
      points.u64(id).f64(x).f64(0).f64(0).u8(1).u8(2).u8(3).f64(0.5).u64(0);
    }
    const files = modelFiles();
    files.set("points3D.bin", points.file("points3D.bin"));
    const reconstruction = await parseReconstruction(files);
    expect([...reconstruction.points3D.ids]).toEqual([3n, 9n]);
    expect(point3DAt(reconstruction.points3D, findPoint3DIndex(reconstruction.points3D, 3n)).xyz).toEqual([30, 0, 0]);
    expect(point3DAt(reconstruction.points3D, findPoint3DIndex(reconstruction.points3D, 9n)).xyz).toEqual([90, 0, 0]);
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
