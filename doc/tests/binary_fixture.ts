type Numeric = number | bigint;

export class BinaryWriter {
  private bytes: number[] = [];

  private add(size: number, write: (view: DataView) => void): this {
    const buffer = new ArrayBuffer(size);
    write(new DataView(buffer));
    this.bytes.push(...new Uint8Array(buffer));
    return this;
  }

  u8(value: number): this { this.bytes.push(value); return this; }
  u32(value: number): this { return this.add(4, (view) => view.setUint32(0, value, true)); }
  i32(value: number): this { return this.add(4, (view) => view.setInt32(0, value, true)); }
  u64(value: Numeric): this { return this.add(8, (view) => view.setBigUint64(0, BigInt(value), true)); }
  f64(value: number): this { return this.add(8, (view) => view.setFloat64(0, value, true)); }
  string(value: string): this { this.bytes.push(...new TextEncoder().encode(value), 0); return this; }
  file(name: string): File { return new File([new Uint8Array(this.bytes)], name); }
}

function rigid(writer: BinaryWriter, translation: [number, number, number]): void {
  writer.f64(1).f64(0).f64(0).f64(0).f64(translation[0]).f64(translation[1]).f64(translation[2]);
}

export function cameraFile(cameraId = 1): File {
  return new BinaryWriter()
    .u64(1).u32(cameraId).i32(0).u64(640).u64(480)
    .f64(500).f64(320).f64(240)
    .file("cameras.bin");
}

export function imageFile(imageId = 2, cameraId = 1, pointId = 42n): File {
  const writer = new BinaryWriter().u64(1).u32(imageId);
  rigid(writer, [4, 5, 6]);
  return writer.u32(cameraId).string("images/frame.jpg").u64(1)
    .f64(330).f64(220).u64(pointId)
    .file("images.bin");
}

export function pointFile(imageId = 2, pointId = 42n, trackLength = 1): File {
  const writer = new BinaryWriter().u64(1).u64(pointId)
    .f64(1).f64(2).f64(3).u8(10).u8(20).u8(30).f64(0.25)
    .u64(trackLength);
  for (let i = 0; i < trackLength; ++i) writer.u32(imageId).u32(0);
  return writer.file("points3D.bin");
}

export function modernRigFiles(cameraId = 2, imageId = 2): [File, File] {
  const rigs = new BinaryWriter().u64(1).u32(7).u32(2)
    .i32(0).u32(1)
    .i32(0).u32(cameraId).u8(1);
  rigid(rigs, [1, 0, 0]);
  const frames = new BinaryWriter().u64(1).u32(9).u32(7);
  rigid(frames, [0, 2, 0]);
  frames.u32(1).i32(0).u32(cameraId).u64(imageId);
  return [rigs.file("rigs.bin"), frames.file("frames.bin")];
}
