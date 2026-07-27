"""Build deterministic COLMAP benchmark shards from TartanAir V2."""

import argparse
import concurrent.futures
import dataclasses
import hashlib
import io
import json
import struct
import tarfile
import time
import zlib
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import numpy as np
import PIL.Image
import requests

if TYPE_CHECKING:
    from .tartanair_v2 import (
        MANIFEST_PATH,
        SceneSelection,
        load_manifest,
        scene_shards,
        select_frame_window,
        shard_name,
    )
else:
    try:
        from .tartanair_v2 import (
            MANIFEST_PATH,
            SceneSelection,
            load_manifest,
            scene_shards,
            select_frame_window,
            shard_name,
        )
    except ImportError:  # Support direct execution as a packaging script.
        from tartanair_v2 import (
            MANIFEST_PATH,
            SceneSelection,
            load_manifest,
            scene_shards,
            select_frame_window,
            shard_name,
        )


@dataclasses.dataclass(frozen=True)
class ZipMember:
    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


class RemoteZip:
    """Read selected members of a remote ZIP using HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        self.members = self._read_central_directory()

    def _range(self, byte_range: str) -> tuple[bytes, requests.Response]:
        for attempt in range(5):
            try:
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={byte_range}"},
                    timeout=180,
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Server ignored byte range for {self.url}"
                    )
                return response.content, response
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    @staticmethod
    def _zip64_values(
        extra: bytes, uncompressed: int, compressed: int, offset: int
    ) -> tuple[int, int, int]:
        position = 0
        while position + 4 <= len(extra):
            tag, size = struct.unpack_from("<HH", extra, position)
            value = extra[position + 4 : position + 4 + size]
            position += 4 + size
            if tag != 1:
                continue
            value_position = 0
            values = []
            for old_value in (uncompressed, compressed, offset):
                if old_value == 0xFFFFFFFF:
                    values.append(
                        struct.unpack_from("<Q", value, value_position)[0]
                    )
                    value_position += 8
                else:
                    values.append(old_value)
            return tuple(values)
        return uncompressed, compressed, offset

    def _read_central_directory(self) -> dict[str, ZipMember]:
        tail, response = self._range("-262144")
        total_size = int(response.headers["Content-Range"].split("/")[-1])
        eocd_offset = tail.rfind(b"PK\x05\x06")
        if eocd_offset < 0 or eocd_offset + 22 > len(tail):
            raise RuntimeError(f"Cannot locate ZIP directory for {self.url}")
        eocd = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
        num_entries, directory_size, directory_offset = eocd[4:7]
        if (
            num_entries == 0xFFFF
            or directory_size == 0xFFFFFFFF
            or directory_offset == 0xFFFFFFFF
        ):
            locator_offset = tail.rfind(
                b"PK\x06\x07", max(0, eocd_offset - 64), eocd_offset
            )
            if locator_offset < 0:
                raise RuntimeError(f"Cannot locate ZIP64 directory: {self.url}")
            zip64_offset = struct.unpack_from("<4sLQL", tail, locator_offset)[2]
            zip64, _ = self._range(f"{zip64_offset}-{zip64_offset + 55}")
            values = struct.unpack_from("<4sQ2H2L4Q", zip64)
            num_entries, directory_size, directory_offset = values[7:10]
        if directory_offset + directory_size > total_size:
            raise RuntimeError(f"Invalid ZIP directory bounds: {self.url}")

        directory, _ = self._range(
            f"{directory_offset}-{directory_offset + directory_size - 1}"
        )
        members = {}
        position = 0
        while (
            position + 46 <= len(directory)
            and directory[position : position + 4] == b"PK\x01\x02"
        ):
            values = struct.unpack_from("<4s6H3L5H2L", directory, position)
            method = values[4]
            crc32 = values[7]
            compressed = values[8]
            uncompressed = values[9]
            name_size, extra_size, comment_size = values[10:13]
            local_offset = values[16]
            name_start = position + 46
            name = directory[name_start : name_start + name_size].decode()
            extra = directory[
                name_start + name_size : name_start + name_size + extra_size
            ]
            uncompressed, compressed, local_offset = self._zip64_values(
                extra, uncompressed, compressed, local_offset
            )
            members[name] = ZipMember(
                name=name,
                method=method,
                crc32=crc32,
                compressed_size=compressed,
                uncompressed_size=uncompressed,
                local_offset=local_offset,
            )
            position += 46 + name_size + extra_size + comment_size
        if len(members) != num_entries:
            raise RuntimeError(
                f"Expected {num_entries} ZIP members, parsed {len(members)}"
            )
        return members

    def read(self, name: str) -> tuple[bytes, ZipMember]:
        member = self.members[name]
        header, _ = self._range(
            f"{member.local_offset}-{member.local_offset + 29}"
        )
        values = struct.unpack_from("<4s5H3L2H", header)
        name_size, extra_size = values[9:11]
        data_offset = member.local_offset + 30 + name_size + extra_size
        compressed, _ = self._range(
            f"{data_offset}-{data_offset + member.compressed_size - 1}"
        )
        if member.method == 0:
            data = compressed
        elif member.method == 8:
            data = zlib.decompress(compressed, -15)
        else:
            raise RuntimeError(f"Unsupported ZIP method {member.method}")
        if len(data) != member.uncompressed_size:
            raise RuntimeError(f"Invalid size for ZIP member {name}")
        if zlib.crc32(data) != member.crc32:
            raise RuntimeError(f"Invalid CRC for ZIP member {name}")
        return data, member


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(data))


DEPTH_SIZE = (512, 256)
DEPTH_SCALE = 8.0
DEPTH_INVALID_VALUE = 0
IMAGE_QUALITY = 97
IMAGE_SUBSAMPLING = 0


def encode_image_jpeg(source_png: bytes) -> bytes:
    output = io.BytesIO()
    PIL.Image.open(io.BytesIO(source_png)).convert("RGB").save(
        output,
        format="JPEG",
        quality=IMAGE_QUALITY,
        subsampling=IMAGE_SUBSAMPLING,
        optimize=True,
        progressive=True,
    )
    return output.getvalue()


def encode_depth_png(source_png: bytes) -> bytes:
    """Convert packed float32 RGBA depth to min-pooled uint16 depth."""
    rgba = np.asarray(PIL.Image.open(io.BytesIO(source_png))).copy()
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected an 8-bit RGBA depth image")
    source_height = int(rgba.shape[0])
    source_width = int(rgba.shape[1])
    depth = rgba.view("<f4").reshape(source_height, source_width)

    output_width, output_height = DEPTH_SIZE
    if source_width % output_width != 0 or source_height % output_height != 0:
        raise ValueError(
            f"Cannot downsample depth image of shape {depth.shape} "
            f"to {(output_height, output_width)}"
        )
    block_height = source_height // output_height
    block_width = source_width // output_width
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, np.inf)
    depth = depth.reshape(
        output_height, block_height, output_width, block_width
    ).min(axis=(1, 3))

    quantized = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth)
    quantized[valid] = np.clip(
        np.rint(depth[valid] * DEPTH_SCALE), 1, np.iinfo(np.uint16).max
    ).astype(np.uint16)

    output = io.BytesIO()
    PIL.Image.fromarray(quantized).save(
        output, format="PNG", compress_level=9, optimize=True
    )
    return output.getvalue()


def source_url(manifest: dict, scene: SceneSelection) -> str:
    source = manifest["source"]
    path = quote(scene.source_archive)
    return (
        f"https://huggingface.co/datasets/{source['repository']}/resolve/"
        f"{source['revision']}/{path}"
    )


def package_scene(
    archive: tarfile.TarFile,
    manifest: dict,
    scene: SceneSelection,
    remote_zips: dict[str, RemoteZip],
    num_workers: int,
) -> None:
    image_remote_zip = remote_zips.get(scene.source_archive)
    if image_remote_zip is None:
        image_remote_zip = RemoteZip(source_url(manifest, scene))
        remote_zips[scene.source_archive] = image_remote_zip
    depth_remote_zip = remote_zips.get(scene.depth_source_archive)
    if depth_remote_zip is None:
        depth_url = source_url(manifest, scene).replace(
            "image_lcam_equirect.zip", "depth_lcam_equirect.zip"
        )
        depth_remote_zip = RemoteZip(depth_url)
        remote_zips[scene.depth_source_archive] = depth_remote_zip
    source_prefix = (
        f"{scene.environment}/Data_{scene.difficulty}/{scene.trajectory}"
    )
    pose_name = f"{source_prefix}/pose_lcam_front.txt"
    pose_data, pose_member = image_remote_zip.read(pose_name)
    pose_lines = [line for line in pose_data.decode().splitlines() if line]
    poses = np.loadtxt(io.StringIO("\n".join(pose_lines)))
    options = manifest["frame_selection"]
    selected_window = select_frame_window(
        poses,
        num_frames=options["num_frames"],
        max_adjacent_translation_m=options["max_adjacent_translation_m"],
        max_adjacent_rotation_deg=options["max_adjacent_rotation_deg"],
    )
    selection_key = f"{scene.environment}:{scene.difficulty}:{scene.trajectory}"
    canonical_start = manifest["frame_starts"][selection_key]
    frame_ids = range(canonical_start, canonical_start + options["num_frames"])
    if frame_ids != selected_window:
        raise RuntimeError(
            f"Canonical frame window changed for {selection_key}: "
            f"expected {frame_ids.start}, selected {selected_window.start}"
        )

    scene_root = f"{scene.category}/{scene.name}"
    selected_pose_lines = []
    frames = []
    source_names = [
        (
            f"{source_prefix}/image_lcam_equirect/"
            f"{frame_id:06d}_lcam_equirect_image.png"
        )
        for frame_id in frame_ids
    ]
    depth_source_names = [
        (
            f"{source_prefix}/depth_lcam_equirect/"
            f"{frame_id:06d}_lcam_equirect_depth.png"
        )
        for frame_id in frame_ids
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        image_members = list(executor.map(image_remote_zip.read, source_names))
        depth_members = list(
            executor.map(depth_remote_zip.read, depth_source_names)
        )

    for frame_id, (image_data, image_member), (depth_data, depth_member) in zip(
        frame_ids, image_members, depth_members, strict=True
    ):
        image_name = f"{frame_id:06d}.jpg"
        depth_name = f"{frame_id:06d}.png"
        add_bytes(
            archive,
            f"{scene_root}/images/{image_name}",
            encode_image_jpeg(image_data),
        )
        add_bytes(
            archive,
            f"{scene_root}/depth/{depth_name}",
            encode_depth_png(depth_data),
        )
        selected_pose_lines.append(f"{frame_id} {pose_lines[frame_id]}")
        frames.append(
            {
                "source_frame": frame_id,
                "image_name": image_name,
                "depth_name": depth_name,
                "source_crc32": f"{image_member.crc32:08x}",
                "depth_source_crc32": f"{depth_member.crc32:08x}",
            }
        )

    metadata = {
        "manifest_version": manifest["version"],
        "category": scene.category,
        "environment": scene.environment,
        "difficulty": scene.difficulty,
        "trajectory": scene.trajectory,
        "source_archive": scene.source_archive,
        "depth_source_archive": scene.depth_source_archive,
        "pose_crc32": f"{pose_member.crc32:08x}",
        "image_size": manifest["source"]["image_size"],
        "image_encoding": {
            "format": "jpeg",
            "quality": IMAGE_QUALITY,
            "subsampling": "4:4:4",
            "progressive": True,
        },
        "depth_size": list(DEPTH_SIZE),
        "depth_scale": DEPTH_SCALE,
        "depth_invalid_value": DEPTH_INVALID_VALUE,
        "depth_downsampling": "min_pool",
        "frames": frames,
    }
    add_bytes(
        archive,
        f"{scene_root}/poses.txt",
        ("\n".join(selected_pose_lines) + "\n").encode(),
    )
    add_bytes(
        archive,
        f"{scene_root}/scene.json",
        (json.dumps(metadata, indent=2) + "\n").encode(),
    )


def build_shard(
    output_path: Path,
    manifest: dict,
    shard_index: int,
    scenes: list[SceneSelection],
    overwrite: bool,
    num_workers: int,
) -> tuple[str, str]:
    filename = shard_name(manifest, shard_index)
    target = output_path / filename
    if target.exists() and not overwrite:
        digest = sha256_file(target)
        return filename, digest

    temporary = target.with_suffix(target.suffix + ".part")
    remote_zips: dict[str, RemoteZip] = {}
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
        root = Path(__file__).parent
        add_bytes(
            archive,
            "TARTANAIR_LICENSE",
            (root / "TARTANAIR_LICENSE").read_bytes(),
        )
        add_bytes(
            archive,
            "TARTANAIR_README.md",
            (root / "TARTANAIR_README.md").read_bytes(),
        )
        add_bytes(archive, MANIFEST_PATH.name, MANIFEST_PATH.read_bytes())
        for scene in scenes:
            print(f"Packaging {scene.category}/{scene.name}")
            package_scene(archive, manifest, scene, remote_zips, num_workers)
    if temporary.stat().st_size >= 2 * 1024**3:
        raise RuntimeError(f"Release asset exceeds 2 GiB: {temporary}")
    temporary.replace(target)
    digest = sha256_file(target)
    return filename, digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        while chunk := fid.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--shards", type=int, nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    manifest = load_manifest()
    shards = scene_shards(manifest)
    selected = args.shards or list(range(len(shards)))
    if any(index < 0 or index >= len(shards) for index in selected):
        raise ValueError(f"Shard index must be in [0, {len(shards) - 1}]")
    args.output_path.mkdir(parents=True, exist_ok=True)

    checksums = {}
    for index in selected:
        filename, digest = build_shard(
            args.output_path,
            manifest,
            index,
            shards[index],
            args.overwrite,
            args.num_workers,
        )
        checksums[filename] = digest
    checksum_path = args.output_path / "tartanair_v2_checksums.json"
    existing = (
        json.loads(checksum_path.read_text()) if checksum_path.exists() else {}
    )
    existing.update(checksums)
    checksum_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"Wrote {checksum_path}")


if __name__ == "__main__":
    main()
