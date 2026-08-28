#!/usr/bin/env python3
"""Selectively download one scene from DTU's official SampleSet ZIP.

The official archive is more than 6 GB, but ZIP members can be fetched using
HTTP byte ranges. This script downloads one light condition, calibration,
observability masks, evaluation code, and the structured-light reference cloud
without storing the complete archive.
"""

from __future__ import annotations

import argparse
import binascii
import io
import re
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath


DEFAULT_URL = "https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip"
LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_SIGNATURE = 0x04034B50


def request(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": "COLMAP-Metal-DTU-Benchmark/1.0"}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120
            ) as response:
                data = response.read()
            if start is not None and len(data) != end - start + 1:
                raise OSError(
                    f"HTTP range {start}-{end} returned {len(data)} bytes"
                )
            return data
        except Exception as error:  # Network failures are retried uniformly.
            last_error = error
            if attempt != 3:
                time.sleep(2**attempt)
    raise OSError(f"Failed to read {url}: {last_error}") from last_error


def content_length(url: str) -> int:
    headers = {"User-Agent": "COLMAP-Metal-DTU-Benchmark/1.0"}
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers, method="HEAD"), timeout=60
    ) as response:
        return int(response.headers["Content-Length"])


class RemoteZipReader(io.RawIOBase):
    """Seekable HTTP-range reader used only for the ZIP central directory."""

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Invalid seek mode: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before the start of the remote file")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size < 0:
            size = self.size - self.position
        end = min(self.size - 1, self.position + size - 1)
        data = request(self.url, self.position, end)
        self.position += len(data)
        return data


def selected_members(
    infos: list[zipfile.ZipInfo], scan: int, light: int
) -> list[zipfile.ZipInfo]:
    scan_prefix = f"SampleSet/MVS Data/Rectified/scan{scan}/"
    image_pattern = re.compile(rf"rect_\d{{3}}_{light}_r5000\.png$")
    calibration_prefix = "SampleSet/MVS Data/Calibration/cal18/"
    exact_names = {
        "SampleSet/ReadMe.txt",
        f"SampleSet/MVS Data/ObsMask/ObsMask{scan}_10.mat",
        f"SampleSet/MVS Data/ObsMask/Plane{scan}.mat",
        f"SampleSet/MVS Data/Points/stl/stl{scan:03d}_total.ply",
    }
    result = []
    for info in infos:
        name = info.filename
        is_image = name.startswith(scan_prefix) and image_pattern.search(name)
        is_calibration = name.startswith(calibration_prefix) and (
            name.endswith(".mat") or re.search(r"/pos_\d{3}\.txt$", name)
        )
        is_evaluation_code = name.startswith("SampleSet/Matlab evaluation code/") and (
            name.endswith(".m") or name.endswith("ReadMe.txt")
        )
        if name in exact_names or is_image or is_calibration or is_evaluation_code:
            if not info.is_dir():
                result.append(info)
    return result


def safe_output_path(output_root: Path, archive_name: str) -> Path:
    relative = PurePosixPath(archive_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe ZIP member path: {archive_name}")
    return output_root.joinpath(*relative.parts)


def extract_member(url: str, info: zipfile.ZipInfo, output_root: Path) -> Path:
    output_path = safe_output_path(output_root, info.filename)
    if output_path.is_file() and output_path.stat().st_size == info.file_size:
        return output_path

    header = request(
        url,
        info.header_offset,
        info.header_offset + LOCAL_FILE_HEADER.size - 1,
    )
    fields = LOCAL_FILE_HEADER.unpack(header)
    if fields[0] != LOCAL_FILE_SIGNATURE:
        raise ValueError(f"Invalid local ZIP header for {info.filename}")
    name_length = fields[-2]
    extra_length = fields[-1]
    data_start = info.header_offset + LOCAL_FILE_HEADER.size + name_length + extra_length
    compressed = request(url, data_start, data_start + info.compress_size - 1)

    if info.compress_type == zipfile.ZIP_STORED:
        contents = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        contents = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(
            f"Unsupported compression type {info.compress_type} for {info.filename}"
        )
    if len(contents) != info.file_size:
        raise ValueError(
            f"Size mismatch for {info.filename}: expected {info.file_size}, "
            f"got {len(contents)}"
        )
    if binascii.crc32(contents) & 0xFFFFFFFF != info.CRC:
        raise ValueError(f"CRC mismatch for {info.filename}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    temporary_path.write_bytes(contents)
    temporary_path.replace(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan", type=int, default=6)
    parser.add_argument("--light", type=int, choices=range(7), default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    archive_size = content_length(args.url)
    with zipfile.ZipFile(RemoteZipReader(args.url, archive_size)) as archive:
        members = selected_members(archive.infolist(), args.scan, args.light)
    if not members:
        parser.error(f"No members found for DTU scan {args.scan}, light {args.light}")

    total_compressed = sum(member.compress_size for member in members)
    print(
        f"Selected {len(members)} members "
        f"({total_compressed / 1024**2:.1f} MiB compressed)"
    )
    if args.list:
        for member in members:
            print(f"{member.compress_size:12d}  {member.filename}")
        return

    completed_bytes = 0
    output_root = args.output.resolve()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(extract_member, args.url, member, output_root): member
            for member in members
        }
        for index, future in enumerate(as_completed(futures), start=1):
            member = futures[future]
            future.result()
            completed_bytes += member.compress_size
            print(
                f"[{index:02d}/{len(members):02d}] "
                f"{completed_bytes / total_compressed:6.1%}  {member.filename}"
            )

    print(f"DTU scan {args.scan} is ready under {output_root / 'SampleSet'}")


if __name__ == "__main__":
    main()
