# SPDX-License-Identifier: BSD-3-Clause
"""
Visual-inertial refinement of a GoPro reconstruction.

Refines an existing COLMAP reconstruction of frames extracted from a GoPro video
with the IMU recorded in the video's GPMF telemetry track, and writes a new
reconstruction in metric scale with gravity along -Z.

Inputs:
  - a COLMAP reconstruction (e.g. from `colmap mapper`) and optionally its
    database, to complete and re-triangulate tracks during refinement,
  - the original GoPro video (.mp4/.mov with a GPMF track), or the telemetry
    pre-parsed into an .npz archive (see `load_gopro_telemetry_npz`).

Images must be named after their 0-based frame index in the original video: the
last integer in the file stem is the frame index (e.g. `frame_000123.jpg`). The
image timestamp is `index / video_fps`, with the frame rate read from the
telemetry.

The cameras must be in the video's native stabilized (HyperSmooth) frames. The
per-frame stabilization rotation (GPMF IORI, virtual camera from physical
camera) is passed to the IMU cost function, which relates the IMU to the
physical camera.

Assumptions (GoPro HERO11/HERO12, Linear lens, fixed capture mode):
  - IMU axes are mapped to the camera optical frame by `transform_raw_to_cam`,
    so the camera-from-IMU rotation is close to identity (refined by default),
    and the camera-IMU translation is zero.
  - IMU noise model from `gopro_imu_calibration`.
  - The camera-IMU clock offset (t_imu = t_video + offset) defaults to the value
    calibrated for this camera and capture mode.

Reading the telemetry from a video file requires ffmpeg/ffprobe and the `gpmf`
package (KLV parser).

Example:
    python gopro_visual_inertial_refinement.py \
        --sfm_path sparse/0 --telemetry_path GX010001.MP4 \
        --output_path sparse_vi
"""

import argparse
import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyceres
from scipy.spatial.transform import Rotation

import pycolmap
import pycolmap.inertial
from pycolmap import logging

# Camera-IMU clock offset (t_imu = t_video + offset) calibrated for the GoPro
# HERO capture mode assumed by this script.
DEFAULT_TIME_OFFSET_S = 0.00566

# -----------------------------------------------------------------------------
# GoPro telemetry (GPMF) parsing.
# -----------------------------------------------------------------------------

# Transformation matrix from GoPro accelerometer/gyroscope XYZ coordinates to
# the COLMAP camera optical frame (X right, Y down, Z forward).
P_CAM_FROM_XYZ = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# Transformation matrix from native GPMF ORIN="ZXY" channels [ch0, ch1, ch2]
# to the COLMAP camera optical frame: x = -ch1, y = -ch0, z = -ch2.
P_CAM_FROM_RAW = np.array(
    [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)


def transform_xyz_to_cam(vectors_xyz: np.ndarray) -> np.ndarray:
    """Transform (N, 3) vectors from GoPro sensor XYZ to the camera frame."""
    return np.asarray(vectors_xyz, dtype=np.float64) @ P_CAM_FROM_XYZ.T


def transform_raw_to_cam(channels_zxy: np.ndarray) -> np.ndarray:
    """Transform (N, 3) channels from native GPMF ZXY to the camera frame."""
    return np.asarray(channels_zxy, dtype=np.float64) @ P_CAM_FROM_RAW.T


def gopro_imu_calibration() -> pycolmap.ImuCalibration:
    """Noise model of the Bosch BMI260/270 IMU in GoPro HERO cameras."""
    calib = pycolmap.ImuCalibration()
    calib.imu_rate = 200.0  # Nominal, about 197.33 Hz.
    calib.gyro_noise_density = 5.0e-4  # rad / (s * sqrt(Hz))
    calib.accel_noise_density = 8.0e-3  # m / (s^2 * sqrt(Hz))
    calib.bias_gyro_random_walk_sigma = 5.0e-5  # rad / (s^2 * sqrt(Hz))
    calib.bias_accel_random_walk_sigma = 1.0e-3  # m / (s^3 * sqrt(Hz))
    calib.gravity_magnitude = 9.81
    return calib


@dataclass
class GoProTelemetry:
    """IMU and orientation streams of a GoPro video, on the video clock."""

    video_fps: float
    accl_timestamps_s: np.ndarray
    accl_cam: np.ndarray  # (N, 3) in m/s^2, camera optical frame.
    gyro_timestamps_s: np.ndarray
    gyro_cam: np.ndarray  # (N, 3) in rad/s, camera optical frame.
    iori_quats_wxyz: np.ndarray  # (F, 4) virtual cam from physical cam.
    grav_cam: np.ndarray  # (F, 3) downward gravity, physical camera frame.

    def to_imu_measurements(
        self, t_start_s: float, t_end_s: float, time_offset_s: float
    ) -> pycolmap.ImuMeasurements:
        """IMU samples in [t_start_s, t_end_s] (video clock), timestamped on
        the video clock: t_video = t_imu - time_offset_s."""
        t_imu = self.gyro_timestamps_s
        accel = np.column_stack(
            [
                np.interp(t_imu, self.accl_timestamps_s, self.accl_cam[:, c])
                for c in range(3)
            ]
        )
        t_video = t_imu - time_offset_s
        mask = (t_video >= t_start_s) & (t_video <= t_end_s)
        measurements = pycolmap.ImuMeasurements()
        prev_ts = None
        for t, gyro, acc in zip(
            t_video[mask], self.gyro_cam[mask], accel[mask], strict=True
        ):
            ts = pycolmap.timestamp_from_seconds(float(t))
            if prev_ts is not None and ts <= prev_ts:
                ts = prev_ts + 1
            prev_ts = ts
            measurements.insert(
                pycolmap.ImuMeasurement(timestamp=ts, gyro=gyro, accel=acc)
            )
        return measurements

    def frame_quantity(
        self, values: np.ndarray, frame_index: int
    ) -> np.ndarray:
        if frame_index >= len(values):
            raise ValueError(
                f"Frame {frame_index} is beyond the telemetry ({len(values)} "
                "frames). Do the image names follow the frame indices of the "
                "original video?"
            )
        return values[frame_index]


def load_gopro_telemetry_npz(path: Path) -> GoProTelemetry:
    """Load telemetry from an .npz archive with the keys accl_timestamps_s,
    gyro_timestamps_s (video clock), accl_raw|accl_xyz, gyro_raw|gyro_xyz,
    iori (wxyz, one per frame), grav, and video_fps (or cori_timestamps_s)."""
    data = np.load(str(path))

    def vectors(name: str) -> np.ndarray:
        if f"{name}_xyz" in data:
            return transform_xyz_to_cam(data[f"{name}_xyz"])
        if f"{name}_raw" in data:
            return transform_raw_to_cam(data[f"{name}_raw"])
        raise KeyError(f"Archive misses '{name}_xyz' or '{name}_raw'")

    if "video_fps" in data:
        video_fps = float(data["video_fps"])
    else:
        video_fps = 1.0 / float(np.median(np.diff(data["cori_timestamps_s"])))
    return GoProTelemetry(
        video_fps=video_fps,
        accl_timestamps_s=np.asarray(data["accl_timestamps_s"], np.float64),
        accl_cam=vectors("accl"),
        gyro_timestamps_s=np.asarray(data["gyro_timestamps_s"], np.float64),
        gyro_cam=vectors("gyro"),
        iori_quats_wxyz=np.asarray(data["iori"], np.float64),
        grav_cam=np.asarray(data["grav"], np.float64),
    )


def _parse_strm(strm_items: Sequence) -> tuple:
    """Extract (stream_key, scaled_data, stmp_us) from a GPMF STRM block."""
    scale, stmp_us, key, raw = 1.0, None, None, None
    for sub in strm_items:
        if sub.key == "SCAL":
            value = np.asarray(sub.value).reshape(-1)
            if value.size and value[0] != 0:
                scale = float(value[0])
        elif sub.key == "STMP":
            stmp_us = int(np.asarray(sub.value).reshape(-1)[0])
        elif sub.key in ("ACCL", "GYRO", "CORI", "IORI", "GRAV"):
            key, raw = sub.key, sub.value
    if key is None:
        return None, None, stmp_us
    return key, np.asarray(raw, dtype=np.float64) / scale, stmp_us


def _stmp_anchored_timestamps(
    blocks: list[np.ndarray], stmps_s: np.ndarray
) -> np.ndarray:
    """Anchor every payload at its STMP (camera clock time of its first
    sample), with samples spaced uniformly up to the next payload."""
    counts = np.array([len(block) for block in blocks], dtype=np.float64)
    periods = np.diff(stmps_s) / counts[:-1]
    if np.any(periods <= 0):
        raise ValueError("STMP timestamps must be strictly increasing")
    periods = np.append(periods, periods[-1])
    return np.concatenate(
        [
            stmp + np.arange(len(block)) * period
            for stmp, period, block in zip(
                stmps_s, periods, blocks, strict=True
            )
        ]
    )


def extract_gpmf_stream(video_path: Path) -> bytes:
    """Extract the raw GPMF track ('gpmd' stream) of a video with ffmpeg."""
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json"]
            + [str(video_path)]
        )
    )
    for stream in probe["streams"]:
        if stream.get("codec_tag_string") == "gpmd":
            return subprocess.check_output(
                ["ffmpeg", "-v", "error", "-i", str(video_path)]
                + ["-map", f"0:{stream['index']}", "-codec", "copy"]
                + ["-f", "rawvideo", "pipe:"]
            )
    raise ValueError(f"No GPMF track in {video_path}")


def load_gopro_telemetry_gpmf(path: Path) -> GoProTelemetry:
    """Parse the GPMF track of a GoPro video (or a raw .bin GPMF dump)."""
    import gpmf.parse

    if path.suffix.lower() in (".mp4", ".mov"):
        buf = extract_gpmf_stream(path)
    else:
        buf = path.read_bytes()
    keys = ("ACCL", "GYRO", "CORI", "IORI", "GRAV")
    blocks: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    stmps: dict[str, list[int]] = {k: [] for k in keys}
    for devc in gpmf.parse.iter_klv(buf):
        if devc.key != "DEVC":
            continue
        for item in devc.value:
            if item.key != "STRM":
                continue
            key, data, stmp_us = _parse_strm(list(item.value))
            if key is not None:
                blocks[key].append(data)
                if stmp_us is not None:
                    stmps[key].append(stmp_us)
    for key in keys:
        if not blocks[key]:
            raise ValueError(f"No {key} stream in the GPMF track of {path}")
        if len(stmps[key]) != len(blocks[key]):
            raise ValueError(f"Missing STMP for some {key} payloads")

    # The first CORI sample is the first video frame: use it as time origin.
    origin_us = stmps["CORI"][0]

    def timestamps(key: str) -> np.ndarray:
        stmps_s = (np.asarray(stmps[key], np.float64) - origin_us) * 1e-6
        return _stmp_anchored_timestamps(blocks[key], stmps_s)

    # One CORI sample per video frame.
    cori_stmps_s = (np.asarray(stmps["CORI"], np.float64) - origin_us) * 1e-6
    num_frames = sum(len(b) for b in blocks["CORI"][:-1])
    video_fps = num_frames / (cori_stmps_s[-1] - cori_stmps_s[0])
    return GoProTelemetry(
        video_fps=video_fps,
        accl_timestamps_s=timestamps("ACCL"),
        accl_cam=transform_raw_to_cam(np.concatenate(blocks["ACCL"])),
        gyro_timestamps_s=timestamps("GYRO"),
        gyro_cam=transform_raw_to_cam(np.concatenate(blocks["GYRO"])),
        iori_quats_wxyz=np.concatenate(blocks["IORI"]),
        grav_cam=np.concatenate(blocks["GRAV"]),
    )


def load_gopro_telemetry(path: Path) -> GoProTelemetry:
    if path.suffix.lower() == ".npz":
        return load_gopro_telemetry_npz(path)
    return load_gopro_telemetry_gpmf(path)


# -----------------------------------------------------------------------------
# IMU edges and initialization.
# -----------------------------------------------------------------------------


def frame_index_from_name(name: str) -> int:
    matches = re.findall(r"\d+", Path(name).stem)
    if not matches:
        raise ValueError(f"No frame index in image name '{name}'")
    return int(matches[-1])


@dataclass
class ImuEdge:
    image_id1: int
    image_id2: int
    integrator: pycolmap.inertial.ImuPreintegrator
    data: pycolmap.inertial.PreintegratedImuData


@dataclass
class ImageTiming:
    image_id: int
    frame_index: int
    timestamp_s: float
    q_iori_xyzw: np.ndarray  # Virtual (stabilized) camera from physical camera.


def collect_image_timings(
    rec: pycolmap.Reconstruction, telemetry: GoProTelemetry
) -> list[ImageTiming]:
    timings = []
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        index = frame_index_from_name(image.name)
        q_wxyz = telemetry.frame_quantity(telemetry.iori_quats_wxyz, index)
        q_xyzw = np.asarray(q_wxyz)[[1, 2, 3, 0]]
        timings.append(
            ImageTiming(
                image_id,
                index,
                index / telemetry.video_fps,
                q_xyzw / np.linalg.norm(q_xyzw),
            )
        )
    timings.sort(key=lambda t: t.frame_index)
    return timings


def build_imu_edges(
    timings: list[ImageTiming],
    telemetry: GoProTelemetry,
    calib: pycolmap.ImuCalibration,
    time_offset_s: float,
    max_edge_duration_s: float,
) -> list[ImuEdge]:
    """Preintegrate the IMU between consecutive registered images."""
    measurements = telemetry.to_imu_measurements(
        timings[0].timestamp_s - 0.5,
        timings[-1].timestamp_s + 0.5,
        time_offset_s,
    )
    options = pycolmap.inertial.ImuPreintegrationOptions()
    options.method = pycolmap.inertial.ImuIntegrationMethod.RK4
    edges = []
    for prev, curr in zip(timings[:-1], timings[1:], strict=True):
        if curr.timestamp_s - prev.timestamp_s > max_edge_duration_s:
            logging.warning(
                f"Skip IMU edge between frames {prev.frame_index} and "
                f"{curr.frame_index}: longer than {max_edge_duration_s} s"
            )
            continue
        t1 = pycolmap.timestamp_from_seconds(prev.timestamp_s)
        t2 = pycolmap.timestamp_from_seconds(curr.timestamp_s)
        ms = measurements.extract_measurements_in_range(t1, t2)
        if len(ms) < 2:
            logging.warning(
                f"Skip IMU edge between frames {prev.frame_index} and "
                f"{curr.frame_index}: no IMU measurements"
            )
            continue
        integrator = pycolmap.inertial.ImuPreintegrator(options, calib, t1, t2)
        integrator.integrate(ms)
        edges.append(
            ImuEdge(
                prev.image_id, curr.image_id, integrator, integrator.extract()
            )
        )
    return edges


def so3_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def world_from_imu(
    rec: pycolmap.Reconstruction,
    timing: ImageTiming,
    imu_from_cam: pycolmap.Rigid3d,
) -> tuple[np.ndarray, np.ndarray]:
    """IMU orientation and position in the SfM world (position in SfM units,
    assuming a zero camera-IMU translation)."""
    cam_from_world = rec.images[timing.image_id].cam_from_world()
    R_wc = cam_from_world.rotation.matrix().T
    R_c_phys = Rotation.from_quat(timing.q_iori_xyzw).as_matrix()
    R_wi = R_wc @ R_c_phys @ imu_from_cam.rotation.matrix().T
    return R_wi, cam_from_world.inverse().translation


def initialize_gyro_bias(
    rec: pycolmap.Reconstruction,
    timings: dict[int, ImageTiming],
    edges: list[ImuEdge],
    imu_from_cam: pycolmap.Rigid3d,
) -> np.ndarray:
    """Least-squares gyroscope bias from visual relative rotations."""
    A, b = [], []
    for edge in edges:
        R_i, _ = world_from_imu(rec, timings[edge.image_id1], imu_from_cam)
        R_j, _ = world_from_imu(rec, timings[edge.image_id2], imu_from_cam)
        # The preintegrated rotation is j_from_i: delta_R * Exp(dR_dbg * dbg)
        # should match R_j^T * R_i.
        delta_R = Rotation.from_quat(edge.data.delta_R.quat).as_matrix()
        A.append(edge.data.dR_dbg)
        b.append(so3_log(delta_R.T @ R_j.T @ R_i))
    return np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]


def initialize_scale_gravity_velocities(
    rec: pycolmap.Reconstruction,
    timings: dict[int, ImageTiming],
    edges: list[ImuEdge],
    imu_from_cam: pycolmap.Rigid3d,
    gravity_magnitude: float,
) -> tuple[float, np.ndarray, dict[int, np.ndarray]]:
    """Linear visual-inertial alignment with known rotations and zero
    accelerometer bias. Returns the metric scale, the unit gravity direction
    in the SfM world, and the metric velocities of the IMU."""
    image_ids = sorted(
        {e.image_id1 for e in edges} | {e.image_id2 for e in edges}
    )
    col = {image_id: 3 * k for k, image_id in enumerate(image_ids)}
    num_vel = 3 * len(image_ids)

    def solve(gravity: np.ndarray | None) -> np.ndarray:
        num_unknowns = num_vel + (4 if gravity is None else 1)
        A = np.zeros((6 * len(edges), num_unknowns))
        b = np.zeros(6 * len(edges))
        for k, edge in enumerate(edges):
            R_i, p_i = world_from_imu(
                rec, timings[edge.image_id1], imu_from_cam
            )
            _, p_j = world_from_imu(rec, timings[edge.image_id2], imu_from_cam)
            dt = edge.data.delta_t
            rp, rv = slice(6 * k, 6 * k + 3), slice(6 * k + 3, 6 * k + 6)
            ci, cj = col[edge.image_id1], col[edge.image_id2]
            # s * (p_j - p_i) - v_i * dt - 0.5 * g * dt^2 = R_i * delta_p
            A[rp, ci : ci + 3] = -dt * np.eye(3)
            A[rp, num_vel] = p_j - p_i
            b[rp] = R_i @ edge.data.delta_p
            # v_j - v_i - g * dt = R_i * delta_v
            A[rv, ci : ci + 3] = -np.eye(3)
            A[rv, cj : cj + 3] = np.eye(3)
            b[rv] = R_i @ edge.data.delta_v
            if gravity is None:
                A[rp, num_vel + 1 :] = -0.5 * dt**2 * np.eye(3)
                A[rv, num_vel + 1 :] = -dt * np.eye(3)
            else:
                b[rp] += 0.5 * dt**2 * gravity
                b[rv] += dt * gravity
        return np.linalg.lstsq(A, b, rcond=None)[0]

    x = solve(None)
    gravity = x[num_vel + 1 :]
    logging.info(
        f"Unconstrained alignment: scale={x[num_vel]:.4f}, "
        f"|g|={np.linalg.norm(gravity):.3f} m/s^2"
    )
    gravity = gravity_magnitude * gravity / np.linalg.norm(gravity)
    x = solve(gravity)
    scale = float(x[num_vel])
    if scale <= 0:
        raise RuntimeError(f"Visual-inertial alignment failed (scale={scale})")
    velocities = {i: x[col[i] : col[i] + 3] for i in image_ids}
    return scale, gravity / gravity_magnitude, velocities


def gravity_from_grav_stream(
    rec: pycolmap.Reconstruction,
    timings: list[ImageTiming],
    telemetry: GoProTelemetry,
) -> np.ndarray:
    """Mean world gravity direction from the camera's GRAV stream (gravity
    sensor fusion in the physical camera frame), for diagnostics."""
    samples = []
    for timing in timings:
        grav = telemetry.frame_quantity(telemetry.grav_cam, timing.frame_index)
        R_wc = rec.images[timing.image_id].cam_from_world().rotation.matrix().T
        R_c_phys = Rotation.from_quat(timing.q_iori_xyzw).as_matrix()
        samples.append(R_wc @ R_c_phys @ (grav / np.linalg.norm(grav)))
    mean = np.mean(samples, axis=0)
    return mean / np.linalg.norm(mean)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# -----------------------------------------------------------------------------
# Visual-inertial bundle adjustment.
# -----------------------------------------------------------------------------


class ImuReintegrationCallback(pyceres.IterationCallback):
    """Reintegrates the IMU when the optimized biases drift beyond the
    linearization point of the preintegration."""

    def __init__(
        self,
        options: pycolmap.inertial.ImuReintegrationOptions,
        edges: list[ImuEdge],
        imu_states: dict[int, pycolmap.ImuState],
    ) -> None:
        pyceres.IterationCallback.__init__(self)
        self.options = options
        self.edges = edges
        self.imu_states = imu_states

    def __call__(
        self, summary: pyceres.IterationSummary
    ) -> pyceres.CallbackReturnType:
        for edge in self.edges:
            biases = self.imu_states[edge.image_id1].params[3:9]
            diff = biases - edge.data.biases
            dt = edge.data.delta_t
            if (
                np.linalg.norm(diff[:3]) * dt
                > self.options.reintegrate_angle_norm_thres
                or np.linalg.norm(diff[3:]) * dt
                > self.options.reintegrate_vel_norm_thres
            ):
                edge.integrator.reintegrate(biases)
                edge.integrator.update(edge.data)
        return pyceres.CallbackReturnType.SOLVER_CONTINUE


@dataclass
class InertialVariables:
    log_scale: np.ndarray  # (1,)
    gravity: np.ndarray  # (3,) unit direction in the SfM world.
    imu_from_cam: pycolmap.Rigid3d
    imu_states: dict[int, pycolmap.ImuState]


def add_imu_residuals(
    problem: pyceres.Problem,
    rec: pycolmap.Reconstruction,
    edges: list[ImuEdge],
    timings: dict[int, ImageTiming],
    variables: InertialVariables,
    refine_imu_from_cam_rotation: bool,
) -> None:
    # The residuals are whitened by the preintegration covariance.
    loss = pyceres.TrivialLoss()
    for edge in edges:
        frame_i = rec.images[edge.image_id1].frame
        frame_j = rec.images[edge.image_id2].frame
        if len(frame_i.rig.non_ref_sensors) or len(frame_j.rig.non_ref_sensors):
            raise ValueError("The IMU cost function requires trivial rigs")
        cost = pycolmap.inertial.AnalyticalVisualCentricImuPreintegrationCost(
            edge.data,
            q_iori_i=pycolmap.Rotation3d(timings[edge.image_id1].q_iori_xyzw),
            q_iori_j=pycolmap.Rotation3d(timings[edge.image_id2].q_iori_xyzw),
        )
        problem.add_residual_block(
            cost,
            loss,
            [
                variables.log_scale,
                variables.gravity,
                variables.imu_from_cam.params,
                frame_i.rig_from_world.params,
                variables.imu_states[edge.image_id1].params,
                frame_j.rig_from_world.params,
                variables.imu_states[edge.image_id2].params,
            ],
        )
    problem.set_manifold(variables.gravity, pyceres.SphereManifold(3))
    # The camera-IMU translation is not observable enough on handheld video:
    # keep it at zero and optionally refine the rotation.
    problem.set_manifold(
        variables.imu_from_cam.params,
        pyceres.ProductManifold(
            pyceres.EigenQuaternionManifold(),
            pyceres.SubsetManifold(3, np.arange(3)),
        ),
    )
    if not refine_imu_from_cam_rotation:
        problem.set_parameter_block_constant(variables.imu_from_cam.params)


def solve_visual_inertial_bundle_adjustment(
    rec: pycolmap.Reconstruction,
    ba_options: pycolmap.BundleAdjustmentOptions,
    edges: list[ImuEdge],
    timings: dict[int, ImageTiming],
    variables: InertialVariables,
    refine_imu_from_cam_rotation: bool,
) -> pyceres.SolverSummary:
    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():
        ba_config.add_image(image_id)
    # Fix the 7-DoF gauge of the SfM world: its scale is separate from the
    # metric scale and its orientation is free relative to gravity.
    ba_config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    bundle_adjuster = pycolmap.create_default_ceres_bundle_adjuster(
        ba_options, ba_config, rec
    )
    problem = bundle_adjuster.problem
    add_imu_residuals(
        problem,
        rec,
        edges,
        timings,
        variables,
        refine_imu_from_cam_rotation,
    )
    solver_options = pyceres.SolverOptions(
        ba_options.ceres.create_solver_options(ba_config, problem)
    )
    callback = ImuReintegrationCallback(
        pycolmap.inertial.ImuReintegrationOptions(), edges, variables.imu_states
    )
    # The callbacks getter returns a copy: assign the full list.
    solver_options.callbacks = [callback]
    solver_options.update_state_every_iteration = True
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, problem, summary)
    logging.info(summary.BriefReport())
    return summary


def refine(
    rec: pycolmap.Reconstruction,
    database_path: Path | None,
    edges: list[ImuEdge],
    timings: dict[int, ImageTiming],
    variables: InertialVariables,
    refine_imu_from_cam_rotation: bool,
) -> pycolmap.Reconstruction:
    """Iterative global visual-inertial refinement with outlier filtering.

    With a database, tracks are also completed, merged, and re-triangulated
    from the database correspondences, as in
    IncrementalMapper.iterative_global_refinement.
    """
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    pipeline_options.fix_existing_frames = False
    mapper_options = pipeline_options.get_mapper()
    tri_options = pipeline_options.get_triangulation()
    ba_options = pipeline_options.get_global_bundle_adjustment()
    ba_options.refine_focal_length = True
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = True

    mapper = None
    if database_path is not None:
        with pycolmap.Database.open(str(database_path)) as database:
            database_cache = pycolmap.DatabaseCache.create(
                database, pycolmap.DatabaseCacheOptions()
            )
        mapper = pycolmap.IncrementalMapper(database_cache)
        mapper.begin_reconstruction(rec)
        rec = mapper.reconstruction
        observation_manager = mapper.observation_manager
        mapper.complete_and_merge_tracks(tri_options)
        mapper.retriangulate(tri_options)
    else:
        observation_manager = pycolmap.ObservationManager(rec)

    for _ in range(pipeline_options.ba_global_max_refinements):
        num_observations = rec.compute_num_observations()
        observation_manager.filter_observations_with_negative_depth()
        solve_visual_inertial_bundle_adjustment(
            rec,
            ba_options,
            edges,
            timings,
            variables,
            refine_imu_from_cam_rotation,
        )
        if mapper is not None:
            num_changed = mapper.complete_and_merge_tracks(tri_options)
            num_changed += mapper.filter_points(mapper_options)
        else:
            num_changed = observation_manager.filter_all_points3D(
                mapper_options.filter_max_reproj_error,
                mapper_options.filter_min_tri_angle,
            )
        changed = num_changed / max(num_observations, 1)
        logging.info(f"Changed observations: {changed:.6f}")
        if changed < pipeline_options.ba_global_max_refinement_change:
            break
    return rec


# -----------------------------------------------------------------------------
# Output.
# -----------------------------------------------------------------------------


def metric_gravity_aligned_from_sfm(
    rec: pycolmap.Reconstruction,
    timings: list[ImageTiming],
    variables: InertialVariables,
) -> pycolmap.Sim3d:
    """Similarity from the SfM world to a metric world with gravity along -Z
    and the first camera at the origin."""
    scale = float(np.exp(variables.log_scale[0]))
    gravity = variables.gravity / np.linalg.norm(variables.gravity)
    rotation, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [gravity])
    R = rotation.as_matrix()
    center = (
        rec.images[timings[0].image_id].cam_from_world().inverse().translation
    )
    return pycolmap.Sim3d(scale, pycolmap.Rotation3d(R), -scale * R @ center)


def write_inertial_states(
    path: Path,
    rec: pycolmap.Reconstruction,
    timings: list[ImageTiming],
    variables: InertialVariables,
    metric_from_sfm: pycolmap.Sim3d,
    gravity_vs_grav_deg: float,
) -> None:
    scale = float(np.exp(variables.log_scale[0]))
    R = metric_from_sfm.rotation.matrix()
    states: dict[str, Any] = {
        "metric_scale": scale,
        "gravity_direction_in_world": [0.0, 0.0, -1.0],
        "gravity_vs_gopro_grav_deg": gravity_vs_grav_deg,
        "imu_from_cam_xyzw": variables.imu_from_cam.rotation.quat.tolist(),
        "imu_from_cam_angle_deg": float(
            np.degrees(variables.imu_from_cam.rotation.angle())
        ),
        "images": [],
    }
    for timing in timings:
        state = variables.imu_states.get(timing.image_id)
        entry: dict[str, Any] = {
            "image_id": timing.image_id,
            "name": rec.images[timing.image_id].name,
            "timestamp_s": timing.timestamp_s,
        }
        if state is not None:
            entry["velocity_m_s"] = (R @ (scale * state.velocity)).tolist()
            entry["bias_gyro_rad_s"] = np.asarray(state.bias_gyro).tolist()
            entry["bias_accel_m_s2"] = np.asarray(state.bias_accel).tolist()
        states["images"].append(entry)
    path.write_text(json.dumps(states, indent=1) + "\n")


def run(args: argparse.Namespace) -> None:
    rec = pycolmap.Reconstruction(str(args.sfm_path))
    telemetry = load_gopro_telemetry(args.telemetry_path)
    logging.info(
        f"Telemetry: {len(telemetry.gyro_timestamps_s)} IMU samples, "
        f"{len(telemetry.iori_quats_wxyz)} frames at "
        f"{telemetry.video_fps:.3f} fps"
    )
    calib = gopro_imu_calibration()
    timings = collect_image_timings(rec, telemetry)
    timing_of = {t.image_id: t for t in timings}
    edges = build_imu_edges(
        timings, telemetry, calib, args.time_offset_s, args.max_edge_duration_s
    )
    logging.info(f"{len(edges)} IMU edges between {len(timings)} images")
    if not edges:
        raise RuntimeError("No IMU edges")

    # Initialization: gyroscope bias, then scale, gravity, and velocities.
    imu_from_cam = pycolmap.Rigid3d()
    bias_gyro = initialize_gyro_bias(rec, timing_of, edges, imu_from_cam)
    logging.info(f"Initial gyroscope bias: {bias_gyro} rad/s")
    for edge in edges:
        edge.integrator.reintegrate(np.concatenate([bias_gyro, np.zeros(3)]))
        edge.integrator.update(edge.data)
    scale, gravity, velocities = initialize_scale_gravity_velocities(
        rec, timing_of, edges, imu_from_cam, calib.gravity_magnitude
    )
    grav_ref = gravity_from_grav_stream(rec, timings, telemetry)
    logging.info(
        f"Initial scale={scale:.4f}, gravity vs GoPro GRAV: "
        f"{angle_deg(gravity, grav_ref):.2f} deg"
    )
    imu_states = {}
    for image_id, velocity in velocities.items():
        state = pycolmap.ImuState()
        state.velocity = velocity / scale  # SfM units.
        state.bias_gyro = bias_gyro
        imu_states[image_id] = state
    variables = InertialVariables(
        log_scale=np.array([np.log(scale)]),
        gravity=gravity.copy(),
        imu_from_cam=imu_from_cam,
        imu_states=imu_states,
    )

    rec = refine(
        rec,
        args.database_path,
        edges,
        timing_of,
        variables,
        not args.fix_imu_from_cam,
    )

    grav_ref = gravity_from_grav_stream(rec, timings, telemetry)
    gravity_vs_grav = angle_deg(variables.gravity, grav_ref)
    logging.info(
        f"Final scale={np.exp(variables.log_scale[0]):.4f}, gravity vs GoPro "
        f"GRAV: {gravity_vs_grav:.2f} deg, camera-IMU rotation: "
        f"{np.degrees(variables.imu_from_cam.rotation.angle()):.3f} deg, "
        "mean reprojection error: "
        f"{rec.compute_mean_reprojection_error():.3f} px"
    )
    metric_from_sfm = metric_gravity_aligned_from_sfm(rec, timings, variables)
    write_inertial_states(
        args.output_path / "inertial_states.json",
        rec,
        timings,
        variables,
        metric_from_sfm,
        gravity_vs_grav,
    )
    rec.transform(metric_from_sfm)
    args.output_path.mkdir(parents=True, exist_ok=True)
    rec.write(str(args.output_path))
    logging.info(
        f"Wrote the metric, gravity-aligned model to {args.output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--sfm_path", type=Path, required=True)
    parser.add_argument(
        "--database_path",
        type=Path,
        help="Database of the reconstruction. If given, tracks are completed, "
        "merged, and re-triangulated between bundle adjustment iterations.",
    )
    parser.add_argument(
        "--telemetry_path",
        type=Path,
        required=True,
        help="GoPro video (.mp4/.mov), raw GPMF dump (.bin), or .npz archive.",
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--time_offset_s",
        type=float,
        default=DEFAULT_TIME_OFFSET_S,
        help="Camera-IMU clock offset: t_imu = t_video + offset.",
    )
    parser.add_argument(
        "--max_edge_duration_s",
        type=float,
        default=1.0,
        help="Do not preintegrate between images further apart in time.",
    )
    parser.add_argument(
        "--fix_imu_from_cam",
        action="store_true",
        help="Do not refine the camera-IMU rotation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    pycolmap.set_random_seed(0)
    args = parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    run(args)
