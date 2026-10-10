# SPDX-License-Identifier: BSD-3-Clause
"""
Visual-inertial refinement of a GoPro reconstruction.

Refines an existing COLMAP reconstruction of frames extracted from a GoPro video
with the IMU recorded in the video's GPMF telemetry track, and writes a new
reconstruction in metric scale with gravity along +Y (the convention of
pycolmap.gravity_aligned_rotation).

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
stabilization is a per-frame rotation about the camera center (GPMF IORI,
virtual camera from physical camera). It is undone upfront: keypoints are mapped
by the infinite homography K * R_iori^T * K^-1 and poses are rotated into the
physical camera frame, which is rigidly attached to the IMU, so that the
standard IMU cost function applies. The camera intrinsics are thus kept fixed.
The output is rotated back into the stabilized frames, with the original
keypoints.

Assumptions (GoPro HERO11/HERO12, Linear lens, fixed capture mode):
  - IMU axes are mapped to the camera optical frame by `P_CAM_FROM_RAW`, so the
    camera-from-IMU rotation is close to identity (refined by default), and the
    camera-IMU translation is zero.
  - IMU noise model from `gopro_imu_calibration`.
  - The camera-IMU clock offset defaults to `DEFAULT_TIME_OFFSET_S`.

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

import pycolmap
import pycolmap.inertial
from pycolmap import logging

# Camera-IMU clock offset such that t_imu = t_video + offset. Estimated on a
# 90 s walking sequence recorded with the camera and capture mode assumed here:
# for consecutive frames, the relative rotation of the physical camera from a
# vision-only reconstruction (stabilized poses composed with the IORI
# rotations) was compared with the bias-corrected gyroscope integration over
# [t_i + offset, t_j + offset], scanning the offset and re-estimating the gyro
# bias for each candidate. The residual is minimal at 5.66 ms, and independent
# 30 s sub-windows agree within 5.0-6.0 ms. Re-estimate it for other cameras or
# capture modes (e.g. different frame rates).
DEFAULT_TIME_OFFSET_S = 0.00566

# -----------------------------------------------------------------------------
# GoPro telemetry (GPMF) parsing.
# -----------------------------------------------------------------------------

# GPMF stream keys: accelerometer, gyroscope, camera orientation (one sample
# per frame, defines the video time origin), image orientation (stabilization
# rotation, one per frame), and gravity direction (one per frame).
ACCL, GYRO, CORI, IORI, GRAV = "ACCL", "GYRO", "CORI", "IORI", "GRAV"
GPMF_STREAM_KEYS = (ACCL, GYRO, CORI, IORI, GRAV)

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
        samples = []
        prev_ts = None
        for t, gyro, acc in zip(
            t_video[mask], self.gyro_cam[mask], accel[mask], strict=True
        ):
            ts = pycolmap.timestamp_from_seconds(float(t))
            if prev_ts is not None and ts <= prev_ts:
                ts = prev_ts + 1  # Strictly increasing timestamps.
            prev_ts = ts
            samples.append(
                pycolmap.ImuMeasurement(timestamp=ts, gyro=gyro, accel=acc)
            )
        measurements = pycolmap.ImuMeasurements()
        measurements.insert_sorted(samples)
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

    def vectors_cam(key: str) -> np.ndarray:
        name = key.lower()
        if f"{name}_xyz" in data:
            return (
                np.asarray(data[f"{name}_xyz"], np.float64) @ P_CAM_FROM_XYZ.T
            )
        if f"{name}_raw" in data:
            return (
                np.asarray(data[f"{name}_raw"], np.float64) @ P_CAM_FROM_RAW.T
            )
        raise KeyError(f"Archive misses '{name}_xyz' or '{name}_raw'")

    def array(key: str, suffix: str = "") -> np.ndarray:
        return np.asarray(data[key.lower() + suffix], np.float64)

    if "video_fps" in data:
        video_fps = float(data["video_fps"])
    else:
        video_fps = 1.0 / float(
            np.median(np.diff(array(CORI, "_timestamps_s")))
        )
    return GoProTelemetry(
        video_fps=video_fps,
        accl_timestamps_s=array(ACCL, "_timestamps_s"),
        accl_cam=vectors_cam(ACCL),
        gyro_timestamps_s=array(GYRO, "_timestamps_s"),
        gyro_cam=vectors_cam(GYRO),
        iori_quats_wxyz=array(IORI),
        grav_cam=array(GRAV),
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
        elif sub.key in GPMF_STREAM_KEYS:
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
    blocks: dict[str, list[np.ndarray]] = {k: [] for k in GPMF_STREAM_KEYS}
    stmps: dict[str, list[int]] = {k: [] for k in GPMF_STREAM_KEYS}
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
    for key in GPMF_STREAM_KEYS:
        if not blocks[key]:
            raise ValueError(f"No {key} stream in the GPMF track of {path}")
        if len(stmps[key]) != len(blocks[key]):
            raise ValueError(f"Missing STMP for some {key} payloads")

    # The first CORI sample is the first video frame: use it as time origin.
    origin_us = stmps[CORI][0]

    def stmps_s(key: str) -> np.ndarray:
        return (np.asarray(stmps[key], np.float64) - origin_us) * 1e-6

    # One CORI sample per video frame.
    num_frames = sum(len(b) for b in blocks[CORI][:-1])
    video_fps = num_frames / (stmps_s(CORI)[-1] - stmps_s(CORI)[0])
    return GoProTelemetry(
        video_fps=video_fps,
        accl_timestamps_s=_stmp_anchored_timestamps(
            blocks[ACCL], stmps_s(ACCL)
        ),
        accl_cam=np.concatenate(blocks[ACCL]) @ P_CAM_FROM_RAW.T,
        gyro_timestamps_s=_stmp_anchored_timestamps(
            blocks[GYRO], stmps_s(GYRO)
        ),
        gyro_cam=np.concatenate(blocks[GYRO]) @ P_CAM_FROM_RAW.T,
        iori_quats_wxyz=np.concatenate(blocks[IORI]),
        grav_cam=np.concatenate(blocks[GRAV]),
    )


def load_gopro_telemetry(path: Path) -> GoProTelemetry:
    if path.suffix.lower() == ".npz":
        return load_gopro_telemetry_npz(path)
    return load_gopro_telemetry_gpmf(path)


# -----------------------------------------------------------------------------
# Image timing and IMU preintegration.
# -----------------------------------------------------------------------------


def frame_index_from_name(name: str) -> int:
    matches = re.findall(r"\d+", Path(name).stem)
    if not matches:
        raise ValueError(f"No frame index in image name '{name}'")
    return int(matches[-1])


@dataclass
class ImageTiming:
    image_id: int
    frame_index: int
    timestamp_s: float
    q_iori: pycolmap.Rotation3d  # Virtual (stabilized) from physical camera.


@dataclass
class ImuEdge:
    image_id1: int
    image_id2: int
    integrator: pycolmap.inertial.ImuPreintegrator
    data: pycolmap.inertial.PreintegratedImuData


def collect_image_timings(
    rec: pycolmap.Reconstruction, telemetry: GoProTelemetry
) -> list[ImageTiming]:
    timings = []
    for image_id in rec.reg_image_ids():
        index = frame_index_from_name(rec.images[image_id].name)
        q_wxyz = telemetry.frame_quantity(telemetry.iori_quats_wxyz, index)
        q_iori = pycolmap.Rotation3d(np.asarray(q_wxyz)[[1, 2, 3, 0]])
        q_iori.normalize()
        timings.append(
            ImageTiming(image_id, index, index / telemetry.video_fps, q_iori)
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


def undo_stabilization(
    rec: pycolmap.Reconstruction, timings: list[ImageTiming]
) -> dict[int, np.ndarray]:
    """Express each image in its physical camera frame. Returns the original
    (stabilized) keypoints, to restore them with `redo_stabilization`.

    The stabilization rotates the camera about its center, so the 3D points
    are unchanged: cam_phys_from_world = R_iori^T * cam_from_world, and the
    keypoints follow the infinite homography K * R_iori^T * K^-1 with the
    current intrinsics K.
    """
    original_keypoints = {}
    for timing in timings:
        image = rec.images[timing.image_id]
        camera = rec.cameras[image.camera_id]
        keypoints = np.array([p.xy for p in image.points2D]).reshape(-1, 2)
        original_keypoints[timing.image_id] = keypoints
        rays = np.column_stack(
            [camera.cam_from_img(keypoints), np.ones(len(keypoints))]
        )
        phys_keypoints = camera.img_from_cam(timing.q_iori.inverse() * rays)
        for point2D, xy in zip(image.points2D, phys_keypoints, strict=True):
            point2D.xy = xy
        phys_from_virtual = pycolmap.Rigid3d(
            timing.q_iori.inverse(), np.zeros(3)
        )
        image.frame.rig_from_world = phys_from_virtual * image.cam_from_world()
    return original_keypoints


def redo_stabilization(
    rec: pycolmap.Reconstruction,
    timings: list[ImageTiming],
    original_keypoints: dict[int, np.ndarray],
) -> None:
    """Inverse of `undo_stabilization`."""
    for timing in timings:
        image = rec.images[timing.image_id]
        for point2D, xy in zip(
            image.points2D, original_keypoints[timing.image_id], strict=True
        ):
            point2D.xy = xy
        virtual_from_phys = pycolmap.Rigid3d(timing.q_iori, np.zeros(3))
        image.frame.rig_from_world = virtual_from_phys * image.cam_from_world()


def gravity_from_grav_stream(
    rec: pycolmap.Reconstruction,
    timings: list[ImageTiming],
    telemetry: GoProTelemetry,
) -> np.ndarray:
    """Mean world gravity direction from the camera's GRAV stream (gravity
    sensor fusion in the physical camera frame), for diagnostics. Expects
    images in their physical camera frames."""
    total = np.zeros(3)
    for timing in timings:
        grav = telemetry.frame_quantity(telemetry.grav_cam, timing.frame_index)
        cam_from_world = rec.images[timing.image_id].cam_from_world()
        total += cam_from_world.rotation.inverse() * (
            grav / np.linalg.norm(grav)
        )
    return total / np.linalg.norm(total)


def mean_reprojection_error(rec: pycolmap.Reconstruction) -> float:
    rec.update_point_3d_errors()
    return rec.compute_mean_reprojection_error()


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# -----------------------------------------------------------------------------
# Visual-inertial optimization.
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
            edge.data
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


def solve_with_reintegration(
    problem: pyceres.Problem,
    solver_options: pyceres.SolverOptions,
    edges: list[ImuEdge],
    variables: InertialVariables,
) -> pyceres.SolverSummary:
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


def initialize_inertial_variables(
    rec: pycolmap.Reconstruction,
    timings: list[ImageTiming],
    edges: list[ImuEdge],
) -> InertialVariables:
    """Inertial-only optimization with the visual poses held constant.

    Solves for the metric scale, gravity direction, velocities, and gyroscope
    bias with the same IMU cost function as the visual-inertial bundle
    adjustment (accelerometer bias fixed at zero), starting from:
      - velocities from finite differences of the camera centers,
      - gravity from the preintegrated velocity changes: summed over the
        sequence, v_end - v_start - g * T = sum_ij R_i * delta_v_ij, where the
        velocity change is negligible compared to g * T,
      - unit scale.
    """
    timing_of = {t.image_id: t for t in timings}
    imu_from_cam = pycolmap.Rigid3d()
    imu_states = {}
    image_ids = sorted(
        {e.image_id1 for e in edges} | {e.image_id2 for e in edges},
        key=lambda i: timing_of[i].frame_index,
    )
    for k, image_id in enumerate(image_ids):
        prev = image_ids[max(k - 1, 0)]
        next_ = image_ids[min(k + 1, len(image_ids) - 1)]
        dt = timing_of[next_].timestamp_s - timing_of[prev].timestamp_s
        state = pycolmap.ImuState()
        state.velocity = (
            rec.images[next_].projection_center()
            - rec.images[prev].projection_center()
        ) / dt
        imu_states[image_id] = state

    imu_from_cam_rotation = imu_from_cam.rotation
    sum_delta_v = np.zeros(3)
    for edge in edges:
        world_from_imu = (
            rec.images[edge.image_id1].cam_from_world().rotation.inverse()
            * imu_from_cam_rotation.inverse()
        )
        sum_delta_v += world_from_imu * edge.data.delta_v
    variables = InertialVariables(
        log_scale=np.zeros(1),
        gravity=-sum_delta_v / np.linalg.norm(sum_delta_v),
        imu_from_cam=imu_from_cam,
        imu_states=imu_states,
    )

    problem = pyceres.Problem()
    add_imu_residuals(problem, rec, edges, variables, False)
    for image_id in image_ids:
        problem.set_parameter_block_constant(
            rec.images[image_id].frame.rig_from_world.params
        )
    for state in imu_states.values():
        problem.set_manifold(state.params, pyceres.SubsetManifold(9, [6, 7, 8]))
    solver_options = pyceres.SolverOptions()
    solver_options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    solver_options.max_num_iterations = 100
    solve_with_reintegration(problem, solver_options, edges, variables)
    return variables


def solve_visual_inertial_bundle_adjustment(
    rec: pycolmap.Reconstruction,
    ba_options: pycolmap.BundleAdjustmentOptions,
    edges: list[ImuEdge],
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
        problem, rec, edges, variables, refine_imu_from_cam_rotation
    )
    solver_options = pyceres.SolverOptions(
        ba_options.ceres.create_solver_options(ba_config, problem)
    )
    return solve_with_reintegration(problem, solver_options, edges, variables)


def refine(
    rec: pycolmap.Reconstruction,
    database_path: Path | None,
    edges: list[ImuEdge],
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
    # Keep the intrinsics of the input reconstruction. The keypoints were
    # mapped to the physical cameras with these intrinsics: refining them
    # would make the mapping inconsistent and, through it, let the focal
    # length absorb rotational disagreements between IMU and vision.
    ba_options.refine_focal_length = False
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = False

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
    """Similarity from the SfM world to a metric world with gravity along +Y
    and the first camera at the origin."""
    scale = float(np.exp(variables.log_scale[0]))
    gravity = variables.gravity / np.linalg.norm(variables.gravity)
    rotation = pycolmap.gravity_aligned_rotation(gravity).inverse()
    center = rec.images[timings[0].image_id].projection_center()
    return pycolmap.Sim3d(scale, rotation, -scale * (rotation * center))


def write_inertial_states(
    path: Path,
    rec: pycolmap.Reconstruction,
    timings: list[ImageTiming],
    variables: InertialVariables,
    metric_from_sfm: pycolmap.Sim3d,
    gravity_vs_grav_deg: float,
) -> None:
    scale = float(np.exp(variables.log_scale[0]))
    states: dict[str, Any] = {
        "metric_scale": scale,
        "gravity_direction_in_world": [0.0, 1.0, 0.0],
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
            velocity = metric_from_sfm.rotation * (scale * state.velocity)
            entry["velocity_m_s"] = velocity.tolist()
            entry["bias_gyro_rad_s"] = np.asarray(state.bias_gyro).tolist()
            entry["bias_accel_m_s2"] = np.asarray(state.bias_accel).tolist()
        states["images"].append(entry)
    path.write_text(json.dumps(states, indent=1) + "\n")


def run(
    sfm_path: Path,
    telemetry_path: Path,
    output_path: Path,
    database_path: Path | None = None,
    time_offset_s: float = DEFAULT_TIME_OFFSET_S,
    max_edge_duration_s: float = 1.0,
    refine_imu_from_cam_rotation: bool = True,
) -> pycolmap.Reconstruction:
    output_path.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(sfm_path))
    # Bring the arbitrary SfM scale to a well-conditioned range; the output is
    # metric anyway.
    rec.normalize()
    telemetry = load_gopro_telemetry(telemetry_path)
    logging.info(
        f"Telemetry: {len(telemetry.gyro_timestamps_s)} IMU samples, "
        f"{len(telemetry.iori_quats_wxyz)} frames at "
        f"{telemetry.video_fps:.3f} fps"
    )
    timings = collect_image_timings(rec, telemetry)
    edges = build_imu_edges(
        timings,
        telemetry,
        gopro_imu_calibration(),
        time_offset_s,
        max_edge_duration_s,
    )
    logging.info(f"{len(edges)} IMU edges between {len(timings)} images")
    if not edges:
        raise RuntimeError("No IMU edges")

    original_keypoints = undo_stabilization(rec, timings)
    variables = initialize_inertial_variables(rec, timings, edges)
    grav_ref = gravity_from_grav_stream(rec, timings, telemetry)
    logging.info(
        f"Initial scale={np.exp(variables.log_scale[0]):.4f}, gravity vs "
        f"GoPro GRAV: {angle_deg(variables.gravity, grav_ref):.2f} deg"
    )

    rec = refine(
        rec,
        database_path,
        edges,
        variables,
        refine_imu_from_cam_rotation,
    )

    grav_ref = gravity_from_grav_stream(rec, timings, telemetry)
    gravity_vs_grav = angle_deg(variables.gravity, grav_ref)
    logging.info(
        f"Final scale={np.exp(variables.log_scale[0]):.4f}, gravity vs GoPro "
        f"GRAV: {gravity_vs_grav:.2f} deg, camera-IMU rotation: "
        f"{np.degrees(variables.imu_from_cam.rotation.angle()):.3f} deg, "
        "mean reprojection error: "
        f"{mean_reprojection_error(rec):.3f} px"
    )
    redo_stabilization(rec, timings, original_keypoints)
    logging.info(
        "Mean reprojection error in the stabilized frames: "
        f"{mean_reprojection_error(rec):.3f} px"
    )
    metric_from_sfm = metric_gravity_aligned_from_sfm(rec, timings, variables)
    write_inertial_states(
        output_path / "inertial_states.json",
        rec,
        timings,
        variables,
        metric_from_sfm,
        gravity_vs_grav,
    )
    rec.transform(metric_from_sfm)
    rec.write(str(output_path))
    logging.info(f"Wrote the metric, gravity-aligned model to {output_path}")
    return rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--sfm_path", type=Path, required=True)
    parser.add_argument(
        "--telemetry_path",
        type=Path,
        required=True,
        help="GoPro video (.mp4/.mov), raw GPMF dump (.bin), or .npz archive.",
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--database_path",
        type=Path,
        help="Database of the reconstruction. If given, tracks are completed, "
        "merged, and re-triangulated between bundle adjustment iterations.",
    )
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
    run(
        sfm_path=args.sfm_path,
        telemetry_path=args.telemetry_path,
        output_path=args.output_path,
        database_path=args.database_path,
        time_offset_s=args.time_offset_s,
        max_edge_duration_s=args.max_edge_duration_s,
        refine_imu_from_cam_rotation=not args.fix_imu_from_cam,
    )
