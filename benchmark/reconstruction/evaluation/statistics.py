# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Statistical inference for reconstruction benchmark scores.

Scenes are the independent benchmark units. Random seeds are crossed repeated
measurements. A/B differences always retain their within-scene, within-seed
pairing. Camera pairs must not be bootstrapped independently because pairs
sharing a camera are strongly dependent.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True)
class SimultaneousInterval:
    estimate: npt.NDArray[np.float64]
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]
    standard_error: npt.NDArray[np.float64]
    adjusted_p_values: npt.NDArray[np.float64]
    confidence: float
    num_bootstrap_samples: int


@dataclasses.dataclass(frozen=True)
class NoiseCeiling:
    scene_keys: tuple[str, ...]
    scores: npt.NDArray[np.float64]
    thresholds: npt.NDArray[np.float64]
    error_type: str
    metadata: dict


def _validate_scores(scores: npt.ArrayLike, name: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError(
            f"{name} must have shape (scenes, replicates, thresholds), "
            f"got {scores.shape}"
        )
    if 0 in scores.shape:
        raise ValueError(f"{name} dimensions must be non-empty")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} must contain only finite scores")
    return scores


def _bootstrap_weights(
    rng: np.random.Generator, num_samples: int, num_units: int
) -> npt.NDArray[np.float64]:
    probabilities = np.full(num_units, 1 / num_units)
    return (
        rng.multinomial(num_units, probabilities, size=num_samples).astype(
            np.float64
        )
        / num_units
    )


def _simultaneous_interval(
    estimate: npt.NDArray[np.float64],
    bootstrap_estimates: npt.NDArray[np.float64],
    confidence: float,
) -> SimultaneousInterval:
    """Studentized max-deviation intervals across all score thresholds."""
    standard_error = np.std(bootstrap_estimates, axis=0, ddof=1)
    active = standard_error > np.finfo(np.float64).eps
    standardized = np.zeros_like(bootstrap_estimates)
    standardized[:, active] = (
        np.abs(bootstrap_estimates[:, active] - estimate[None, active])
        / standard_error[None, active]
    )
    max_statistics = np.max(standardized, axis=1)
    critical_value = np.quantile(max_statistics, confidence)
    half_width = critical_value * standard_error

    adjusted_p_values = np.ones_like(estimate)
    nonzero_estimate = active & (np.abs(estimate) > 0)
    observed_statistics = np.zeros_like(estimate)
    observed_statistics[nonzero_estimate] = (
        np.abs(estimate[nonzero_estimate]) / standard_error[nonzero_estimate]
    )
    for index in np.flatnonzero(nonzero_estimate):
        adjusted_p_values[index] = (
            1 + np.count_nonzero(max_statistics >= observed_statistics[index])
        ) / (len(max_statistics) + 1)
    adjusted_p_values[~active & (np.abs(estimate) > 0)] = 1 / (
        len(max_statistics) + 1
    )

    return SimultaneousInterval(
        estimate=estimate,
        lower=estimate - half_width,
        upper=estimate + half_width,
        standard_error=standard_error,
        adjusted_p_values=adjusted_p_values,
        confidence=confidence,
        num_bootstrap_samples=len(bootstrap_estimates),
    )


def paired_difference_interval(
    scores_a: npt.ArrayLike,
    scores_b: npt.ArrayLike,
    num_bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> SimultaneousInterval:
    """Simultaneous interval for macro scene-average A - B differences.

    Inputs have shape (scenes, seeds, thresholds). Scenes and seeds are
    resampled independently as crossed factors; A/B pairing is retained by
    bootstrapping their difference.
    """
    scores_a = _validate_scores(scores_a, "scores_a")
    scores_b = _validate_scores(scores_b, "scores_b")
    if scores_a.shape != scores_b.shape:
        raise ValueError(
            f"scores_a and scores_b shapes differ: "
            f"{scores_a.shape} vs {scores_b.shape}"
        )
    _validate_inference_options(
        num_bootstrap_samples=num_bootstrap_samples,
        confidence=confidence,
    )

    differences = scores_a - scores_b
    num_scenes, num_seeds, _ = differences.shape
    rng = np.random.default_rng(random_seed)
    scene_weights = _bootstrap_weights(rng, num_bootstrap_samples, num_scenes)
    seed_weights = _bootstrap_weights(rng, num_bootstrap_samples, num_seeds)
    bootstrap_estimates = np.einsum(
        "bs,bk,skt->bt",
        scene_weights,
        seed_weights,
        differences,
        optimize=True,
    )
    return _simultaneous_interval(
        estimate=np.mean(differences, axis=(0, 1)),
        bootstrap_estimates=bootstrap_estimates,
        confidence=confidence,
    )


def ceiling_gap_interval(
    run_scores: npt.ArrayLike,
    ceiling_scores: npt.ArrayLike,
    num_bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> SimultaneousInterval:
    """Simultaneous interval for expected ceiling - run score.

    run_scores has shape (scenes, seeds, thresholds); ceiling_scores has shape
    (scenes, Monte Carlo draws, thresholds). The same bootstrap scene weights
    are applied to both, preserving the paired scene comparison.
    """
    run_scores = _validate_scores(run_scores, "run_scores")
    ceiling_scores = _validate_scores(ceiling_scores, "ceiling_scores")
    if run_scores.shape[0] != ceiling_scores.shape[0]:
        raise ValueError("run and ceiling scene counts differ")
    if run_scores.shape[2] != ceiling_scores.shape[2]:
        raise ValueError("run and ceiling threshold counts differ")
    _validate_inference_options(
        num_bootstrap_samples=num_bootstrap_samples,
        confidence=confidence,
    )

    num_scenes, num_seeds, _ = run_scores.shape
    num_ceiling_draws = ceiling_scores.shape[1]
    rng = np.random.default_rng(random_seed)
    scene_weights = _bootstrap_weights(rng, num_bootstrap_samples, num_scenes)
    seed_weights = _bootstrap_weights(rng, num_bootstrap_samples, num_seeds)
    ceiling_weights = _bootstrap_weights(
        rng, num_bootstrap_samples, num_ceiling_draws
    )
    run_bootstrap = np.einsum(
        "bs,bk,skt->bt",
        scene_weights,
        seed_weights,
        run_scores,
        optimize=True,
    )
    ceiling_bootstrap = np.einsum(
        "bs,bm,smt->bt",
        scene_weights,
        ceiling_weights,
        ceiling_scores,
        optimize=True,
    )
    estimate = np.mean(ceiling_scores, axis=(0, 1)) - np.mean(
        run_scores, axis=(0, 1)
    )
    return _simultaneous_interval(
        estimate=estimate,
        bootstrap_estimates=ceiling_bootstrap - run_bootstrap,
        confidence=confidence,
    )


def classify_difference(
    lower: float, upper: float, minimum_effect: float
) -> str:
    """Classify an A - B interval using a smallest effect of interest."""
    if lower > minimum_effect:
        return "A better"
    if upper < -minimum_effect:
        return "B better"
    if lower >= -minimum_effect and upper <= minimum_effect:
        return "equivalent"
    return "inconclusive"


def classify_ceiling_gap(lower: float, upper: float, margin: float) -> str:
    """Classify a ceiling - run interval using an equivalence margin."""
    if upper < margin:
        return "ceiling-limited"
    if lower > margin:
        return "headroom remains"
    return "inconclusive"


def save_noise_ceiling(path: Path, ceiling: NoiseCeiling) -> None:
    scores = _validate_scores(ceiling.scores, "ceiling.scores")
    if scores.shape[0] != len(ceiling.scene_keys):
        raise ValueError("ceiling scene key and score counts differ")
    if scores.shape[2] != len(ceiling.thresholds):
        raise ValueError("ceiling threshold and score counts differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scene_keys=np.asarray(ceiling.scene_keys),
        scores=scores,
        thresholds=np.asarray(ceiling.thresholds, dtype=np.float64),
        error_type=np.asarray(ceiling.error_type),
        metadata_json=np.asarray(json.dumps(ceiling.metadata, sort_keys=True)),
    )


def load_noise_ceiling(path: Path) -> NoiseCeiling:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "scene_keys",
            "scores",
            "thresholds",
            "error_type",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"noise ceiling artifact is missing fields: {sorted(missing)}"
            )
        ceiling = NoiseCeiling(
            scene_keys=tuple(str(key) for key in data["scene_keys"]),
            scores=np.asarray(data["scores"], dtype=np.float64),
            thresholds=np.asarray(data["thresholds"], dtype=np.float64),
            error_type=str(data["error_type"]),
            metadata=json.loads(str(data["metadata_json"])),
        )
    _validate_scores(ceiling.scores, "ceiling.scores")
    if ceiling.scores.shape[0] != len(ceiling.scene_keys):
        raise ValueError("noise ceiling scene key and score counts differ")
    if ceiling.scores.shape[2] != len(ceiling.thresholds):
        raise ValueError("noise ceiling threshold and score counts differ")
    if len(set(ceiling.scene_keys)) != len(ceiling.scene_keys):
        raise ValueError("noise ceiling scene keys must be unique")
    return ceiling


def _validate_inference_options(
    num_bootstrap_samples: int, confidence: float
) -> None:
    if num_bootstrap_samples < 2:
        raise ValueError("num_bootstrap_samples must be at least 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
