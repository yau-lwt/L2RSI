"""Spatial-Temporal Particle Estimation (STPE).

This module implements the STEP/STPE algorithm described in the repository
materials.  The estimator replaces the rejection-based update strategy of a
classic particle filter with particle aggregation.  Instead of discarding
particles that are inconsistent with the most recent observation, STPE keeps a
sliding window of Gaussian mixtures that summarise historical evidence.  The
historical distributions are propagated to the current frame and averaged,
which smooths out the influence of sporadic failures ("toxic queries").

Key stages implemented below
============================
1. Particle initialisation from retrieval scores.
2. DBSCAN clustering of the top-K particles followed by Gaussian modelling to
   form a Gaussian Mixture Model (GMM).
3. Spatio-temporal propagation that uses relative poses (FastGICP) and
   magnetometer heading correction every 20 m.
4. Probability estimation by averaging the historical GMMs.
5. Sliding window maintenance (250 m, 30 % sampling) with dynamic parameter
   updates.

The code is organised into dataclasses that describe particles and Gaussian
components as well as a high-level ``SpatialTemporalParticleEstimator`` class
that exposes a single ``update`` method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import math
import numpy as np


@dataclass(frozen=True)
class Particle:
    """A single particle state.

    Attributes
    ----------
    position:
        Two dimensional coordinates (x, y) of the particle in the remote
        sensing map.
    descriptor:
        The descriptor associated with the candidate submap.
    score:
        Retrieval similarity score for the particle (higher is better).
    """

    position: np.ndarray
    descriptor: np.ndarray
    score: float


@dataclass
class GaussianComponent:
    """Gaussian component of a mixture model."""

    weight: float
    mean: np.ndarray
    covariance: np.ndarray

    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        """Apply a rigid transformation to the Gaussian component.

        Parameters
        ----------
        rotation:
            ``2 x 2`` rotation matrix.
        translation:
            Length-2 translation vector.
        """

        self.mean = rotation @ self.mean + translation
        self.covariance = rotation @ self.covariance @ rotation.T

    def rotate(self, rotation: np.ndarray) -> None:
        """Rotate the mean and covariance around the origin."""

        self.mean = rotation @ self.mean
        self.covariance = rotation @ self.covariance @ rotation.T

    def probability(self, point: np.ndarray) -> float:
        """Evaluate the Gaussian probability density at ``point``."""

        diff = point - self.mean
        cov = self.covariance
        det_cov = np.linalg.det(cov)
        if det_cov <= 0:
            # Regularise degenerate covariances.
            cov = cov + np.eye(2) * 1e-6
            det_cov = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)
        normalisation = 1.0 / (2.0 * math.pi * math.sqrt(det_cov))
        exponent = -0.5 * float(diff.T @ inv_cov @ diff)
        return self.weight * normalisation * math.exp(exponent)


@dataclass
class GaussianMixture:
    """Gaussian mixture model comprising multiple components."""

    components: List[GaussianComponent] = field(default_factory=list)

    def probability(self, point: np.ndarray) -> float:
        return sum(component.probability(point) for component in self.components)

    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        for component in self.components:
            component.transform(rotation, translation)

    def rotate(self, rotation: np.ndarray) -> None:
        for component in self.components:
            component.rotate(rotation)

    def copy(self) -> "GaussianMixture":
        components = [
            GaussianComponent(component.weight, component.mean.copy(), component.covariance.copy())
            for component in self.components
        ]
        return GaussianMixture(components)

    @property
    def total_weight(self) -> float:
        return sum(component.weight for component in self.components)


@dataclass
class HistoricalState:
    """History item maintained by STPE."""

    gmm: GaussianMixture
    distance_from_current: float = 0.0


class SpatialTemporalParticleEstimator:
    """Implementation of the STPE algorithm.

    Parameters
    ----------
    top_c:
        Number of highest scoring retrieval candidates used for particle
        initialisation.
    top_k:
        Subset of particles used to build the GMM through DBSCAN clustering.
    cluster_radius:
        DBSCAN neighbourhood radius (metres).
    history_length:
        Maximum number of query steps retained in the sliding window.
    sliding_window_m:
        Maximum spatial distance covered by the sliding window (metres).
    sampling_rate:
        Ratio of top-C candidates retained to reduce the computational load.
    magnetometer_interval:
        Distance between magnetometer based heading corrections (metres).
    min_cluster_size:
        Minimum number of particles required to form a cluster.
    """

    def __init__(
        self,
        top_c: int = 200,
        top_k: int = 50,
        cluster_radius: float = 30.0,
        history_length: int = 50,
        sliding_window_m: float = 250.0,
        sampling_rate: float = 0.3,
        magnetometer_interval: float = 20.0,
        min_cluster_size: int = 1,
    ) -> None:
        self.top_c = top_c
        self.top_k = top_k
        self.cluster_radius = cluster_radius
        self.history_length = history_length
        self.sliding_window_m = sliding_window_m
        self.sampling_rate = sampling_rate
        self.magnetometer_interval = magnetometer_interval
        self.min_cluster_size = min_cluster_size
        self.history: List[HistoricalState] = []
        self.distance_since_correction = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        query_descriptor: np.ndarray,
        db_descriptors: np.ndarray,
        candidate_positions: np.ndarray,
        relative_pose: Optional[np.ndarray],
        magnetometer_heading: Optional[float],
        distance_travelled: float,
    ) -> Tuple[List[int], np.ndarray]:
        """Process a new query observation.

        Returns
        -------
        ranking : list of int
            Indices of the database submaps sorted by the aggregated STPE
            probability.
        scores : ndarray
            Probability scores associated with the ranking.
        """

        particles = self._initialise_particles(
            query_descriptor, db_descriptors, candidate_positions
        )
        gmm = self._build_gmm(particles)

        # Propagate existing history to the new frame.
        if relative_pose is not None:
            rotation, translation = self._decompose_pose(relative_pose)
            self._propagate_history(rotation, translation, distance_travelled, magnetometer_heading)
        else:
            self._increment_history_distances(distance_travelled)
            self._maybe_correct_heading(magnetometer_heading)

        # Add current observation as the newest history element.
        self._append_history(gmm)
        self._enforce_history_limits()

        # Compute final ranking based on averaged probabilities.
        scores = self._estimate_probabilities(candidate_positions)
        ranking = list(np.argsort(scores)[::-1])
        return ranking, scores

    # ------------------------------------------------------------------
    # Particle initialisation and clustering
    # ------------------------------------------------------------------
    def _initialise_particles(
        self,
        query_descriptor: np.ndarray,
        db_descriptors: np.ndarray,
        candidate_positions: np.ndarray,
    ) -> List[Particle]:
        """Initialise particles by retrieving the top-C candidates."""

        if db_descriptors.ndim != 2 or query_descriptor.ndim != 1:
            raise ValueError("Descriptors must be 2D (database) and 1D (query).")

        similarities = self._cosine_similarity(query_descriptor, db_descriptors)
        top_indices = np.argsort(similarities)[-self.top_c :][::-1]

        # Apply sampling to reduce computational cost.
        sample_count = max(1, int(len(top_indices) * self.sampling_rate))
        sampled_indices = top_indices[:sample_count]

        particles = [
            Particle(position=candidate_positions[idx], descriptor=db_descriptors[idx], score=similarities[idx])
            for idx in sampled_indices
        ]
        return particles

    def _build_gmm(self, particles: Sequence[Particle]) -> GaussianMixture:
        """Construct a Gaussian mixture model using DBSCAN clustering."""

        if not particles:
            raise ValueError("At least one particle is required to build a GMM.")

        top_particles = list(particles[: self.top_k])
        positions = np.array([particle.position for particle in top_particles])
        labels = self._dbscan(positions, self.cluster_radius, self.min_cluster_size)

        clusters: List[np.ndarray] = []
        for label in sorted(set(labels)):
            if label == -1:
                continue
            mask = labels == label
            cluster_positions = positions[mask]
            if cluster_positions.shape[0] >= self.min_cluster_size:
                clusters.append(cluster_positions)

        if not clusters:
            # Fallback: treat each particle as a single Gaussian.
            clusters = [positions]

        total_points = sum(cluster.shape[0] for cluster in clusters)
        components = []
        for cluster in clusters:
            weight = cluster.shape[0] / total_points
            mean = np.mean(cluster, axis=0)
            centered = cluster - mean
            covariance = centered.T @ centered / max(cluster.shape[0] - 1, 1)
            covariance += np.eye(2) * 1e-3
            components.append(GaussianComponent(weight, mean, covariance))

        return GaussianMixture(components)

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def _propagate_history(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        distance_travelled: float,
        magnetometer_heading: Optional[float],
    ) -> None:
        for state in self.history:
            state.gmm.transform(rotation, translation)
            state.distance_from_current += distance_travelled

        self.distance_since_correction += distance_travelled
        self._maybe_correct_heading(magnetometer_heading)

    def _increment_history_distances(self, distance_travelled: float) -> None:
        for state in self.history:
            state.distance_from_current += distance_travelled
        self.distance_since_correction += distance_travelled

    def _maybe_correct_heading(self, magnetometer_heading: Optional[float]) -> None:
        if magnetometer_heading is None:
            return
        if self.distance_since_correction < self.magnetometer_interval:
            return

        rotation = self._heading_to_rotation(magnetometer_heading)
        for state in self.history:
            state.gmm.rotate(rotation)
        self.distance_since_correction = 0.0

    def _append_history(self, gmm: GaussianMixture) -> None:
        self.history.insert(0, HistoricalState(gmm.copy(), 0.0))

    def _enforce_history_limits(self) -> None:
        if len(self.history) > self.history_length:
            self.history = self.history[: self.history_length]

        self.history = [
            state for state in self.history if state.distance_from_current <= self.sliding_window_m
        ]

    # ------------------------------------------------------------------
    # Probability estimation and ranking
    # ------------------------------------------------------------------
    def _estimate_probabilities(self, candidate_positions: np.ndarray) -> np.ndarray:
        if not self.history:
            return np.zeros(candidate_positions.shape[0])

        scores = np.zeros(candidate_positions.shape[0])
        history_count = len(self.history)
        for state in self.history:
            for idx, position in enumerate(candidate_positions):
                scores[idx] += state.gmm.probability(position)

        return scores / max(history_count, 1)

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(query: np.ndarray, database: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query)
        db_norms = np.linalg.norm(database, axis=1)
        similarities = database @ query / (db_norms * query_norm + 1e-10)
        return similarities

    @staticmethod
    def _decompose_pose(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if pose.shape == (4, 4):
            rotation = pose[:2, :2]
            translation = pose[:2, 3]
        elif pose.shape == (3, 3):
            rotation = pose[:2, :2]
            translation = pose[:2, 2]
        else:
            raise ValueError("Pose must be a 3x3 or 4x4 homogeneous matrix.")
        return rotation, translation

    @staticmethod
    def _heading_to_rotation(heading_rad: float) -> np.ndarray:
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        return np.array([[cos_h, -sin_h], [sin_h, cos_h]])

    @staticmethod
    def _dbscan(points: np.ndarray, radius: float, min_samples: int) -> np.ndarray:
        if points.size == 0:
            return np.array([], dtype=int)

        visited = np.zeros(points.shape[0], dtype=bool)
        labels = np.full(points.shape[0], -1, dtype=int)
        cluster_id = 0

        for idx in range(points.shape[0]):
            if visited[idx]:
                continue
            visited[idx] = True
            neighbours = SpatialTemporalParticleEstimator._region_query(points, idx, radius)
            if len(neighbours) < min_samples:
                continue

            labels[idx] = cluster_id
            seeds = set(neighbours)
            seeds.discard(idx)

            while seeds:
                current = seeds.pop()
                if not visited[current]:
                    visited[current] = True
                    current_neighbours = SpatialTemporalParticleEstimator._region_query(points, current, radius)
                    if len(current_neighbours) >= min_samples:
                        seeds.update(current_neighbours)
                if labels[current] == -1:
                    labels[current] = cluster_id
            cluster_id += 1

        if cluster_id == 0:
            # All points are treated as noise; assign each to its own cluster.
            labels = np.arange(points.shape[0])
        return labels

    @staticmethod
    def _region_query(points: np.ndarray, idx: int, radius: float) -> List[int]:
        deltas = points - points[idx]
        distances = np.linalg.norm(deltas, axis=1)
        return [int(i) for i, distance in enumerate(distances) if distance <= radius]


__all__ = [
    "Particle",
    "GaussianComponent",
    "GaussianMixture",
    "HistoricalState",
    "SpatialTemporalParticleEstimator",
]
