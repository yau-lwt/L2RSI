"""Spatial-Temporal Particle Estimation (STPE).

This module implements a self-contained reference version of the STPE
algorithm described in the L2RSI project.  The estimator fuses sequential
place-recognition queries (obtained from LiDAR-to-remote-sensing retrieval)
by treating each retrieved location hypothesis as a measurement update for a
particle filter that runs on top of a sparse spatial graph representing the
remote-sensing map.

The implementation is written as pure Python so that it can be executed in
isolation from the original web application.  It exposes high-level classes
for integrating STPE into other pipelines:

* :class:`SpatialTemporalParticleEstimator` — Maintains the particle set and
  exposes a single :meth:`step` method to integrate each new query.  The
  estimator can be reset with a prior distribution, and the returned result
  contains both the best state estimate and the normalized particle cloud.
* :class:`QueryObservation` — Represents a batch of retrieval candidates for a
  single LiDAR submap query.  Each candidate is defined by a remote-sensing
  map node identifier, an unnormalized similarity score, and the timestamp of
  the query.
* :class:`MapNode` — Describes a node of the remote-sensing map (position and
  neighborhood relationship).  The relationships form the spatial constraints
  that STPE exploits during motion prediction.

The overall algorithm follows the typical particle-filter pipeline:

1. **Initialization** — Particles are drawn from a user-provided prior over
   map nodes, e.g., the retrieval distribution of the first query.
2. **Prediction** — When the next query arrives, particles are propagated
   through the spatial graph while penalizing long transitions.  A temporal
   decay term makes older particles progressively less confident so the filter
   can adapt to scene changes.
3. **Measurement update** — Retrieval scores are converted into likelihoods
   using a softmax temperature and then projected onto the particle set.  Each
   particle receives a likelihood proportional to its distance from the
   retrieved candidates.
4. **Resampling & estimation** — Once the effective number of particles drops
   below a threshold, systematic resampling is performed to keep numerical
   stability.  The final state estimate is given by the weighted centroid and
   by the most likely map node.

The code contains detailed comments so it can serve both as documentation and
as a template implementation for the STPE component of L2RSI.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from random import Random
from typing import Dict, List, Sequence, Tuple


Position = Tuple[float, float]


@dataclass(frozen=True)
class MapNode:
    """Node of the remote-sensing map.

    Attributes
    ----------
    node_id:
        Unique identifier of the map node.  In L2RSI this typically matches
        the filename or database key of a remote-sensing submap.
    xy:
        2-D planar coordinates (meters) of the node center.  They are used to
        measure Euclidean distances between retrieval hypotheses.
    neighbors:
        Identifiers of spatially adjacent nodes.  The neighbors define the
        graph over which the motion model propagates the particles.
    """

    node_id: str
    xy: Position
    neighbors: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalCandidate:
    """Single retrieval hypothesis produced by the global matcher."""

    node_id: str
    similarity: float


@dataclass(frozen=True)
class QueryObservation:
    """Sequential query fed into the STPE filter.

    Attributes
    ----------
    timestamp:
        Timestamp (seconds) of the LiDAR submap query.
    candidates:
        Ordered list of :class:`RetrievalCandidate` objects.  The order should
        reflect the ranking provided by the matcher (top-1 first).  Similarity
        scores can be any real values; they are converted into probabilities
        internally via a temperature-controlled softmax.
    """

    timestamp: float
    candidates: Sequence[RetrievalCandidate]


@dataclass
class Particle:
    """Internal representation of a particle."""

    node_id: str
    weight: float


@dataclass
class STPEResult:
    """Return type of :meth:`SpatialTemporalParticleEstimator.step`."""

    timestamp: float
    best_node_id: str
    position_estimate: Position
    particles: Tuple[Particle, ...]


class SpatialTemporalParticleEstimator:
    """Reference implementation of the STPE filter.

    Parameters
    ----------
    map_nodes:
        Dictionary that maps node identifiers to :class:`MapNode` objects.
    num_particles:
        Number of particles to maintain.  Higher values lead to more precise
        distributions at the cost of computation.
    spatial_sigma:
        Controls how quickly the measurement likelihood decays with spatial
        distance.  The value is expressed in meters.
    temporal_decay:
        Per-second exponential decay applied to particle weights between
        queries.  Values in ``(0, 1]`` are valid, where ``1`` means that past
        particles keep their weight indefinitely.
    resample_ratio:
        Threshold on the effective number of particles.  When the ratio
        ``N_eff / num_particles`` falls below this value, the particle set is
        resampled to avoid degeneracy.
    temperature:
        Softmax temperature used to normalize retrieval similarities into a
        probability distribution.  Lower values sharpen the distribution while
        higher values smooth it.
    rng:
        Optional random number generator to make the estimator deterministic
        (useful for unit tests and reproducibility).
    """

    def __init__(
        self,
        map_nodes: Dict[str, MapNode],
        num_particles: int = 256,
        spatial_sigma: float = 30.0,
        temporal_decay: float = 0.98,
        resample_ratio: float = 0.5,
        temperature: float = 0.07,
        rng: Random | None = None,
    ) -> None:
        if not map_nodes:
            raise ValueError("map_nodes must not be empty")
        if num_particles <= 0:
            raise ValueError("num_particles must be positive")
        self.map_nodes = map_nodes
        self.num_particles = num_particles
        self.spatial_sigma = spatial_sigma
        self.temporal_decay = temporal_decay
        self.resample_ratio = resample_ratio
        self.temperature = temperature
        self.rng = rng or Random()

        self._particles: List[Particle] = []
        self._last_timestamp: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, prior: Dict[str, float]) -> None:
        """Initializes the particle set from a discrete prior distribution."""

        if not prior:
            raise ValueError("prior must contain at least one entry")
        normalized = self._normalize(prior)
        node_ids = tuple(normalized.keys())
        weights = tuple(normalized.values())

        self._particles = []
        for _ in range(self.num_particles):
            chosen = self._sample_discrete(node_ids, weights)
            self._particles.append(Particle(node_id=chosen, weight=1.0 / self.num_particles))
        self._last_timestamp = None

    def step(self, observation: QueryObservation) -> STPEResult:
        """Processes a new observation and returns the current estimate."""

        if not self._particles:
            raise RuntimeError("The filter has not been initialized. Call reset() first.")
        if not observation.candidates:
            raise ValueError("observation must contain at least one candidate")

        dt = 0.0
        if self._last_timestamp is not None:
            dt = max(0.0, observation.timestamp - self._last_timestamp)

        self._predict(dt)
        self._update(observation)
        if self._should_resample():
            self._resample()

        self._last_timestamp = observation.timestamp
        best_node_id, position = self._estimate_state()
        return STPEResult(
            timestamp=observation.timestamp,
            best_node_id=best_node_id,
            position_estimate=position,
            particles=tuple(self._particles),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _predict(self, dt: float) -> None:
        if dt <= 0.0:
            return

        decay = self.temporal_decay ** dt
        for particle in self._particles:
            particle.weight *= decay
            node = self.map_nodes[particle.node_id]
            if not node.neighbors:
                continue
            # Favor transitions to nearby neighbors by penalizing large jumps.
            candidates = [self.map_nodes[n_id] for n_id in node.neighbors]
            distances = [self._distance(node.xy, neigh.xy) for neigh in candidates]
            # Convert distances into transition probabilities using a Gaussian kernel.
            transitions = [exp(-(d ** 2) / (2 * self.spatial_sigma ** 2)) for d in distances]
            total = sum(transitions) or 1.0
            probabilities = [t / total for t in transitions]
            idx = self._sample_discrete(range(len(candidates)), probabilities)
            particle.node_id = candidates[idx].node_id

    def _update(self, observation: QueryObservation) -> None:
        measurement = self._softmax(observation.candidates)
        for particle in self._particles:
            node = self.map_nodes[particle.node_id]
            # Spatial compatibility: the closer the particle to a retrieved node,
            # the higher its measurement likelihood.
            likelihood = 0.0
            for cand, prob in measurement.items():
                candidate_node = self.map_nodes[cand]
                d = self._distance(node.xy, candidate_node.xy)
                spatial_term = exp(-(d ** 2) / (2 * self.spatial_sigma ** 2))
                likelihood += prob * spatial_term
            particle.weight *= likelihood

        self._normalize_particle_weights()

    def _should_resample(self) -> bool:
        weights = [p.weight for p in self._particles]
        weight_sum = sum(weights)
        if weight_sum == 0:
            return True
        weights = [w / weight_sum for w in weights]
        neff = 1.0 / sum(w * w for w in weights)
        return neff / self.num_particles < self.resample_ratio

    def _resample(self) -> None:
        weights = [p.weight for p in self._particles]
        cumulative = []
        total = 0.0
        for w in weights:
            total += w
            cumulative.append(total)
        if total == 0.0:
            # Fall back to uniform weights when numerical issues occur.
            cumulative = [i / len(weights) for i in range(1, len(weights) + 1)]
            total = 1.0
        step = total / self.num_particles
        u0 = self.rng.uniform(0.0, step)
        new_particles: List[Particle] = []
        i = 0
        for j in range(self.num_particles):
            uj = u0 + j * step
            while uj > cumulative[i]:
                i += 1
            new_particles.append(Particle(node_id=self._particles[i].node_id, weight=1.0 / self.num_particles))
        self._particles = new_particles

    def _estimate_state(self) -> Tuple[str, Position]:
        best_particle = max(self._particles, key=lambda p: p.weight)
        weighted_sum_x = 0.0
        weighted_sum_y = 0.0
        total_weight = 0.0
        for particle in self._particles:
            node = self.map_nodes[particle.node_id]
            weighted_sum_x += node.xy[0] * particle.weight
            weighted_sum_y += node.xy[1] * particle.weight
            total_weight += particle.weight
        if total_weight == 0.0:
            total_weight = 1.0
        centroid = (weighted_sum_x / total_weight, weighted_sum_y / total_weight)
        return best_particle.node_id, centroid

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def _normalize(self, distribution: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, v) for v in distribution.values())
        if total == 0.0:
            raise ValueError("distribution must contain at least one positive value")
        return {key: max(0.0, value) / total for key, value in distribution.items()}

    def _softmax(self, candidates: Sequence[RetrievalCandidate]) -> Dict[str, float]:
        scaled = [cand.similarity / self.temperature for cand in candidates]
        max_val = max(scaled)
        exps = [exp(val - max_val) for val in scaled]
        total = sum(exps)
        return {cand.node_id: e / total for cand, e in zip(candidates, exps)}

    def _normalize_particle_weights(self) -> None:
        total = sum(p.weight for p in self._particles)
        if total == 0.0:
            # Reset to uniform weights to avoid degeneracy.  The subsequent
            # resampling step will quickly rebuild a meaningful distribution.
            for particle in self._particles:
                particle.weight = 1.0 / self.num_particles
            return
        for particle in self._particles:
            particle.weight /= total

    def _sample_discrete(self, items: Sequence, probs: Sequence[float]):
        cumulative = 0.0
        r = self.rng.random()
        for item, prob in zip(items, probs):
            cumulative += prob
            if r <= cumulative:
                return item
        return items[-1]

    @staticmethod
    def _distance(a: Position, b: Position) -> float:
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic map consisting of four connected nodes forming a loop.
    nodes = {
        "A": MapNode("A", (0.0, 0.0), ("B", "D")),
        "B": MapNode("B", (50.0, 0.0), ("A", "C")),
        "C": MapNode("C", (50.0, 50.0), ("B", "D")),
        "D": MapNode("D", (0.0, 50.0), ("A", "C")),
    }

    stpe = SpatialTemporalParticleEstimator(nodes, num_particles=128, spatial_sigma=30.0)
    # Initialize the filter with a prior that strongly prefers node A.
    stpe.reset({"A": 0.8, "B": 0.2})

    observations = [
        QueryObservation(
            timestamp=0.0,
            candidates=[
                RetrievalCandidate("A", similarity=0.9),
                RetrievalCandidate("B", similarity=0.5),
            ],
        ),
        QueryObservation(
            timestamp=1.0,
            candidates=[
                RetrievalCandidate("B", similarity=0.9),
                RetrievalCandidate("C", similarity=0.6),
            ],
        ),
        QueryObservation(
            timestamp=2.0,
            candidates=[
                RetrievalCandidate("C", similarity=1.0),
                RetrievalCandidate("D", similarity=0.4),
            ],
        ),
    ]

    for obs in observations:
        result = stpe.step(obs)
        print(
            f"t={result.timestamp:.1f}s -> node={result.best_node_id} "
            f"position=({result.position_estimate[0]:.1f}, {result.position_estimate[1]:.1f})"
        )
