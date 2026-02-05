"""
hippocampus.py — Episodic Memory, Replay, and Consolidation
============================================================

Prioridad: 6

Input:  Cortical activation patterns
Output: Episodic memory, replay sequences, consolidation signals

The hippocampus provides:

  1. PATTERN COMPLETION: Given partial cortical activity, recall the full pattern
     (e.g., seeing a face → recall the name, context, emotions)

  2. REPLAY: During "offline" periods (sleep-like states), replay stored
     patterns to the cortex → drives consolidation via STDP

  3. TEMPORAL INDEXING: Associates patterns with temporal context
     (what happened when, in what order)

  4. NOVELTY DETECTION: Identifies new patterns that don't match stored ones
     → triggers noradrenaline burst (orientation response)

Implementation based on complementary learning systems theory:
  - Hippocampus: fast learning, sparse coding, episodic
  - Cortex: slow learning, distributed, semantic
  - Transfer: hippocampal replay drives cortical consolidation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import hashlib


@dataclass
class Episode:
    """A stored episodic memory."""
    pattern: np.ndarray           # compressed cortical activation pattern
    timestamp: float              # when it was stored (simulation time)
    salience: float               # how important/novel it was (0-1)
    replay_count: int = 0         # how many times it's been replayed
    last_replay: float = 0.0      # last replay timestamp
    context_hash: str = ""        # hash for fast lookup
    neuromod_state: Dict = field(default_factory=dict)  # state when encoded
    
    def decay_salience(self, dt: float, tau: float = 50000.0):
        """Exponential decay of salience over time."""
        self.salience *= np.exp(-dt / tau)


@dataclass
class HippocampusConfig:
    """Configuration for hippocampal system."""
    
    # Capacity
    max_episodes: int = 10000     # max stored episodes
    pattern_dim: int = 256        # compressed pattern dimensionality
    
    # Encoding thresholds
    novelty_threshold: float = 0.3    # minimum novelty to encode
    salience_threshold: float = 0.2   # minimum salience to keep
    
    # Replay
    replay_rate: float = 10.0     # replays per second during offline
    replay_sequence_length: int = 5   # episodes per replay sequence
    replay_time_scale: float = 20.0   # compression ratio (20x faster)
    
    # Pattern completion
    completion_threshold: float = 0.5  # similarity needed for completion
    sparse_coding_k: int = 32         # number of active units (sparse)
    
    # Consolidation
    consolidation_interval_ms: float = 10000.0  # time between consolidations
    consolidation_strength: float = 0.5


class Hippocampus:
    """
    Hippocampal memory system.
    
    Stores episodic memories as compressed cortical activation patterns.
    Provides pattern completion, novelty detection, and offline replay
    for memory consolidation.
    
    Usage:
        hippo = Hippocampus(n_cortical=100_000)
        
        # During active processing:
        cortical_pattern = engine.get_firing_rates()
        novelty = hippo.encode(cortical_pattern, timestamp=engine.state.time_ms)
        
        # Pattern completion:
        recalled = hippo.recall(partial_pattern)
        
        # During offline/sleep:
        replay_patterns = hippo.generate_replay_sequence()
        for pattern in replay_patterns:
            engine.inject_pattern(pattern)
    """
    
    def __init__(
        self,
        n_cortical: int,
        config: Optional[HippocampusConfig] = None,
    ):
        self.config = config or HippocampusConfig()
        self.n_cortical = n_cortical
        
        # Episode storage
        self.episodes: List[Episode] = []
        
        # Random projection matrix for dimensionality reduction
        # Cortical pattern (n_cortical) → compressed (pattern_dim)
        rng = np.random.default_rng(42)
        self.projection_matrix = rng.standard_normal(
            (self.config.pattern_dim, n_cortical)
        ).astype(np.float32)
        self.projection_matrix /= np.sqrt(n_cortical)  # normalize
        
        # Inverse projection for pattern completion
        # (pseudo-inverse, allows reconstruction from compressed)
        self.reconstruction_matrix = np.linalg.pinv(self.projection_matrix).astype(np.float32)
        
        # Novelty tracking
        self._recent_patterns = deque(maxlen=100)
        
        # Consolidation state
        self._last_consolidation = 0.0
        self._replay_buffer: List[Episode] = []
        
        print(f"[Hippocampus] Initialized:")
        print(f"  Pattern dim: {n_cortical:,} → {self.config.pattern_dim}")
        print(f"  Max episodes: {self.config.max_episodes:,}")
        print(f"  Sparse coding k: {self.config.sparse_coding_k}")
    
    def encode(
        self,
        cortical_pattern: np.ndarray,
        timestamp: float,
        neuromod_state: Optional[Dict] = None,
    ) -> float:
        """
        Attempt to encode a cortical activation pattern as an episode.
        
        Only encodes if the pattern is sufficiently novel (doesn't match
        existing episodes closely).
        
        Args:
            cortical_pattern: Firing rates per cortical neuron
            timestamp: Current simulation time in ms
            neuromod_state: Current neuromodulator levels
            
        Returns:
            novelty: Novelty score (0-1). High = very novel pattern.
        """
        # Compress pattern
        compressed = self._compress(cortical_pattern)
        
        # Compute novelty (how different from stored episodes)
        novelty = self._compute_novelty(compressed)
        
        # Only encode if novel enough
        if novelty >= self.config.novelty_threshold:
            # Compute salience (novelty + dopamine level = importance)
            da_level = neuromod_state.get("dopamine", 1.0) if neuromod_state else 1.0
            salience = novelty * da_level
            
            episode = Episode(
                pattern=compressed,
                timestamp=timestamp,
                salience=salience,
                context_hash=self._hash_pattern(compressed),
                neuromod_state=neuromod_state or {},
            )
            
            self.episodes.append(episode)
            self._recent_patterns.append(compressed)
            
            # Evict old low-salience episodes if at capacity
            if len(self.episodes) > self.config.max_episodes:
                self._evict_least_salient()
        
        return novelty
    
    def recall(
        self,
        partial_pattern: np.ndarray,
        top_k: int = 1,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Pattern completion: given a partial cortical pattern, find the
        closest stored episode and reconstruct the full pattern.
        
        Args:
            partial_pattern: Incomplete cortical activation
            top_k: Number of best matches to return
            
        Returns:
            List of (reconstructed_pattern, similarity_score) tuples
        """
        if not self.episodes:
            return []
        
        # Compress the query
        query = self._compress(partial_pattern)
        
        # Find most similar stored episodes
        similarities = []
        for ep in self.episodes:
            sim = self._cosine_similarity(query, ep.pattern)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Get top-k matches above threshold
        above_threshold = similarities >= self.config.completion_threshold
        if not np.any(above_threshold):
            return []
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.config.completion_threshold:
                # Reconstruct full pattern from compressed
                reconstructed = self._decompress(self.episodes[idx].pattern)
                results.append((reconstructed, float(similarities[idx])))
        
        return results
    
    def generate_replay_sequence(
        self,
        n_episodes: Optional[int] = None,
        mode: str = "priority",
    ) -> List[np.ndarray]:
        """
        Generate a sequence of patterns for offline replay.
        
        During "sleep" or low-activity states, the hippocampus replays
        stored episodes to the cortex. This drives STDP-based consolidation
        of memories into long-term cortical storage.
        
        Args:
            n_episodes: Number of episodes to replay (default: config)
            mode: "priority" (salience-weighted), "sequential" (temporal),
                  "random" (uniform)
                  
        Returns:
            List of decompressed cortical patterns to inject
        """
        if not self.episodes:
            return []
        
        n = n_episodes or self.config.replay_sequence_length
        n = min(n, len(self.episodes))
        
        if mode == "priority":
            # Sample weighted by salience (more important → more replay)
            saliences = np.array([ep.salience for ep in self.episodes])
            saliences = saliences / (np.sum(saliences) + 1e-8)
            indices = np.random.choice(len(self.episodes), size=n, replace=False, p=saliences)
            
        elif mode == "sequential":
            # Replay most recent episodes in temporal order
            indices = list(range(max(0, len(self.episodes) - n), len(self.episodes)))
            
        elif mode == "random":
            indices = np.random.choice(len(self.episodes), size=n, replace=False)
        
        else:
            raise ValueError(f"Unknown replay mode: {mode}")
        
        # Generate replay patterns
        patterns = []
        for idx in indices:
            ep = self.episodes[idx]
            pattern = self._decompress(ep.pattern)
            
            # Apply temporal compression (faster replay)
            # In biology, replay is 5-20x faster than real-time
            patterns.append(pattern * self.config.replay_time_scale)
            
            # Update replay count
            ep.replay_count += 1
            ep.last_replay = self.episodes[-1].timestamp if self.episodes else 0
        
        return patterns
    
    def detect_novelty(self, cortical_pattern: np.ndarray) -> float:
        """
        Quick novelty detection without encoding.
        
        Used to trigger noradrenaline bursts in the neuromodulation system.
        """
        compressed = self._compress(cortical_pattern)
        return self._compute_novelty(compressed)
    
    def consolidate(self, current_time: float):
        """
        Run periodic consolidation: decay salience, prune old episodes.
        
        Should be called periodically (e.g., every 10 seconds of sim time).
        """
        if current_time - self._last_consolidation < self.config.consolidation_interval_ms:
            return
        
        dt = current_time - self._last_consolidation
        
        # Decay salience of all episodes
        for ep in self.episodes:
            ep.decay_salience(dt)
        
        # Remove episodes below salience threshold
        n_before = len(self.episodes)
        self.episodes = [
            ep for ep in self.episodes 
            if ep.salience >= self.config.salience_threshold
        ]
        n_removed = n_before - len(self.episodes)
        
        if n_removed > 0:
            print(f"  [Hippocampus] Consolidated: removed {n_removed} episodes, "
                  f"{len(self.episodes)} remaining")
        
        self._last_consolidation = current_time
    
    def _compress(self, cortical_pattern: np.ndarray) -> np.ndarray:
        """
        Compress cortical pattern using random projection + sparse coding.
        
        1. Random projection: n_cortical → pattern_dim
        2. Sparse coding: keep only top-k activations
        """
        # Random projection
        compressed = self.projection_matrix @ cortical_pattern.astype(np.float32)
        
        # Sparse coding: zero out all but top-k values
        k = self.config.sparse_coding_k
        if len(compressed) > k:
            threshold = np.sort(np.abs(compressed))[-k]
            compressed[np.abs(compressed) < threshold] = 0.0
        
        # Normalize
        norm = np.linalg.norm(compressed)
        if norm > 0:
            compressed /= norm
        
        return compressed
    
    def _decompress(self, compressed: np.ndarray) -> np.ndarray:
        """Reconstruct cortical pattern from compressed representation."""
        reconstructed = self.reconstruction_matrix @ compressed
        
        # ReLU: firing rates can't be negative
        reconstructed = np.maximum(reconstructed, 0)
        
        return reconstructed
    
    def _compute_novelty(self, compressed: np.ndarray) -> float:
        """
        Compute novelty of a pattern relative to stored episodes.
        
        Novelty = 1 - max_similarity_to_stored_episodes
        """
        if not self.episodes and not self._recent_patterns:
            return 1.0  # first pattern is always novel
        
        # Compare to recent patterns (fast check)
        max_sim = 0.0
        for stored in self._recent_patterns:
            sim = self._cosine_similarity(compressed, stored)
            max_sim = max(max_sim, sim)
        
        # Also check stored episodes (sample for efficiency)
        n_check = min(100, len(self.episodes))
        if n_check > 0:
            indices = np.random.choice(len(self.episodes), size=n_check, replace=False)
            for idx in indices:
                sim = self._cosine_similarity(compressed, self.episodes[idx].pattern)
                max_sim = max(max_sim, sim)
        
        return 1.0 - max_sim
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
    
    @staticmethod
    def _hash_pattern(pattern: np.ndarray) -> str:
        """Generate a hash for fast pattern lookup."""
        return hashlib.md5(pattern.tobytes()).hexdigest()[:12]
    
    def _evict_least_salient(self):
        """Remove the least salient episode to make room."""
        if not self.episodes:
            return
        
        min_idx = min(range(len(self.episodes)), key=lambda i: self.episodes[i].salience)
        self.episodes.pop(min_idx)
    
    def get_stats(self) -> Dict:
        """Get hippocampal memory statistics."""
        if not self.episodes:
            return {"n_episodes": 0}
        
        saliences = [ep.salience for ep in self.episodes]
        replay_counts = [ep.replay_count for ep in self.episodes]
        
        return {
            "n_episodes": len(self.episodes),
            "mean_salience": float(np.mean(saliences)),
            "max_salience": float(np.max(saliences)),
            "total_replays": int(np.sum(replay_counts)),
            "capacity_used": len(self.episodes) / self.config.max_episodes,
        }
