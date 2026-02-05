"""
environment.py — Virtual Environment (Sensory I/O Interface)
============================================================

Prioridad: 7

Input:  Simulated sensors / actuators
Output: Sensory input streams + motor feedback

The environment provides the CogMind instance with:

  1. SENSORY INPUT: Streams of structured input data
     - Visual: simple pixel grids or feature vectors
     - Auditory: frequency spectrums
     - Proprioceptive: body state (simulated)
     - Reward: scalar reinforcement signals

  2. MOTOR OUTPUT: Actions the system can take
     - The network's cortical output patterns are decoded into actions
     - Actions affect the environment state → new sensory input

  3. EMBODIMENT: Without a body, there's no grounding
     - Even a simple virtual body provides the feedback loops
       necessary for sensorimotor learning

The environment is the source of the experience that shapes the CogMind
instance. Without it, the SNN has the right architecture but nothing
to learn from. Your Cognitive Signature provides the structure;
the environment provides the content.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum


class SensoryModality(Enum):
    """Available sensory modalities."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    PROPRIOCEPTIVE = "proprioceptive"
    REWARD = "reward"
    TEXT = "text"           # tokenized text input
    CUSTOM = "custom"


@dataclass
class SensoryStream:
    """A stream of sensory input to the network."""
    modality: SensoryModality
    dimension: int              # size of input vector
    target_module: str          # which cortical module receives this
    encoding: str = "rate"      # "rate" (firing rate) or "temporal" (spike timing)
    gain: float = 1.0           # amplification factor
    noise_sigma: float = 0.05   # noise level


@dataclass 
class EnvironmentState:
    """Current state of the virtual environment."""
    timestep: int = 0
    time_ms: float = 0.0
    
    # Sensory inputs (modality → vector)
    sensory_inputs: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Reward signal
    reward: float = 0.0
    cumulative_reward: float = 0.0
    
    # Motor output (decoded from network activity)
    motor_output: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Environment-specific state
    custom_state: Dict = field(default_factory=dict)


class VirtualEnvironment:
    """
    Virtual environment for the CogMind instance.
    
    Provides sensory inputs and receives motor outputs. The environment
    is the source of experience that shapes the network through plasticity.
    
    Built-in environments:
      - "pattern_recognition": Present visual patterns, reward correct responses
      - "sequence_learning": Present temporal sequences to learn
      - "sensorimotor": Simple grid world with movement
      - "custom": User-defined environment with callbacks
    
    Usage:
        env = VirtualEnvironment(n_cortical=100_000, env_type="pattern_recognition")
        
        # Simulation loop:
        sensory = env.get_sensory_input(time_ms)
        engine.inject_sensory(sensory)
        
        # ... run engine step ...
        
        motor = engine.get_motor_output()
        reward = env.process_action(motor)
    """
    
    def __init__(
        self,
        n_cortical: int,
        env_type: str = "pattern_recognition",
        sensory_streams: Optional[List[SensoryStream]] = None,
        module_names: Optional[List[str]] = None,
        neuron_modules: Optional[np.ndarray] = None,
    ):
        self.n_cortical = n_cortical
        self.env_type = env_type
        self.module_names = module_names or []
        self.neuron_modules = neuron_modules
        self.state = EnvironmentState()
        
        # Default sensory streams
        if sensory_streams:
            self.streams = sensory_streams
        else:
            self.streams = self._default_streams(env_type)
        
        # Environment-specific initialization
        self._init_environment(env_type)
        
        # Motor output decoder
        self.n_motor = 10  # default motor dimensions
        
        print(f"[Environment] Initialized '{env_type}':")
        for s in self.streams:
            print(f"  {s.modality.value}: dim={s.dimension} → {s.target_module}")
    
    def _default_streams(self, env_type: str) -> List[SensoryStream]:
        """Create default sensory streams for built-in environments."""
        if env_type == "pattern_recognition":
            return [
                SensoryStream(SensoryModality.VISUAL, 784, "occipital_L",   # 28x28 grid
                             encoding="rate", gain=2.0),
                SensoryStream(SensoryModality.REWARD, 1, "frontal_L",
                             encoding="rate", gain=5.0),
            ]
        elif env_type == "sequence_learning":
            return [
                SensoryStream(SensoryModality.AUDITORY, 64, "temporal_L",
                             encoding="temporal", gain=1.5),
                SensoryStream(SensoryModality.REWARD, 1, "frontal_L",
                             encoding="rate", gain=5.0),
            ]
        elif env_type == "sensorimotor":
            return [
                SensoryStream(SensoryModality.VISUAL, 100, "occipital_L",   # 10x10 grid
                             encoding="rate", gain=2.0),
                SensoryStream(SensoryModality.PROPRIOCEPTIVE, 10, "parietal_L",
                             encoding="rate", gain=1.0),
                SensoryStream(SensoryModality.REWARD, 1, "frontal_L",
                             encoding="rate", gain=5.0),
            ]
        else:  # custom
            return [
                SensoryStream(SensoryModality.CUSTOM, 100, "frontal_L",
                             encoding="rate", gain=1.0),
            ]
    
    def _init_environment(self, env_type: str):
        """Initialize environment-specific state."""
        if env_type == "pattern_recognition":
            # Generate a set of patterns to recognize
            rng = np.random.default_rng(123)
            self.patterns = [
                rng.standard_normal(784).astype(np.float32) * 0.5
                for _ in range(10)  # 10 different patterns
            ]
            self.current_pattern_idx = 0
            self.presentation_time_ms = 500.0  # show each pattern for 500ms
            self.last_switch_time = 0.0
            
        elif env_type == "sequence_learning":
            # Generate sequences
            rng = np.random.default_rng(456)
            self.sequences = [
                rng.standard_normal((5, 64)).astype(np.float32) * 0.3
                for _ in range(5)  # 5 sequences of length 5
            ]
            self.current_seq_idx = 0
            self.current_step_in_seq = 0
            self.step_duration_ms = 200.0
            self.last_step_time = 0.0
            
        elif env_type == "sensorimotor":
            # Simple grid world
            self.grid_size = 10
            self.agent_pos = np.array([5, 5])
            self.goal_pos = np.array([8, 8])
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            self.grid[self.goal_pos[0], self.goal_pos[1]] = 1.0
    
    def get_sensory_input(self, time_ms: float) -> np.ndarray:
        """
        Generate sensory input vector for the current timestep.
        
        Returns an array of size n_cortical with sensory inputs mapped
        to the appropriate cortical modules.
        """
        self.state.time_ms = time_ms
        sensory = np.zeros(self.n_cortical, dtype=np.float32)
        
        for stream in self.streams:
            # Get raw input for this modality
            raw_input = self._get_raw_input(stream, time_ms)
            
            if raw_input is None:
                continue
            
            # Store in state
            self.state.sensory_inputs[stream.modality.value] = raw_input
            
            # Map to cortical neurons in target module
            sensory = self._map_to_cortex(sensory, raw_input, stream)
        
        return sensory
    
    def _get_raw_input(
        self, stream: SensoryStream, time_ms: float
    ) -> Optional[np.ndarray]:
        """Get raw sensory input for a modality at the current time."""
        
        if stream.modality == SensoryModality.REWARD:
            return np.array([self.state.reward], dtype=np.float32)
        
        if self.env_type == "pattern_recognition":
            if stream.modality == SensoryModality.VISUAL:
                # Switch patterns periodically
                if time_ms - self.last_switch_time >= self.presentation_time_ms:
                    self.current_pattern_idx = (self.current_pattern_idx + 1) % len(self.patterns)
                    self.last_switch_time = time_ms
                
                pattern = self.patterns[self.current_pattern_idx].copy()
                # Add noise
                pattern += np.random.normal(0, stream.noise_sigma, size=pattern.shape)
                return pattern
                
        elif self.env_type == "sequence_learning":
            if stream.modality == SensoryModality.AUDITORY:
                if time_ms - self.last_step_time >= self.step_duration_ms:
                    self.current_step_in_seq += 1
                    if self.current_step_in_seq >= len(self.sequences[self.current_seq_idx]):
                        self.current_step_in_seq = 0
                        self.current_seq_idx = (self.current_seq_idx + 1) % len(self.sequences)
                    self.last_step_time = time_ms
                
                return self.sequences[self.current_seq_idx][self.current_step_in_seq].copy()
                
        elif self.env_type == "sensorimotor":
            if stream.modality == SensoryModality.VISUAL:
                return self.grid.flatten()
            elif stream.modality == SensoryModality.PROPRIOCEPTIVE:
                # Encode agent position and velocity
                prop = np.zeros(10, dtype=np.float32)
                prop[0] = self.agent_pos[0] / self.grid_size
                prop[1] = self.agent_pos[1] / self.grid_size
                # Distance to goal
                dist = np.linalg.norm(self.agent_pos - self.goal_pos)
                prop[2] = dist / (self.grid_size * np.sqrt(2))
                return prop
        
        return np.zeros(stream.dimension, dtype=np.float32)
    
    def _map_to_cortex(
        self,
        sensory: np.ndarray,
        raw_input: np.ndarray,
        stream: SensoryStream,
    ) -> np.ndarray:
        """Map raw sensory input to cortical neuron positions."""
        if self.neuron_modules is None or not self.module_names:
            # No module mapping: distribute evenly across first N neurons
            n = min(len(raw_input), self.n_cortical)
            sensory[:n] += raw_input[:n] * stream.gain
            return sensory
        
        # Find neurons in target module
        target_idx = None
        for i, name in enumerate(self.module_names):
            if stream.target_module in name:
                target_idx = i
                break
        
        if target_idx is None:
            return sensory
        
        target_neurons = np.where(self.neuron_modules == target_idx)[0]
        
        if len(target_neurons) == 0:
            return sensory
        
        # Map input dimensions to target neurons
        # If more neurons than input dims: tile the input
        # If fewer neurons than input dims: subsample
        n_target = len(target_neurons)
        n_input = len(raw_input)
        
        if n_target >= n_input:
            # Tile input across neurons
            mapped = np.tile(raw_input, (n_target // n_input + 1))[:n_target]
        else:
            # Subsample input
            indices = np.linspace(0, n_input - 1, n_target, dtype=int)
            mapped = raw_input[indices]
        
        sensory[target_neurons] += mapped * stream.gain
        
        return sensory
    
    def process_action(self, motor_output: np.ndarray) -> float:
        """
        Process the network's motor output and compute reward.
        
        Args:
            motor_output: Decoded motor output from cortical activity
            
        Returns:
            reward: Scalar reward signal
        """
        self.state.motor_output = motor_output
        reward = 0.0
        
        if self.env_type == "pattern_recognition":
            # Decode: which pattern does the network think it's seeing?
            if len(motor_output) >= len(self.patterns):
                predicted = np.argmax(motor_output[:len(self.patterns)])
                if predicted == self.current_pattern_idx:
                    reward = 1.0
                else:
                    reward = -0.1
                    
        elif self.env_type == "sensorimotor":
            # Decode: movement direction from motor output
            if len(motor_output) >= 4:
                action = np.argmax(motor_output[:4])  # 0=up, 1=down, 2=left, 3=right
                
                moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
                move = np.array(moves[action])
                
                new_pos = self.agent_pos + move
                new_pos = np.clip(new_pos, 0, self.grid_size - 1)
                self.agent_pos = new_pos
                
                # Reward: closer to goal = positive
                dist = np.linalg.norm(self.agent_pos - self.goal_pos)
                if dist < 1.0:
                    reward = 10.0  # reached goal!
                    # Reset goal
                    rng = np.random.default_rng()
                    self.goal_pos = rng.integers(0, self.grid_size, size=2)
                    self.grid[:] = 0
                    self.grid[self.goal_pos[0], self.goal_pos[1]] = 1.0
                else:
                    reward = -0.01  # small cost for each step
        
        self.state.reward = reward
        self.state.cumulative_reward += reward
        
        return reward
    
    def get_novelty_signal(self) -> float:
        """
        Compute novelty of current sensory input.
        Returns 0-1 score (1 = completely new).
        """
        if self.env_type == "pattern_recognition":
            # Novelty when pattern switches
            if self.state.time_ms - self.last_switch_time < 10.0:
                return 0.8  # new pattern = novel
            return 0.1
        return 0.0
    
    def reset(self):
        """Reset environment to initial state."""
        self.state = EnvironmentState()
        self._init_environment(self.env_type)
        print(f"[Environment] Reset to '{self.env_type}'")
