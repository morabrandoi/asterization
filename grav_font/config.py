"""
Default configuration parameters for the gravitational font tracing system.
"""

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for the physics simulation."""

    # Canvas dimensions
    canvas_size: int = 256

    # Font rendering
    font_size: int = 200

    # Time integration
    dt: float = 0.01
    n_steps: int = 1000

    # Gravitational physics
    G: float = 100.0
    softening: float = 5.0
    damping: float = 0.999999999  # Effectively no damping (preserves orbital energy)
    max_velocity: float = 500.0

    # Pen rendering
    stroke_width: float = 2.0
    stroke_opacity: float = 1.0


@dataclass
class ParticleConfig:
    """Default particle parameters."""

    default_mass: float = 10.0
    pen_mass: float = 1.0


# Global default configuration instance
DEFAULT_CONFIG = SimulationConfig()
DEFAULT_PARTICLE_CONFIG = ParticleConfig()
