"""
Preset particle configurations for gravitational font tracing.

Provides ready-to-use particle setups for various orbital patterns:
- Simple circular orbits
- Binary systems (figure-8 patterns)
- Three-body chaotic systems
- Glyph-centered configurations
"""

import numpy as np

from .config import DEFAULT_CONFIG
from .physics.simulator import Particle


def single_orbit(
    center: tuple[float, float] = (128, 128),
    orbital_radius: float = 60,
    central_mass: float = 50.0,
    pen_mass: float = 1.0,
    clockwise: bool = False,
) -> list[Particle]:
    """
    Create a simple two-body system with pen orbiting a central mass.

    Produces clean circular/elliptical orbits.

    Args:
        center: Position of the central mass
        orbital_radius: Starting distance of pen from center
        central_mass: Mass of the fixed central body
        pen_mass: Mass of the pen particle
        clockwise: If True, orbit clockwise; else counterclockwise

    Returns:
        List of [pen, central_mass] particles
    """
    center = np.array(center, dtype=np.float64)

    # Pen starts to the right of center
    pen_pos = center + np.array([orbital_radius, 0])

    # Circular orbit velocity: v = sqrt(G * M / r)
    speed = np.sqrt(DEFAULT_CONFIG.G * central_mass / orbital_radius)

    # Tangential velocity (perpendicular to radius)
    direction = 1 if clockwise else -1
    pen_vel = np.array([0, direction * speed])

    pen = Particle(
        position=pen_pos,
        velocity=pen_vel,
        mass=pen_mass,
        is_pen=True,
    )

    central = Particle(
        position=np.array(center),
        velocity=np.zeros(2),
        mass=central_mass,
        is_fixed=True,
    )

    return [pen, central]


def binary_system(
    center: tuple[float, float] = (128, 128),
    separation: float = 80,
    mass1: float = 30.0,
    mass2: float = 30.0,
    pen_offset: tuple[float, float] = (0, 60),
    pen_velocity: tuple[float, float] = (35, 0),
    pen_mass: float = 1.0,
) -> list[Particle]:
    """
    Create a binary star system with two fixed masses and a pen particle.

    Can produce figure-8 patterns or chaotic orbits depending on
    initial conditions.

    Args:
        center: Center point between the two masses
        separation: Distance between the two masses
        mass1: Mass of first attractor (left)
        mass2: Mass of second attractor (right)
        pen_offset: Pen starting position offset from center
        pen_velocity: Initial pen velocity
        pen_mass: Mass of pen particle

    Returns:
        List of [pen, mass1, mass2] particles
    """
    center = np.array(center, dtype=np.float64)

    pos1 = center + np.array([-separation / 2, 0])
    pos2 = center + np.array([separation / 2, 0])

    pen = Particle(
        position=center + np.array(pen_offset),
        velocity=np.array(pen_velocity),
        mass=pen_mass,
        is_pen=True,
    )

    body1 = Particle(
        position=pos1,
        velocity=np.zeros(2),
        mass=mass1,
        is_fixed=True,
    )

    body2 = Particle(
        position=pos2,
        velocity=np.zeros(2),
        mass=mass2,
        is_fixed=True,
    )

    return [pen, body1, body2]


def three_body_triangle(
    center: tuple[float, float] = (128, 128),
    radius: float = 70,
    masses: tuple[float, float, float] = (25.0, 25.0, 25.0),
    pen_offset: tuple[float, float] = (0, 0),
    pen_velocity: tuple[float, float] = (25, 15),
    pen_mass: float = 1.0,
) -> list[Particle]:
    """
    Create a three-body system with masses at triangle vertices.

    Produces chaotic, complex orbital patterns.

    Args:
        center: Center of the triangle
        radius: Distance from center to each mass
        masses: Masses of the three attractors
        pen_offset: Pen starting position offset from center
        pen_velocity: Initial pen velocity
        pen_mass: Mass of pen particle

    Returns:
        List of [pen, mass1, mass2, mass3] particles
    """
    center = np.array(center, dtype=np.float64)

    # Equilateral triangle vertices
    angles = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3]

    pen = Particle(
        position=center + np.array(pen_offset),
        velocity=np.array(pen_velocity),
        mass=pen_mass,
        is_pen=True,
    )

    particles = [pen]

    for i, angle in enumerate(angles):
        pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
        particles.append(
            Particle(
                position=pos,
                velocity=np.zeros(2),
                mass=masses[i],
                is_fixed=True,
            )
        )

    return particles


def four_corners(
    center: tuple[float, float] = (128, 128),
    half_size: float = 60,
    corner_mass: float = 20.0,
    pen_offset: tuple[float, float] = (30, 30),
    pen_velocity: tuple[float, float] = (20, -20),
    pen_mass: float = 1.0,
) -> list[Particle]:
    """
    Create a system with masses at four corners of a square.

    Good for creating bounded, structured patterns.

    Args:
        center: Center of the square
        half_size: Half the side length of the square
        corner_mass: Mass of each corner attractor
        pen_offset: Pen starting position offset from center
        pen_velocity: Initial pen velocity
        pen_mass: Mass of pen particle

    Returns:
        List of [pen, corner1, corner2, corner3, corner4] particles
    """
    center = np.array(center, dtype=np.float64)

    corners = [
        center + np.array([-half_size, -half_size]),
        center + np.array([half_size, -half_size]),
        center + np.array([half_size, half_size]),
        center + np.array([-half_size, half_size]),
    ]

    pen = Particle(
        position=center + np.array(pen_offset),
        velocity=np.array(pen_velocity),
        mass=pen_mass,
        is_pen=True,
    )

    particles = [pen]

    for corner in corners:
        particles.append(
            Particle(
                position=corner,
                velocity=np.zeros(2),
                mass=corner_mass,
                is_fixed=True,
            )
        )

    return particles


def spiral_inward(
    center: tuple[float, float] = (128, 128),
    start_radius: float = 100,
    central_mass: float = 80.0,
    pen_mass: float = 1.0,
    spiral_factor: float = 0.7,
) -> list[Particle]:
    """
    Create a system that produces inward spiraling motion.

    The pen starts with less than orbital velocity, causing it to spiral in.

    Args:
        center: Position of the central mass
        start_radius: Starting distance of pen from center
        central_mass: Mass of the fixed central body
        pen_mass: Mass of the pen particle
        spiral_factor: Fraction of circular orbit velocity (< 1 spirals in)

    Returns:
        List of [pen, central_mass] particles
    """
    center = np.array(center, dtype=np.float64)

    pen_pos = center + np.array([start_radius, 0])

    # Sub-orbital velocity causes spiral
    circular_speed = np.sqrt(DEFAULT_CONFIG.G * central_mass / start_radius)
    pen_vel = np.array([0, -circular_speed * spiral_factor])

    pen = Particle(
        position=pen_pos,
        velocity=pen_vel,
        mass=pen_mass,
        is_pen=True,
    )

    central = Particle(
        position=np.array(center),
        velocity=np.zeros(2),
        mass=central_mass,
        is_fixed=True,
    )

    return [pen, central]


def glyph_centered(
    glyph_center: tuple[float, float],
    canvas_size: int = 256,
    n_attractors: int = 3,
    attractor_mass: float = 25.0,
    pen_mass: float = 1.0,
    pen_speed: float = 30.0,
) -> list[Particle]:
    """
    Create a system with attractors distributed around a glyph center.

    Useful for creating patterns that roughly follow glyph shapes.

    Args:
        glyph_center: Center of the glyph (from GlyphData.center)
        canvas_size: Size of the canvas
        n_attractors: Number of attractors to place
        attractor_mass: Mass of each attractor
        pen_mass: Mass of pen particle
        pen_speed: Initial speed of pen

    Returns:
        List of particles with pen and attractors
    """
    center = np.array(glyph_center, dtype=np.float64)
    radius = canvas_size * 0.3

    # Start pen offset from center
    pen_pos = center + np.array([radius * 0.5, 0])
    pen_vel = np.array([0, -pen_speed])

    pen = Particle(
        position=pen_pos,
        velocity=pen_vel,
        mass=pen_mass,
        is_pen=True,
    )

    particles = [pen]

    # Distribute attractors evenly around center
    for i in range(n_attractors):
        angle = 2 * np.pi * i / n_attractors
        pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
        particles.append(
            Particle(
                position=pos,
                velocity=np.zeros(2),
                mass=attractor_mass,
                is_fixed=True,
            )
        )

    return particles


def random_system(
    canvas_size: int = 256,
    n_attractors: int = 3,
    mass_range: tuple[float, float] = (15.0, 40.0),
    pen_speed_range: tuple[float, float] = (20.0, 50.0),
    margin: float = 40.0,
    seed: int | None = None,
) -> list[Particle]:
    """
    Create a random particle configuration.

    Useful for exploring diverse orbital patterns.

    Args:
        canvas_size: Size of the canvas
        n_attractors: Number of random attractors
        mass_range: (min, max) mass for attractors
        pen_speed_range: (min, max) initial speed for pen
        margin: Minimum distance from canvas edges
        seed: Random seed for reproducibility

    Returns:
        List of particles with pen and random attractors
    """
    if seed is not None:
        np.random.seed(seed)

    # Random pen position and velocity
    pen_pos = np.random.uniform(margin, canvas_size - margin, 2)
    speed = np.random.uniform(*pen_speed_range)
    angle = np.random.uniform(0, 2 * np.pi)
    pen_vel = speed * np.array([np.cos(angle), np.sin(angle)])

    pen = Particle(
        position=pen_pos,
        velocity=pen_vel,
        mass=1.0,
        is_pen=True,
    )

    particles = [pen]

    # Random attractors
    for _ in range(n_attractors):
        pos = np.random.uniform(margin, canvas_size - margin, 2)
        mass = np.random.uniform(*mass_range)
        particles.append(
            Particle(
                position=pos,
                velocity=np.zeros(2),
                mass=mass,
                is_fixed=True,
            )
        )

    return particles


# Dictionary of all presets for easy access
PRESETS = {
    "single_orbit": single_orbit,
    "binary_system": binary_system,
    "three_body_triangle": three_body_triangle,
    "four_corners": four_corners,
    "spiral_inward": spiral_inward,
    "glyph_centered": glyph_centered,
    "random_system": random_system,
}


def list_presets() -> list[str]:
    """Return list of available preset names."""
    return list(PRESETS.keys())


def get_preset(name: str, **kwargs) -> list[Particle]:
    """
    Get particles from a named preset.

    Args:
        name: Name of the preset
        **kwargs: Arguments to pass to the preset function

    Returns:
        List of particles

    Raises:
        ValueError: If preset name is not found
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return PRESETS[name](**kwargs)
