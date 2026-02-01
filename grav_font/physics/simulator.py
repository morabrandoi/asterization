"""
2D Gravitational Particle Simulator.

Implements softened N-body gravitational physics with:
- Configurable gravitational constant
- Softening parameter to prevent singularities
- Velocity damping
- Optional velocity clamping
- Euler and RK4 integration methods
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..config import DEFAULT_CONFIG, SimulationConfig


class IntegrationMethod(Enum):
    """Available numerical integration methods."""

    EULER = "euler"
    RK4 = "rk4"


@dataclass
class Particle:
    """
    A particle in the gravitational simulation.

    Attributes:
        position: [x, y] position in canvas coordinates
        velocity: [vx, vy] velocity
        mass: Gravitational mass (affects attraction strength)
        is_pen: If True, this particle's trajectory is recorded for drawing
        is_fixed: If True, particle doesn't move (acts as fixed attractor)
    """

    position: np.ndarray
    velocity: np.ndarray
    mass: float = 10.0
    is_pen: bool = False
    is_fixed: bool = False

    def __post_init__(self):
        """Ensure position and velocity are numpy arrays."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)

    def copy(self) -> "Particle":
        """Create a deep copy of this particle."""
        return Particle(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass,
            is_pen=self.is_pen,
            is_fixed=self.is_fixed,
        )


@dataclass
class SimulationState:
    """Snapshot of the simulation state at a point in time."""

    time: float
    particles: list[Particle]
    pen_position: np.ndarray | None = None


class Simulator:
    """
    2D N-body gravitational simulator with softening and damping.

    Example usage:
        >>> pen = Particle([100, 100], [10, 0], mass=1.0, is_pen=True)
        >>> attractor = Particle([128, 128], [0, 0], mass=50.0, is_fixed=True)
        >>> sim = Simulator([pen, attractor])
        >>> trajectory = sim.run(1000)
    """

    def __init__(
        self,
        particles: list[Particle],
        config: SimulationConfig | None = None,
        integration_method: IntegrationMethod = IntegrationMethod.EULER,
    ):
        """
        Initialize the simulator.

        Args:
            particles: List of particles in the simulation
            config: Simulation configuration (uses defaults if None)
            integration_method: Numerical integration method to use
        """
        self.initial_particles = [p.copy() for p in particles]
        self.particles = [p.copy() for p in particles]
        self.config = config or DEFAULT_CONFIG
        self.integration_method = integration_method

        self.time = 0.0
        self.trajectory: list[np.ndarray] = []
        self.history: list[SimulationState] = []

        # Cache pen particle index for fast access
        self._pen_indices = [i for i, p in enumerate(self.particles) if p.is_pen]

        # Record initial pen position
        self._record_pen_position()

    def _record_pen_position(self) -> None:
        """Record the current pen particle position(s) to the trajectory."""
        for idx in self._pen_indices:
            self.trajectory.append(self.particles[idx].position.copy())

    def _record_state(self) -> None:
        """Record a full simulation state snapshot."""
        self.history.append(
            SimulationState(
                time=self.time,
                particles=[p.copy() for p in self.particles],
                pen_position=self.trajectory[-1].copy() if self.trajectory else None,
            )
        )

    def compute_gravitational_force(
        self,
        p1: Particle,
        p2: Particle,
    ) -> np.ndarray:
        """
        Compute softened gravitational force on p1 due to p2.

        Uses softened gravity: F = G * m1 * m2 * r / (|r|^2 + eps^2)^(3/2)

        Args:
            p1: Particle experiencing the force
            p2: Particle exerting the force

        Returns:
            Force vector [fx, fy] on p1
        """
        r = p2.position - p1.position
        r_squared = np.dot(r, r)

        # Softened distance to prevent singularity
        softened_dist = np.sqrt(r_squared + self.config.softening**2)

        # Gravitational force magnitude
        force_magnitude = self.config.G * p1.mass * p2.mass / (softened_dist**3)

        # Force vector (points from p1 toward p2)
        force = force_magnitude * r

        return force

    def compute_total_force(self, particle_idx: int) -> np.ndarray:
        """
        Compute the total force on a particle from all other particles.

        Args:
            particle_idx: Index of the particle to compute force for

        Returns:
            Total force vector [fx, fy]
        """
        particle = self.particles[particle_idx]
        total_force = np.zeros(2, dtype=np.float64)

        for i, other in enumerate(self.particles):
            if i == particle_idx:
                continue
            total_force += self.compute_gravitational_force(particle, other)

        return total_force

    def compute_acceleration(self, particle_idx: int) -> np.ndarray:
        """
        Compute the acceleration of a particle (F/m).

        Args:
            particle_idx: Index of the particle

        Returns:
            Acceleration vector [ax, ay]
        """
        force = self.compute_total_force(particle_idx)
        return force / self.particles[particle_idx].mass

    def _euler_step(self) -> None:
        """Perform one Euler integration step."""
        dt = self.config.dt

        # Compute all accelerations first (before updating positions)
        accelerations = [
            self.compute_acceleration(i) if not p.is_fixed else np.zeros(2)
            for i, p in enumerate(self.particles)
        ]

        # Update velocities and positions
        for i, particle in enumerate(self.particles):
            if particle.is_fixed:
                continue

            # Update velocity
            particle.velocity += accelerations[i] * dt

            # Apply damping
            particle.velocity *= self.config.damping

            # Clamp velocity if needed
            speed = np.linalg.norm(particle.velocity)
            if speed > self.config.max_velocity:
                particle.velocity = (particle.velocity / speed) * self.config.max_velocity

            # Update position
            particle.position += particle.velocity * dt

    def _rk4_step(self) -> None:
        """Perform one RK4 (Runge-Kutta 4th order) integration step."""
        dt = self.config.dt

        # Store original state
        original_positions = [p.position.copy() for p in self.particles]
        original_velocities = [p.velocity.copy() for p in self.particles]

        def get_derivatives():
            """Get velocity and acceleration for all particles."""
            derivs = []
            for i, p in enumerate(self.particles):
                if p.is_fixed:
                    derivs.append((np.zeros(2), np.zeros(2)))
                else:
                    derivs.append((p.velocity.copy(), self.compute_acceleration(i)))
            return derivs

        # k1: derivatives at current state
        k1 = get_derivatives()

        # k2: derivatives at midpoint using k1
        for i, p in enumerate(self.particles):
            if not p.is_fixed:
                p.position = original_positions[i] + 0.5 * dt * k1[i][0]
                p.velocity = original_velocities[i] + 0.5 * dt * k1[i][1]
        k2 = get_derivatives()

        # k3: derivatives at midpoint using k2
        for i, p in enumerate(self.particles):
            if not p.is_fixed:
                p.position = original_positions[i] + 0.5 * dt * k2[i][0]
                p.velocity = original_velocities[i] + 0.5 * dt * k2[i][1]
        k3 = get_derivatives()

        # k4: derivatives at endpoint using k3
        for i, p in enumerate(self.particles):
            if not p.is_fixed:
                p.position = original_positions[i] + dt * k3[i][0]
                p.velocity = original_velocities[i] + dt * k3[i][1]
        k4 = get_derivatives()

        # Final update using weighted average
        for i, p in enumerate(self.particles):
            if p.is_fixed:
                continue

            # RK4 weighted average
            p.position = original_positions[i] + (dt / 6.0) * (
                k1[i][0] + 2 * k2[i][0] + 2 * k3[i][0] + k4[i][0]
            )
            p.velocity = original_velocities[i] + (dt / 6.0) * (
                k1[i][1] + 2 * k2[i][1] + 2 * k3[i][1] + k4[i][1]
            )

            # Apply damping
            p.velocity *= self.config.damping

            # Clamp velocity
            speed = np.linalg.norm(p.velocity)
            if speed > self.config.max_velocity:
                p.velocity = (p.velocity / speed) * self.config.max_velocity

    def step(self) -> None:
        """
        Advance the simulation by one timestep.

        Updates particle positions and velocities, records pen trajectory.
        """
        if self.integration_method == IntegrationMethod.EULER:
            self._euler_step()
        elif self.integration_method == IntegrationMethod.RK4:
            self._rk4_step()

        self.time += self.config.dt
        self._record_pen_position()

    def run(
        self,
        n_steps: int | None = None,
        record_history: bool = False,
        history_interval: int = 10,
    ) -> np.ndarray:
        """
        Run the simulation for a specified number of steps.

        Args:
            n_steps: Number of timesteps to run (uses config default if None)
            record_history: If True, record full state snapshots
            history_interval: Record every N steps (if record_history=True)

        Returns:
            Pen trajectory as numpy array of shape (n_steps + 1, 2)
        """
        n_steps = n_steps or self.config.n_steps

        for i in range(n_steps):
            self.step()

            if record_history and (i % history_interval == 0):
                self._record_state()

        return self.get_trajectory()

    def get_trajectory(self) -> np.ndarray:
        """
        Get the pen particle trajectory.

        Returns:
            Numpy array of shape (n_points, 2) containing pen positions
        """
        return np.array(self.trajectory)

    def get_particle_positions(self) -> np.ndarray:
        """
        Get current positions of all particles.

        Returns:
            Numpy array of shape (n_particles, 2)
        """
        return np.array([p.position for p in self.particles])

    def reset(self) -> None:
        """Reset the simulation to initial conditions."""
        self.particles = [p.copy() for p in self.initial_particles]
        self.time = 0.0
        self.trajectory = []
        self.history = []
        self._record_pen_position()

    def set_initial_conditions(self, particles: list[Particle]) -> None:
        """
        Set new initial conditions and reset the simulation.

        Args:
            particles: New list of particles
        """
        self.initial_particles = [p.copy() for p in particles]
        self._pen_indices = [i for i, p in enumerate(particles) if p.is_pen]
        self.reset()


def create_orbital_system(
    center: tuple[float, float] = (128, 128),
    orbital_radius: float = 60,
    central_mass: float = 50.0,
    pen_mass: float = 1.0,
    pen_speed: float | None = None,
) -> list[Particle]:
    """
    Create a simple two-body orbital system.

    Sets up a pen particle orbiting a fixed central mass.

    Args:
        center: Position of the central mass
        orbital_radius: Distance of pen from center
        central_mass: Mass of the central attractor
        pen_mass: Mass of the pen particle
        pen_speed: Initial tangential speed (circular orbit if None)

    Returns:
        List of [pen, central_mass] particles
    """
    center = np.array(center, dtype=np.float64)

    # Pen starts to the right of center
    pen_pos = center + np.array([orbital_radius, 0])

    # Compute circular orbit velocity if not specified
    if pen_speed is None:
        # v = sqrt(G * M / r) for circular orbit
        pen_speed = np.sqrt(DEFAULT_CONFIG.G * central_mass / orbital_radius)

    # Tangential velocity (perpendicular to radius, counterclockwise)
    pen_vel = np.array([0, -pen_speed])

    pen = Particle(
        position=pen_pos,
        velocity=pen_vel,
        mass=pen_mass,
        is_pen=True,
    )

    central = Particle(
        position=center.copy(),
        velocity=np.zeros(2),
        mass=central_mass,
        is_fixed=True,
    )

    return [pen, central]


def create_binary_system(
    center: tuple[float, float] = (128, 128),
    separation: float = 80,
    mass1: float = 30.0,
    mass2: float = 30.0,
    pen_offset: tuple[float, float] = (0, 80),
    pen_mass: float = 1.0,
    pen_velocity: tuple[float, float] = (30, 0),
) -> list[Particle]:
    """
    Create a binary star system with a pen particle.

    Two fixed masses with a pen particle moving between/around them.

    Args:
        center: Center point between the two masses
        separation: Distance between the two masses
        mass1: Mass of first attractor
        mass2: Mass of second attractor
        pen_offset: Pen starting position offset from center
        pen_mass: Mass of pen particle
        pen_velocity: Initial pen velocity

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


def create_three_body_system(
    center: tuple[float, float] = (128, 128),
    radius: float = 60,
    masses: tuple[float, float, float] = (25.0, 25.0, 25.0),
    pen_offset: tuple[float, float] = (0, 0),
    pen_mass: float = 1.0,
    pen_velocity: tuple[float, float] = (20, 20),
) -> list[Particle]:
    """
    Create a three-body system with masses in a triangle and a pen particle.

    Args:
        center: Center of the triangle
        radius: Distance from center to each mass
        masses: Masses of the three attractors
        pen_offset: Pen starting position offset from center
        pen_mass: Mass of pen particle
        pen_velocity: Initial pen velocity

    Returns:
        List of [pen, mass1, mass2, mass3] particles
    """
    center = np.array(center, dtype=np.float64)

    # Place masses at vertices of equilateral triangle
    angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]

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
