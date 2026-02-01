"""
Visualization module for gravitational font tracing simulations.

Provides static and animated visualizations using matplotlib to
display particle trajectories, glyph targets, and simulation state.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..glyphs.target import GlyphData
from ..physics.simulator import Particle, Simulator
from ..render.pen_render import render_trajectory


def visualize_static(
    glyph: GlyphData | None = None,
    trajectory: np.ndarray | None = None,
    particles: list[Particle] | None = None,
    title: str = "Gravitational Font Tracing",
    figsize: tuple[float, float] = (10, 10),
    show_glyph: bool = True,
    show_trajectory: bool = True,
    show_particles: bool = True,
    trajectory_color: str = "cyan",
    trajectory_alpha: float = 0.8,
    trajectory_linewidth: float = 1.0,
    glyph_cmap: str = "gray",
    glyph_alpha: float = 0.5,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a static visualization of the simulation results.

    Args:
        glyph: Optional GlyphData with target glyph image
        trajectory: Optional trajectory array of shape (n_points, 2)
        particles: Optional list of particles to show current positions
        title: Plot title
        figsize: Figure size in inches
        show_glyph: Whether to display the glyph background
        show_trajectory: Whether to display the pen trajectory
        show_particles: Whether to display particle positions
        trajectory_color: Color for the trajectory line
        trajectory_alpha: Alpha for trajectory
        trajectory_linewidth: Line width for trajectory
        glyph_cmap: Colormap for glyph image
        glyph_alpha: Alpha for glyph background
        save_path: Optional path to save the figure
        dpi: DPI for saved figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    canvas_size = 256
    if glyph is not None:
        canvas_size = glyph.image.shape[0]

    # Show glyph as background
    if show_glyph and glyph is not None:
        ax.imshow(
            glyph.image,
            cmap=glyph_cmap,
            alpha=glyph_alpha,
            extent=[0, canvas_size, canvas_size, 0],
        )

    # Show trajectory
    if show_trajectory and trajectory is not None and len(trajectory) > 1:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=trajectory_color,
            alpha=trajectory_alpha,
            linewidth=trajectory_linewidth,
        )

    # Show particles
    if show_particles and particles is not None:
        for p in particles:
            if p.is_pen:
                color = "red"
                size = 50
                marker = "o"
            else:
                color = "blue"
                size = 100 * (p.mass / 10.0) ** 0.5
                marker = "o"

            ax.scatter(
                p.position[0],
                p.position[1],
                c=color,
                s=size,
                marker=marker,
                edgecolors="white",
                linewidths=1,
                zorder=10,
            )

    ax.set_xlim(0, canvas_size)
    ax.set_ylim(canvas_size, 0)  # Flip y-axis to match image coordinates
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def visualize_comparison(
    glyph: GlyphData,
    trajectory: np.ndarray,
    stroke_width: float = 2.0,
    blur: float = 0.0,
    figsize: tuple[float, float] = (15, 5),
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a side-by-side comparison of target glyph and rendered trajectory.

    Args:
        glyph: GlyphData with target glyph
        trajectory: Trajectory array
        stroke_width: Width of pen stroke for rendering
        blur: Gaussian blur sigma for pen trail
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure

    Returns:
        matplotlib Figure object
    """
    canvas_size = glyph.image.shape[0]

    # Render trajectory to image
    rendered = render_trajectory(
        trajectory,
        canvas_size=canvas_size,
        stroke_width=stroke_width,
        blur=blur,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Target glyph
    axes[0].imshow(glyph.image, cmap="gray")
    axes[0].set_title(f"Target: '{glyph.character}'")
    axes[0].axis("off")

    # Rendered trajectory
    axes[1].imshow(rendered, cmap="gray")
    axes[1].set_title("Rendered Trajectory")
    axes[1].axis("off")

    # Overlay comparison
    overlay = np.zeros((*glyph.image.shape, 3))
    overlay[:, :, 0] = glyph.image  # Red channel = target
    overlay[:, :, 1] = rendered  # Green channel = rendered
    overlay[:, :, 2] = 0  # Blue channel = 0

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Red=Target, Green=Trace)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def animate_simulation(
    simulator: Simulator,
    glyph: GlyphData | None = None,
    n_steps: int = 500,
    interval: int = 20,
    figsize: tuple[float, float] = (8, 8),
    trail_length: int | None = None,
    stroke_width: float = 1.0,
    show_velocity: bool = False,
    velocity_scale: float = 0.1,
    save_path: str | Path | None = None,
    fps: int = 30,
) -> FuncAnimation:
    """
    Create an animated visualization of the simulation.

    Args:
        simulator: Simulator instance (will be reset and run)
        glyph: Optional GlyphData for background
        n_steps: Number of simulation steps
        interval: Milliseconds between frames
        figsize: Figure size
        trail_length: Number of trajectory points to show (None = all)
        stroke_width: Width of the pen trail line
        show_velocity: Whether to show velocity vectors
        velocity_scale: Scale factor for velocity vectors
        save_path: Optional path to save animation (gif or mp4)
        fps: Frames per second for saved animation

    Returns:
        FuncAnimation object
    """
    simulator.reset()

    canvas_size = 256
    if glyph is not None:
        canvas_size = glyph.image.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Show glyph background
    if glyph is not None:
        ax.imshow(
            glyph.image,
            cmap="gray",
            alpha=0.4,
            extent=[0, canvas_size, canvas_size, 0],
        )

    # Initialize plot elements
    (trajectory_line,) = ax.plot([], [], "c-", alpha=0.7, linewidth=stroke_width)
    pen_scatter = ax.scatter([], [], c="red", s=30, zorder=10)
    mass_scatter = ax.scatter([], [], c="blue", s=100, zorder=9)

    velocity_arrows = []
    if show_velocity:
        for _ in simulator.particles:
            arrow = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "->", "color": "yellow", "lw": 1},
            )
            velocity_arrows.append(arrow)

    ax.set_xlim(0, canvas_size)
    ax.set_ylim(canvas_size, 0)
    ax.set_aspect("equal")
    ax.set_title("Gravitational Font Tracing")

    # Text for step counter
    step_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        color="white",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.5},
    )

    def init():
        trajectory_line.set_data([], [])
        pen_scatter.set_offsets(np.empty((0, 2)))
        mass_scatter.set_offsets(np.empty((0, 2)))
        step_text.set_text("")
        return [trajectory_line, pen_scatter, mass_scatter, step_text]

    def update(frame):
        # Run one simulation step
        simulator.step()

        # Get trajectory
        traj = simulator.get_trajectory()
        if trail_length is not None and len(traj) > trail_length:
            traj_display = traj[-trail_length:]
        else:
            traj_display = traj

        trajectory_line.set_data(traj_display[:, 0], traj_display[:, 1])

        # Update particle positions
        pen_positions = []
        mass_positions = []
        mass_sizes = []

        for p in simulator.particles:
            if p.is_pen:
                pen_positions.append(p.position)
            else:
                mass_positions.append(p.position)
                mass_sizes.append(50 * (p.mass / 10.0) ** 0.5)

        if pen_positions:
            pen_scatter.set_offsets(np.array(pen_positions))
        if mass_positions:
            mass_scatter.set_offsets(np.array(mass_positions))
            mass_scatter.set_sizes(mass_sizes)

        # Update velocity arrows
        if show_velocity:
            for _i, (p, arrow) in enumerate(
                zip(simulator.particles, velocity_arrows, strict=False)
            ):
                arrow.xy = p.position + p.velocity * velocity_scale
                arrow.xyann = p.position

        step_text.set_text(f"Step: {frame + 1}/{n_steps}")

        return [trajectory_line, pen_scatter, mass_scatter, step_text]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        init_func=init,
        blit=True,
        interval=interval,
    )

    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == ".gif":
            anim.save(save_path, writer="pillow", fps=fps)
        elif save_path.suffix == ".mp4":
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            anim.save(save_path, fps=fps)

    return anim


def visualize_distance_field(
    glyph: GlyphData,
    figsize: tuple[float, float] = (12, 4),
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Visualize the glyph's distance field and gradient.

    Args:
        glyph: GlyphData with distance_field and gradient_field computed
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original glyph
    axes[0].imshow(glyph.image, cmap="gray")
    axes[0].set_title(f"Glyph: '{glyph.character}'")
    axes[0].axis("off")

    # Distance field
    if glyph.distance_field is not None:
        im = axes[1].imshow(glyph.distance_field, cmap="RdBu")
        axes[1].set_title("Signed Distance Field")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Distance field\nnot computed",
            ha="center",
            va="center",
        )
        axes[1].axis("off")

    # Gradient field (as quiver plot)
    if glyph.gradient_field is not None:
        grad_y, grad_x = glyph.gradient_field
        h, w = grad_x.shape

        # Subsample for cleaner visualization
        step = max(1, h // 20)
        y_coords, x_coords = np.mgrid[0:h:step, 0:w:step]

        axes[2].imshow(glyph.image, cmap="gray", alpha=0.5)
        axes[2].quiver(
            x_coords,
            y_coords,
            grad_x[::step, ::step],
            grad_y[::step, ::step],
            color="red",
            alpha=0.7,
            scale=30,
        )
        axes[2].set_title("Gradient Field")
        axes[2].axis("off")
    else:
        axes[2].text(
            0.5,
            0.5,
            "Gradient field\nnot computed",
            ha="center",
            va="center",
        )
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def visualize_trajectory_evolution(
    trajectory: np.ndarray,
    n_frames: int = 6,
    glyph: GlyphData | None = None,
    figsize: tuple[float, float] = (15, 3),
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Show trajectory evolution at different time points.

    Args:
        trajectory: Full trajectory array
        n_frames: Number of time snapshots to show
        glyph: Optional GlyphData for background
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, n_frames, figsize=figsize)

    n_points = len(trajectory)
    indices = np.linspace(0, n_points - 1, n_frames, dtype=int)

    canvas_size = 256
    if glyph is not None:
        canvas_size = glyph.image.shape[0]

    for ax, idx in zip(axes, indices, strict=False):
        if glyph is not None:
            ax.imshow(
                glyph.image,
                cmap="gray",
                alpha=0.4,
                extent=[0, canvas_size, canvas_size, 0],
            )

        traj_slice = trajectory[: idx + 1]
        if len(traj_slice) > 1:
            ax.plot(traj_slice[:, 0], traj_slice[:, 1], "c-", linewidth=1)

        # Mark current position
        ax.scatter(
            trajectory[idx, 0],
            trajectory[idx, 1],
            c="red",
            s=30,
            zorder=10,
        )

        ax.set_xlim(0, canvas_size)
        ax.set_ylim(canvas_size, 0)
        ax.set_aspect("equal")
        ax.set_title(f"Step {idx}")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def show_plot():
    """Display all open matplotlib figures."""
    plt.show()
