#!/usr/bin/env python3
"""
Gravitational Font Tracing Demo

This script demonstrates the gravitational font tracing system by:
1. Loading a font and rasterizing a glyph
2. Setting up a gravitational particle system
3. Running the simulation
4. Visualizing the results

Usage:
    python run_demo.py                          # Run with defaults
    python run_demo.py --font path/to/font.ttf  # Use custom font
    python run_demo.py --char B                 # Trace different character
    python run_demo.py --preset binary_system   # Use different preset
    python run_demo.py --animate                # Show animation
    python run_demo.py --save output.png        # Save result

Requirements:
    conda activate grav-font
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Gravitational Font Tracing Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--font",
        "-f",
        type=str,
        default=None,
        help="Path to TTF/OTF font file (uses system font if not provided)",
    )
    parser.add_argument(
        "--char",
        "-c",
        type=str,
        default="A",
        help="Character to trace (default: A)",
    )
    parser.add_argument(
        "--preset",
        "-p",
        type=str,
        default="three_body_triangle",
        choices=[
            "single_orbit",
            "binary_system",
            "three_body_triangle",
            "four_corners",
            "spiral_inward",
            "random_system",
        ],
        help="Particle configuration preset (default: three_body_triangle)",
    )
    parser.add_argument(
        "--steps",
        "-n",
        type=int,
        default=2000,
        help="Number of simulation steps (default: 2000)",
    )
    parser.add_argument(
        "--animate",
        "-a",
        action="store_true",
        help="Show animated simulation instead of static result",
    )
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        default=None,
        help="Save output to file (png, gif, or mp4)",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=256,
        help="Canvas size in pixels (default: 256)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=200,
        help="Font size for glyph rendering (default: 200)",
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=1.5,
        help="Pen stroke width (default: 1.5)",
    )
    parser.add_argument(
        "--no-glyph",
        action="store_true",
        help="Don't show glyph background (pure orbital art)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for random_system preset",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit",
    )
    parser.add_argument(
        "--pen-angle",
        type=float,
        default=None,
        help="Initial pen velocity angle in degrees (0=right, 90=down, 180=left, 270=up)",
    )
    parser.add_argument(
        "--pen-speed",
        type=float,
        default=None,
        help="Initial pen speed (overrides preset default)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Physics timestep - larger = faster but less accurate (default: 0.01)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=16,
        help="Milliseconds between animation frames (default: 16 = ~60fps)",
    )
    parser.add_argument(
        "--blur",
        type=float,
        default=0.0,
        help="Gaussian blur sigma applied to pen trail (default: 0 = no blur)",
    )

    args = parser.parse_args()

    # Import after argparse to show help faster
    from grav_font.config import SimulationConfig
    from grav_font.physics.simulator import Simulator
    from grav_font.presets import get_preset, glyph_centered, list_presets
    from grav_font.visualize.viewer import (
        animate_simulation,
        show_plot,
        visualize_comparison,
        visualize_static,
    )

    if args.list_presets:
        print("Available presets:")
        for name in list_presets():
            print(f"  - {name}")
        return 0

    # Load glyph if font is provided
    glyph = None
    if args.font:
        font_path = Path(args.font)
        if not font_path.exists():
            print(f"Error: Font file not found: {font_path}")
            return 1

        from grav_font.glyphs.target import load_glyph

        print(f"Loading glyph '{args.char}' from {font_path}...")
        glyph = load_glyph(
            font_path,
            args.char,
            canvas_size=args.canvas_size,
            font_size=args.font_size,
        )
        print(f"  Glyph center: {glyph.center}")
        print(f"  Bounding box: {glyph.bbox}")

    # Create particle configuration
    print(f"\nUsing preset: {args.preset}")

    preset_kwargs = {}
    center = (args.canvas_size / 2, args.canvas_size / 2)

    if args.preset == "random_system":
        preset_kwargs["canvas_size"] = args.canvas_size
        if args.seed is not None:
            preset_kwargs["seed"] = args.seed
        particles = get_preset(args.preset, **preset_kwargs)
    elif glyph is not None and args.preset == "glyph_centered":
        particles = glyph_centered(
            glyph_center=glyph.center,
            canvas_size=args.canvas_size,
        )
    else:
        preset_kwargs["center"] = center
        particles = get_preset(args.preset, **preset_kwargs)

    print(f"  Particles: {len(particles)}")
    for i, p in enumerate(particles):
        ptype = "pen" if p.is_pen else "mass"
        print(f"    [{i}] {ptype}: pos={p.position}, mass={p.mass}")

    # Override pen velocity if --pen-angle or --pen-speed specified
    if args.pen_angle is not None or args.pen_speed is not None:
        import math

        for p in particles:
            if p.is_pen:
                # Get current speed or use override
                current_speed = np.linalg.norm(p.velocity)
                speed = args.pen_speed if args.pen_speed is not None else current_speed

                # Get angle (default to current direction or 0)
                if args.pen_angle is not None:
                    angle_rad = math.radians(args.pen_angle)
                else:
                    angle_rad = math.atan2(p.velocity[1], p.velocity[0])

                # Set new velocity
                p.velocity = np.array(
                    [
                        speed * math.cos(angle_rad),
                        speed * math.sin(angle_rad),
                    ]
                )
                print(
                    f"  Pen velocity override: angle={math.degrees(angle_rad):.1f}°, "
                    f"speed={speed:.1f}, vel={p.velocity}"
                )

    # Create simulation config
    config = SimulationConfig(
        canvas_size=args.canvas_size,
        n_steps=args.steps,
        stroke_width=args.stroke_width,
        dt=args.dt if args.dt is not None else 0.01,
    )

    # Create simulator
    simulator = Simulator(particles, config=config)

    if args.animate:
        # Animated visualization
        print(f"\nRunning animated simulation ({args.steps} steps)...")
        print("Close the window to exit.")

        # IMPORTANT: Store the animation object to prevent garbage collection
        # Without this reference, Python's GC may delete the animation before plt.show()
        _anim = animate_simulation(  # noqa: F841
            simulator,
            glyph=glyph if not args.no_glyph else None,
            n_steps=args.steps,
            interval=args.interval,
            save_path=args.save,
        )

        show_plot()
        del _anim  # Explicit cleanup after show
    else:
        # Static visualization
        print(f"\nRunning simulation ({args.steps} steps)...")
        trajectory = simulator.run(args.steps)
        print(f"  Trajectory points: {len(trajectory)}")

        # Create visualization
        if glyph is not None and not args.no_glyph:
            print("\nCreating comparison visualization...")
            visualize_comparison(
                glyph,
                trajectory,
                stroke_width=args.stroke_width,
                blur=args.blur,
                save_path=args.save if args.save else None,
            )
        else:
            print("\nCreating static visualization...")
            visualize_static(
                glyph=glyph if not args.no_glyph else None,
                trajectory=trajectory,
                particles=simulator.particles,
                title=(f"Gravitational Trace: '{args.char}'" if glyph else "Orbital Art"),
                save_path=args.save if args.save else None,
            )

        if args.save:
            print(f"Saved to: {args.save}")
        else:
            print("Displaying result (close window to exit)...")
            show_plot()

    return 0


def demo_without_font():
    """
    Run a simple demo without requiring a font file.

        Useful for testing the physics and visualization system.
    """
    from grav_font.config import SimulationConfig
    from grav_font.physics.simulator import Simulator
    from grav_font.presets import three_body_triangle
    from grav_font.visualize.viewer import show_plot, visualize_static

    print("=" * 50)
    print("Gravitational Font Tracing - Demo (No Font)")
    print("=" * 50)

    # Create three-body system
    particles = three_body_triangle()

    print("\nParticle configuration:")
    for i, p in enumerate(particles):
        ptype = "PEN" if p.is_pen else "MASS"
        print(
            f"  [{i}] {ptype}: pos=({p.position[0]:.1f}, {p.position[1]:.1f}), "
            f"vel=({p.velocity[0]:.1f}, {p.velocity[1]:.1f}), mass={p.mass}"
        )

    # Run simulation
    config = SimulationConfig(n_steps=3000, damping=0.999)
    simulator = Simulator(particles, config=config)

    print(f"\nRunning simulation ({config.n_steps} steps)...")
    trajectory = simulator.run()
    print(f"  Generated {len(trajectory)} trajectory points")

    # Visualize
    print("\nDisplaying result...")
    visualize_static(
        trajectory=trajectory,
        particles=simulator.particles,
        title="Three-Body Gravitational System",
        trajectory_color="lime",
    )
    show_plot()


def demo_all_presets():
    """
    Run a demo showing all presets side by side.
    """
    import matplotlib.pyplot as plt

    from grav_font.config import SimulationConfig
    from grav_font.physics.simulator import Simulator
    from grav_font.presets import PRESETS

    print("=" * 50)
    print("Gravitational Font Tracing - All Presets Demo")
    print("=" * 50)

    # Skip glyph_centered as it requires a glyph
    preset_names = [name for name in PRESETS.keys() if name != "glyph_centered"]

    n_presets = len(preset_names)
    cols = 3
    rows = (n_presets + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    config = SimulationConfig(n_steps=2000, damping=0.998)

    for i, name in enumerate(preset_names):
        print(f"\nRunning preset: {name}")

        # Get particles
        if name == "random_system":
            particles = PRESETS[name](seed=42)
        else:
            particles = PRESETS[name]()

        # Run simulation
        simulator = Simulator(particles, config=config)
        trajectory = simulator.run()

        # Plot
        ax = axes[i]
        ax.plot(trajectory[:, 0], trajectory[:, 1], "c-", alpha=0.7, linewidth=0.5)

        # Plot masses
        for p in particles:
            if p.is_pen:
                ax.scatter(p.position[0], p.position[1], c="red", s=30, zorder=10)
            else:
                size = 50 * (p.mass / 10) ** 0.5
                ax.scatter(p.position[0], p.position[1], c="blue", s=size, zorder=9)

        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_facecolor("black")

    # Hide unused axes
    for i in range(n_presets, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig("all_presets.png", dpi=150, facecolor="black")
    print("\nSaved to: all_presets.png")
    plt.show()


if __name__ == "__main__":
    # Check if running special demo modes
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo-no-font":
            demo_without_font()
            sys.exit(0)
        elif sys.argv[1] == "--demo-all-presets":
            demo_all_presets()
            sys.exit(0)

    sys.exit(main())
