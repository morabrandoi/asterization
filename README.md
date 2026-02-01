# Gravitational Font Tracing (Asterization)

An experimental system that traces font glyphs using simulated 2D gravitational physics, producing stylized orbital drawings rather than literal outlines.

## Overview

A single "pen" particle draws onto a canvas while interacting with gravitational bodies (1–5). The pen's trajectory creates artistic interpretations of letterforms through orbital mechanics.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Font File  │────▶│   Glyph     │────▶│  Distance   │
│  (TTF/OTF)  │     │  Rasterizer │     │   Field     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Presets   │────▶│  Simulator  │────▶│ Trajectory  │
│ (Particles) │     │  (N-body)   │     │   Array     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Renderer   │────▶│ Visualizer  │
                    │   (Skia)    │     │ (Matplotlib)│
                    └─────────────┘     └─────────────┘
```

---

## Setup with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package installer. If you don't have it:

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### Initialize the project:

```bash
cd asterization

# Create virtual environment with Python 3.11+
uv venv --python 3.11

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies (includes ruff for linting/formatting)
uv pip install -r requirements.txt

# Or install as editable package
uv pip install -e .
```

### Linting & Formatting

The project uses **ruff** for fast linting and formatting (replaces flake8 + black + isort):

```bash
# Check for lint errors
ruff check .

# Auto-fix lint errors + format code
ruff check --fix .
ruff format .

# Or use the Makefile shortcuts:
make lint          # Check for issues
make fix           # Auto-fix everything
make check         # Full check (lint + format + mypy)
```

---

## Quick Start

```bash
# Test without a font file
python run_demo.py --demo-no-font

# Run with a font
python run_demo.py --font fonts/Lorestta.otf --char A

# Show animation
python run_demo.py --font fonts/Lorestta.otf --char A --animate

# Save output
python run_demo.py --font fonts/Lorestta.otf --save output.png
python run_demo.py --font fonts/Lorestta.otf --animate --save trace.gif

# Try different presets
python run_demo.py --preset single_orbit
python run_demo.py --preset binary_system
python run_demo.py --preset three_body_triangle
python run_demo.py --preset four_corners
python run_demo.py --preset spiral_inward
python run_demo.py --preset random_system --seed 42

# View all presets side-by-side
python run_demo.py --demo-all-presets
```

---

## CLI Reference

```
python run_demo.py --help

Options:
  --font, -f        Path to TTF/OTF font file
  --char, -c        Character to trace (default: A)
  --preset, -p      Particle configuration preset
  --steps, -n       Number of simulation steps (default: 2000)
  --animate, -a     Show animated simulation
  --save, -s        Save output to file (png, gif, mp4)
  --canvas-size     Canvas size in pixels (default: 256)
  --font-size       Font size for rendering (default: 200)
  --stroke-width    Pen stroke width (default: 1.5)
  --no-glyph        Hide glyph background (pure orbital art)
  --seed            Random seed for random_system preset
  --list-presets    List available presets
```

---

## Project Structure

```
asterization/
├── pyproject.toml           # Project metadata & dependencies
├── requirements.txt         # Dependencies for uv/pip
├── run_demo.py              # CLI demo script
├── fonts/                   # User font files (.ttf/.otf)
└── grav_font/               # Main Python package
    ├── config.py            # Configuration dataclasses
    ├── presets.py           # Preset particle configurations
    ├── glyphs/
    │   └── target.py        # Font loading & glyph rasterization
    ├── physics/
    │   └── simulator.py     # N-body gravitational simulator
    ├── render/
    │   └── pen_render.py    # Pen trail rendering (skia)
    └── visualize/
        └── viewer.py        # Matplotlib visualization
```

---

## Component Overview

### `grav_font/config.py` — Configuration

Centralized simulation parameters using dataclasses:

| Class              | Parameters                                                                                                                                                     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SimulationConfig` | `canvas_size`, `font_size`, `dt` (timestep), `n_steps`, `G` (gravitational constant), `softening`, `damping`, `max_velocity`, `stroke_width`, `stroke_opacity` |
| `ParticleConfig`   | `default_mass`, `pen_mass`                                                                                                                                     |

```python
from grav_font.config import DEFAULT_CONFIG

DEFAULT_CONFIG.G = 150.0  # Increase gravity
DEFAULT_CONFIG.damping = 0.99  # More energy loss
```

---

### `grav_font/glyphs/target.py` — Glyph Loading

Loads font files and converts characters to images.

| Component                  | Purpose                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `GlyphData`                | Container: `character`, `image`, `distance_field`, `gradient_field`, `bbox`, `center` |
| `load_font()`              | Load TTF/OTF via freetype-py                                                          |
| `rasterize_glyph()`        | Render character to grayscale numpy array `[0,1]`                                     |
| `compute_distance_field()` | Signed distance transform (negative inside, positive outside)                         |
| `compute_gradient_field()` | Normalized gradient for force direction                                               |
| `load_glyph()`             | Convenience function combining all steps                                              |

```python
from grav_font.glyphs import load_glyph

glyph = load_glyph("fonts/Lorestta.otf", "A", canvas_size=256)
print(glyph.center)  # (128.5, 130.0)
```

---

### `grav_font/physics/simulator.py` — Gravitational Simulation

The core N-body physics engine.

| Component           | Purpose                                                         |
| ------------------- | --------------------------------------------------------------- |
| `Particle`          | Dataclass: `position`, `velocity`, `mass`, `is_pen`, `is_fixed` |
| `Simulator`         | Main simulation class with softened gravity                     |
| `IntegrationMethod` | Enum: `EULER`, `RK4`                                            |

**Physics:**

- Softened gravity: `F = G·m₁·m₂·r / (|r|² + ε²)^(3/2)`
- Velocity damping each timestep
- Optional velocity clamping

```python
from grav_font.physics import Particle, Simulator

pen = Particle([100, 100], [10, 0], mass=1.0, is_pen=True)
attractor = Particle([128, 128], [0, 0], mass=50.0, is_fixed=True)

sim = Simulator([pen, attractor])
trajectory = sim.run(1000)  # Returns (1001, 2) numpy array
```

---

### `grav_font/presets.py` — Preset Configurations

Ready-to-use particle setups:

| Preset                  | Description                              |
| ----------------------- | ---------------------------------------- |
| `single_orbit()`        | Pen orbiting one central mass            |
| `binary_system()`       | Two fixed masses, figure-8 patterns      |
| `three_body_triangle()` | Three masses in triangle, chaotic orbits |
| `four_corners()`        | Four masses at square corners            |
| `spiral_inward()`       | Sub-orbital velocity → inward spiral     |
| `glyph_centered()`      | Attractors around glyph center           |
| `random_system()`       | Random positions/masses (seedable)       |

```python
from grav_font.presets import get_preset, list_presets

print(list_presets())  # ['single_orbit', 'binary_system', ...]

particles = get_preset("three_body_triangle", center=(128, 128))
```

---

### `grav_font/render/pen_render.py` — Pen Trail Rendering

Renders trajectories as antialiased strokes using skia-python.

| Function                         | Purpose                                   |
| -------------------------------- | ----------------------------------------- |
| `render_trajectory()`            | Draw trajectory as antialiased polyline   |
| `render_trajectory_with_alpha()` | Alpha-blended (darker where pen overlaps) |
| `trajectory_to_image()`          | Returns grayscale numpy array             |
| `render_particles()`             | Draw particles as colored circles         |

```python
from grav_font.render import render_trajectory

image = render_trajectory(trajectory, canvas_size=256, stroke_width=2.0)
# Returns (256, 256) grayscale array [0, 1]
```

---

### `grav_font/visualize/viewer.py` — Visualization

Matplotlib-based visualization and animation.

| Function                           | Purpose                                     |
| ---------------------------------- | ------------------------------------------- |
| `visualize_static()`               | Static plot: glyph + trajectory + particles |
| `visualize_comparison()`           | Side-by-side: target, rendered, RGB overlay |
| `animate_simulation()`             | Real-time animation, save to GIF/MP4        |
| `visualize_distance_field()`       | Show SDF and gradient quiver plot           |
| `visualize_trajectory_evolution()` | Trajectory at multiple timepoints           |

```python
from grav_font.visualize import animate_simulation, show_plot

anim = animate_simulation(simulator, glyph=glyph, n_steps=500)
show_plot()
```

---

## How It Works

1. **Load Glyph** — Rasterize font character to grayscale bitmap, compute distance field
2. **Setup Physics** — Place gravitational bodies and "pen" particle using presets
3. **Simulate** — Run N-body gravitational simulation with softening & damping
4. **Render** — Draw pen trajectory as antialiased stroke via skia
5. **Visualize** — Display static/animated result or save to file

---

## Dependencies

| Package        | Purpose                   |
| -------------- | ------------------------- |
| `numpy`        | Array operations          |
| `scipy`        | Distance transforms       |
| `matplotlib`   | Visualization & animation |
| `freetype-py`  | Font loading              |
| `skia-python`  | Antialiased rendering     |
| `scikit-image` | Image utilities           |

---

## Future Directions

- [ ] Glyph-derived force fields (attract pen toward letterform)
- [ ] CMA-ES optimization of initial conditions
- [ ] Multi-pen systems
- [ ] Stroke ordering constraints
- [ ] JAX-based differentiable simulation

---

## License

Experimental / Personal Project
