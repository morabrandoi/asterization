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

# Install dependencies
uv pip install -r requirements.txt

# Or install as editable package
uv pip install -e .
```

### Linting & Formatting

The project uses **ruff** for linting and formatting:

```bash
make lint    # Check for issues
make fix     # Auto-fix everything
make check   # Full check (lint + format + mypy)
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
```

---

## Visualization Modes: Static vs Animated

The demo has two distinct visualization modes with different rendering pipelines:

### Static Mode (default)

Runs the **entire simulation first**, then renders the complete trajectory as a single image.

```bash
python run_demo.py --font fonts/Lorestta.otf --char A
```

**How it works:**

1. Simulation runs for all `--steps` (default: 2000)
2. Full trajectory is captured as a numpy array
3. **Skia** renders the trajectory as an antialiased stroke
4. Matplotlib displays the final result

**Best for:**

- High-quality output images
- Blur effects (`--blur`)
- Side-by-side glyph comparison
- Saving to PNG

**Supports:** `--stroke-width`, `--blur`

---

### Animated Mode (`--animate`)

Runs the simulation **step-by-step in real-time**, drawing the trajectory as it evolves.

```bash
python run_demo.py --font fonts/Lorestta.otf --char A --animate
```

**How it works:**

1. Simulation advances one step per frame
2. **Matplotlib** draws the trajectory line directly (no Skia)
3. Display updates in real-time via `FuncAnimation`

**Best for:**

- Watching orbital dynamics unfold
- Debugging particle behavior
- Saving to GIF/MP4

**Supports:** `--stroke-width`, `--interval` (frame timing)

> **Note:** `--blur` only works in static mode (Skia feature). Animation uses Matplotlib's line renderer.

---

## CLI Reference

### Basic Options

| Flag             | Short | Default               | Description                                       |
| ---------------- | ----- | --------------------- | ------------------------------------------------- |
| `--font`         | `-f`  | None                  | Path to TTF/OTF font file                         |
| `--char`         | `-c`  | `A`                   | Character to trace                                |
| `--preset`       | `-p`  | `three_body_triangle` | Particle configuration preset                     |
| `--steps`        | `-n`  | `2000`                | Number of simulation steps                        |
| `--animate`      | `-a`  | off                   | Show real-time animation instead of static result |
| `--save`         | `-s`  | None                  | Save output to file (`.png`, `.gif`, `.mp4`)      |
| `--no-glyph`     |       | off                   | Hide glyph background (pure orbital art)          |
| `--list-presets` |       |                       | List available presets and exit                   |

### Canvas & Font

| Flag            | Default | Description                    |
| --------------- | ------- | ------------------------------ |
| `--canvas-size` | `256`   | Canvas size in pixels (square) |
| `--font-size`   | `200`   | Font size for glyph rendering  |

### Pen Initial Conditions

Override the preset's default pen velocity:

| Flag          | Default       | Description                       |
| ------------- | ------------- | --------------------------------- |
| `--pen-angle` | (from preset) | Initial velocity angle in degrees |
| `--pen-speed` | (from preset) | Initial velocity magnitude        |

**Angle reference:**

- `0°` = right (→)
- `90°` = down (↓)
- `180°` = left (←)
- `270°` = up (↑)

```bash
# Break out of symmetric equilibrium
python run_demo.py --preset three_body_triangle --pen-angle 73 --pen-speed 30
```

### Physics Simulation

| Flag     | Default | Description                                                            |
| -------- | ------- | ---------------------------------------------------------------------- |
| `--dt`   | `0.01`  | Physics timestep. Larger = faster movement per step, but less accurate |
| `--seed` | None    | Random seed for `random_system` preset (reproducibility)               |

```bash
# Faster simulation (pen covers more ground per frame)
python run_demo.py --preset single_orbit --dt 0.03

# Reproducible random system
python run_demo.py --preset random_system --seed 42
```

### Rendering & Visual Style

| Flag             | Default | Mode         | Description                                  |
| ---------------- | ------- | ------------ | -------------------------------------------- |
| `--stroke-width` | `1.5`   | Both         | Pen trail line thickness (pixels)            |
| `--blur`         | `0.0`   | Static only  | Gaussian blur sigma for soft/glowy effect    |
| `--interval`     | `16`    | Animate only | Milliseconds between frames (~60fps default) |

```bash
# Thick, soft strokes (static)
python run_demo.py --font fonts/Lorestta.otf --char A --stroke-width 4 --blur 2

# Slow-motion animation
python run_demo.py --preset single_orbit --animate --interval 50

# Thick animated line
python run_demo.py --preset binary_system --animate --stroke-width 3
```

### Special Demo Modes

| Flag                 | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| `--demo-no-font`     | Quick test without font file (three-body system)           |
| `--demo-all-presets` | Render all presets side-by-side, save to `all_presets.png` |

---

## Presets

| Preset                | Bodies | Pattern             | Description                        |
| --------------------- | ------ | ------------------- | ---------------------------------- |
| `single_orbit`        | 1      | Circular/elliptical | Pen orbiting single central mass   |
| `binary_system`       | 2      | Figure-8, chaotic   | Two fixed masses                   |
| `three_body_triangle` | 3      | Chaotic, complex    | Equilateral triangle of masses     |
| `four_corners`        | 4      | Bounded, structured | Masses at square corners           |
| `spiral_inward`       | 1      | Inward spiral       | Sub-orbital velocity decays inward |
| `glyph_centered`      | 3      | Glyph-following     | Attractors around glyph center     |
| `random_system`       | 3      | Unpredictable       | Random positions/masses            |

```bash
# Try different presets
python run_demo.py --preset single_orbit
python run_demo.py --preset binary_system
python run_demo.py --preset four_corners --animate
python run_demo.py --preset random_system --seed 123
```

---

## Examples

### Basic Usage

```bash
# Simple trace of letter A
python run_demo.py --font fonts/Lorestta.otf --char A

# Animated with custom angle to avoid symmetric trap
python run_demo.py --font fonts/Lorestta.otf --char X --animate --pen-angle 120

# Pure orbital art (no glyph)
python run_demo.py --preset three_body_triangle --no-glyph --steps 5000
```

### High Quality Output

```bash
# Larger canvas, thick soft strokes
python run_demo.py --font fonts/Lorestta.otf --char B \
  --canvas-size 512 --stroke-width 3 --blur 1.5 --save letter_b.png

# Long simulation for complex patterns
python run_demo.py --preset four_corners --steps 10000 --save orbit.png
```

### Animation Tuning

```bash
# Fast physics, normal playback
python run_demo.py --preset single_orbit --animate --dt 0.02

# Normal physics, slow-motion playback
python run_demo.py --preset binary_system --animate --interval 40

# Save as GIF
python run_demo.py --preset three_body_triangle --animate --save trace.gif
```

### Experimentation

```bash
# Override pen initial conditions
python run_demo.py --preset three_body_triangle \
  --pen-angle 45 --pen-speed 50 --steps 3000

# Reproducible random exploration
python run_demo.py --preset random_system --seed 42 --animate
```

---

## Project Structure

```
asterization/
├── pyproject.toml           # Project metadata & ruff config
├── requirements.txt         # Dependencies
├── Makefile                 # lint/fix/check shortcuts
├── run_demo.py              # CLI entry point
├── fonts/                   # Your font files (.ttf/.otf)
└── grav_font/               # Main package
    ├── config.py            # SimulationConfig dataclass
    ├── presets.py           # Preset particle configurations
    ├── glyphs/
    │   └── target.py        # Font loading, SDF computation
    ├── physics/
    │   └── simulator.py     # N-body gravitational engine
    ├── render/
    │   └── pen_render.py    # Skia-based trajectory rendering
    └── visualize/
        └── viewer.py        # Matplotlib visualization
```

---

## Physics Details

### Softened Gravity

```
F = G · m₁ · m₂ · r / (|r|² + ε²)^(3/2)
```

- `G` = gravitational constant (default: 100.0)
- `ε` = softening parameter (default: 5.0) — prevents singularity at r=0
- Damping ≈ 1.0 (energy preserved for stable orbits)

### Integration Methods

- **Euler** (default): Fast, good for visualization
- **RK4**: More accurate, better energy conservation

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
| `ruff`         | Linting & formatting      |

---

## Troubleshooting

### Animation doesn't play

Make sure you're not running in a headless environment. The animation requires an interactive matplotlib backend.

### Orbit decays into spiral

Check `damping` in config — should be very close to `1.0` (default: `0.999999999`).

### Pen stuck in equilibrium

Use `--pen-angle` to break symmetry:

```bash
python run_demo.py --preset three_body_triangle --pen-angle 73
```

### Blur not working

`--blur` only works in static mode (uses Skia). Animation mode uses Matplotlib's line renderer which doesn't support blur.

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
