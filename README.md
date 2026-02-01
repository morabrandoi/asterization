# Gravitational Font Tracing (Asterization)

An experimental system that traces font glyphs using simulated 2D gravitational physics, producing stylized orbital drawings rather than literal outlines.

## Overview

A single "pen" particle draws onto a canvas while interacting with gravitational bodies (1–5). The pen's trajectory creates artistic interpretations of letterforms through orbital mechanics.

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
cd /Users/brandomora/Desktop/personal/asterization

# Create virtual environment with Python 3.11+
uv venv --python 3.11

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Or install as editable package (recommended)
uv pip install -e .
```

## Quick Start

### Test without a font file:

```bash
python run_demo.py --demo-no-font
```

### Run with a font:

```bash
# Add a TTF/OTF font to the fonts/ directory first
python run_demo.py --font fonts/your_font.ttf --char A
```

### Show animation:

```bash
python run_demo.py --font fonts/your_font.ttf --char A --animate
```

### Save output:

```bash
python run_demo.py --font fonts/your_font.ttf --save output.png
python run_demo.py --font fonts/your_font.ttf --animate --save trace.gif
```

### Try different presets:

```bash
python run_demo.py --preset single_orbit
python run_demo.py --preset binary_system
python run_demo.py --preset three_body_triangle
python run_demo.py --preset four_corners
python run_demo.py --preset spiral_inward
python run_demo.py --preset random_system --seed 42
```

### View all presets:

```bash
python run_demo.py --demo-all-presets
```

## CLI Options

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

## Project Structure

```
asterization/
├── pyproject.toml           # Project config & dependencies
├── requirements.txt         # Dependencies for uv/pip
├── run_demo.py              # Main demo script
├── fonts/                   # Place your .ttf/.otf fonts here
└── grav_font/
    ├── config.py            # Simulation configuration
    ├── presets.py           # Particle configuration presets
    ├── glyphs/
    │   └── target.py        # Font loading & rasterization
    ├── physics/
    │   └── simulator.py     # N-body gravitational simulator
    ├── render/
    │   └── pen_render.py    # Skia-based pen trail rendering
    └── visualize/
        └── viewer.py        # Matplotlib visualization
```

## How It Works

1. **Load Glyph**: Rasterize a font character to a grayscale bitmap
2. **Setup Physics**: Place gravitational bodies and a "pen" particle
3. **Simulate**: Run N-body gravitational simulation with softening & damping
4. **Render**: Draw the pen's trajectory as an antialiased stroke
5. **Visualize**: Display or save the result

## License

Experimental / Personal Project
