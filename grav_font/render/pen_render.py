"""
Pen trail rendering using skia-python.

Renders particle trajectories as antialiased strokes on a canvas,
producing grayscale images that can be compared with target glyphs.
"""

from typing import Optional, Tuple

import numpy as np

try:
    import skia

    SKIA_AVAILABLE = True
except ImportError:
    SKIA_AVAILABLE = False


def render_trajectory(
    trajectory: np.ndarray,
    canvas_size: int = 256,
    stroke_width: float = 2.0,
    color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    background: Tuple[int, int, int, int] = (0, 0, 0, 255),
    antialias: bool = True,
) -> np.ndarray:
    """
    Render a trajectory as an antialiased stroke using skia.

    Args:
        trajectory: Numpy array of shape (n_points, 2) with [x, y] positions
        canvas_size: Size of the output square canvas in pixels
        stroke_width: Width of the pen stroke in pixels
        color: RGBA color tuple for the stroke (0-255)
        background: RGBA color tuple for the background (0-255)
        antialias: Whether to use antialiasing

    Returns:
        Grayscale numpy array of shape (canvas_size, canvas_size) in [0, 1]
    """
    if not SKIA_AVAILABLE:
        return _render_trajectory_fallback(trajectory, canvas_size, stroke_width)

    if len(trajectory) < 2:
        return np.zeros((canvas_size, canvas_size), dtype=np.float32)

    # Create surface and canvas
    surface = skia.Surface(canvas_size, canvas_size)
    canvas = surface.getCanvas()

    # Fill background
    canvas.clear(skia.Color(*background))

    # Create path from trajectory
    path = skia.Path()
    path.moveTo(float(trajectory[0, 0]), float(trajectory[0, 1]))

    for i in range(1, len(trajectory)):
        path.lineTo(float(trajectory[i, 0]), float(trajectory[i, 1]))

    # Create paint for stroke
    paint = skia.Paint()
    paint.setColor(skia.Color(*color))
    paint.setStyle(skia.Paint.kStroke_Style)
    paint.setStrokeWidth(stroke_width)
    paint.setAntiAlias(antialias)
    paint.setStrokeCap(skia.Paint.kRound_Cap)
    paint.setStrokeJoin(skia.Paint.kRound_Join)

    # Draw path
    canvas.drawPath(path, paint)

    # Get image data
    image = surface.makeImageSnapshot()
    array = image.toarray()

    # Convert to grayscale (take red channel since stroke is white)
    if array.ndim == 3:
        grayscale = array[:, :, 0].astype(np.float32) / 255.0
    else:
        grayscale = array.astype(np.float32) / 255.0

    return grayscale


def _render_trajectory_fallback(
    trajectory: np.ndarray,
    canvas_size: int = 256,
    stroke_width: float = 2.0,
) -> np.ndarray:
    """
    Fallback renderer using numpy when skia is not available.

    Uses simple line drawing with basic antialiasing approximation.
    """
    from scipy import ndimage

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    if len(trajectory) < 2:
        return canvas

    # Draw lines between consecutive points
    for i in range(len(trajectory) - 1):
        x0, y0 = trajectory[i]
        x1, y1 = trajectory[i + 1]
        _draw_line(canvas, x0, y0, x1, y1)

    # Apply gaussian blur to approximate stroke width and antialiasing
    sigma = stroke_width / 2.0
    if sigma > 0:
        canvas = ndimage.gaussian_filter(canvas, sigma=sigma)

    # Normalize
    if canvas.max() > 0:
        canvas = canvas / canvas.max()

    return canvas


def _draw_line(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    """Draw a line on the canvas using Bresenham's algorithm."""
    h, w = canvas.shape

    # Bresenham's line algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = int(x0), int(y0)
    x1_int, y1_int = int(x1), int(y1)

    while True:
        if 0 <= x < w and 0 <= y < h:
            canvas[y, x] = 1.0

        if x == x1_int and y == y1_int:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def render_trajectory_with_alpha(
    trajectory: np.ndarray,
    canvas_size: int = 256,
    stroke_width: float = 2.0,
    alpha: float = 0.3,
    accumulate: bool = True,
) -> np.ndarray:
    """
    Render trajectory with alpha blending for accumulated strokes.

    When accumulate=True, areas where the pen passes multiple times
    will appear darker/more intense.

    Args:
        trajectory: Numpy array of shape (n_points, 2)
        canvas_size: Size of output canvas
        stroke_width: Width of pen stroke
        alpha: Opacity per stroke segment (0-1)
        accumulate: If True, overlapping strokes add intensity

    Returns:
        Grayscale numpy array in [0, 1]
    """
    if not SKIA_AVAILABLE:
        # Fallback doesn't support alpha accumulation well
        return _render_trajectory_fallback(trajectory, canvas_size, stroke_width)

    if len(trajectory) < 2:
        return np.zeros((canvas_size, canvas_size), dtype=np.float32)

    # Create surface with alpha channel
    surface = skia.Surface(canvas_size, canvas_size)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color(0, 0, 0, 255))

    # Create paint with alpha
    alpha_int = int(alpha * 255)
    paint = skia.Paint()
    paint.setColor(skia.Color(255, 255, 255, alpha_int))
    paint.setStyle(skia.Paint.kStroke_Style)
    paint.setStrokeWidth(stroke_width)
    paint.setAntiAlias(True)
    paint.setStrokeCap(skia.Paint.kRound_Cap)
    paint.setStrokeJoin(skia.Paint.kRound_Join)

    if accumulate:
        # Draw segments individually for accumulation
        paint.setBlendMode(skia.BlendMode.kPlus)
        for i in range(len(trajectory) - 1):
            path = skia.Path()
            path.moveTo(float(trajectory[i, 0]), float(trajectory[i, 1]))
            x_next = float(trajectory[i + 1, 0])
            y_next = float(trajectory[i + 1, 1])
            path.lineTo(x_next, y_next)
            canvas.drawPath(path, paint)
    else:
        # Draw entire path at once
        path = skia.Path()
        path.moveTo(float(trajectory[0, 0]), float(trajectory[0, 1]))
        for i in range(1, len(trajectory)):
            path.lineTo(float(trajectory[i, 0]), float(trajectory[i, 1]))
        canvas.drawPath(path, paint)

    # Get image and convert to grayscale
    image = surface.makeImageSnapshot()
    array = image.toarray()

    if array.ndim == 3:
        grayscale = array[:, :, 0].astype(np.float32) / 255.0
    else:
        grayscale = array.astype(np.float32) / 255.0

    return np.clip(grayscale, 0, 1)


def trajectory_to_image(
    trajectory: np.ndarray,
    canvas_size: int = 256,
    stroke_width: float = 2.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert a trajectory to a grayscale image.

    Convenience wrapper around render_trajectory.

    Args:
        trajectory: Numpy array of shape (n_points, 2)
        canvas_size: Size of output canvas
        stroke_width: Width of pen stroke
        normalize: If True, normalize output to [0, 1]

    Returns:
        Grayscale numpy array
    """
    image = render_trajectory(
        trajectory,
        canvas_size=canvas_size,
        stroke_width=stroke_width,
    )

    if normalize and image.max() > 0:
        image = image / image.max()

    return image


def render_particles(
    positions: np.ndarray,
    canvas_size: int = 256,
    masses: Optional[np.ndarray] = None,
    pen_indices: Optional[list] = None,
    base_radius: float = 5.0,
    pen_color: Tuple[int, int, int, int] = (255, 100, 100, 255),
    mass_color: Tuple[int, int, int, int] = (100, 100, 255, 255),
) -> np.ndarray:
    """
    Render particle positions as circles.

    Args:
        positions: Numpy array of shape (n_particles, 2)
        canvas_size: Size of output canvas
        masses: Optional masses for scaling circle sizes
        pen_indices: Indices of pen particles (rendered differently)
        base_radius: Base radius for particles
        pen_color: RGBA color for pen particles
        mass_color: RGBA color for mass particles

    Returns:
        RGBA numpy array of shape (canvas_size, canvas_size, 4)
    """
    if not SKIA_AVAILABLE:
        return _render_particles_fallback(
            positions, canvas_size, masses, pen_indices, base_radius
        )

    pen_indices = pen_indices or []

    surface = skia.Surface(canvas_size, canvas_size)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color(0, 0, 0, 0))  # Transparent background

    for i, pos in enumerate(positions):
        x, y = float(pos[0]), float(pos[1])

        # Determine radius based on mass
        if masses is not None and i < len(masses):
            mass_ratio = masses[i] / 10.0
            radius = base_radius * (mass_ratio**0.5)
        else:
            radius = base_radius

        radius = max(2.0, min(radius, 30.0))  # Clamp radius

        # Choose color based on particle type
        if i in pen_indices:
            color = pen_color
            radius = base_radius * 0.5  # Pen is smaller
        else:
            color = mass_color

        paint = skia.Paint()
        paint.setColor(skia.Color(*color))
        paint.setStyle(skia.Paint.kFill_Style)
        paint.setAntiAlias(True)

        canvas.drawCircle(x, y, radius, paint)

    image = surface.makeImageSnapshot()
    return image.toarray()


def _render_particles_fallback(
    positions: np.ndarray,
    canvas_size: int = 256,
    masses: Optional[np.ndarray] = None,
    pen_indices: Optional[list] = None,
    base_radius: float = 5.0,
) -> np.ndarray:
    """Fallback particle renderer using numpy."""
    pen_indices = pen_indices or []

    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    for i, pos in enumerate(positions):
        x, y = int(pos[0]), int(pos[1])

        if masses is not None and i < len(masses):
            radius = int(base_radius * (masses[i] / 10.0) ** 0.5)
        else:
            radius = int(base_radius)

        radius = max(2, min(radius, 30))

        if i in pen_indices:
            color = [255, 100, 100, 255]
            radius = max(2, int(base_radius * 0.5))
        else:
            color = [100, 100, 255, 255]

        # Draw filled circle
        yy, xx = np.ogrid[:canvas_size, :canvas_size]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2

        for c in range(4):
            canvas[:, :, c] = np.where(mask, color[c], canvas[:, :, c])

    return canvas
