"""
Glyph loading and rasterization using freetype-py.

Provides functionality to:
- Load TTF/OTF font files
- Rasterize individual glyphs to grayscale bitmaps
- Compute distance fields and gradient fields for physics forces
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import freetype
import numpy as np
from scipy import ndimage


@dataclass
class GlyphData:
    """Container for a rasterized glyph and its derived fields."""

    character: str
    image: np.ndarray  # Grayscale [0, 1] float32
    distance_field: Optional[np.ndarray] = None
    gradient_field: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Return the center of the glyph bounding box."""
        if self.bbox is not None:
            x_min, y_min, x_max, y_max = self.bbox
            return ((x_min + x_max) / 2, (y_min + y_max) / 2)
        # Fall back to image center
        h, w = self.image.shape
        return (w / 2, h / 2)


def load_font(font_path: Union[str, Path], size: int = 200) -> freetype.Face:
    """
    Load a font file and set the character size.

    Args:
        font_path: Path to a TTF or OTF font file
        size: Font size in pixels

    Returns:
        freetype.Face object ready for glyph rendering

    Raises:
        FileNotFoundError: If font file doesn't exist
        freetype.FT_Exception: If font loading fails
    """
    font_path = Path(font_path)
    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")

    face = freetype.Face(str(font_path))
    face.set_pixel_sizes(0, size)
    return face


def rasterize_glyph(
    face: freetype.Face,
    character: str,
    canvas_size: int = 256,
) -> GlyphData:
    """
    Rasterize a single character to a grayscale numpy array.

    The glyph is centered on the canvas.

    Args:
        face: Loaded freetype.Face object
        character: Single character to rasterize
        canvas_size: Size of the output square canvas in pixels

    Returns:
        GlyphData with the rasterized image (grayscale float32 [0, 1])
    """
    if len(character) != 1:
        raise ValueError("character must be a single character")

    # Load the glyph
    face.load_char(character, freetype.FT_LOAD_RENDER)
    bitmap = face.glyph.bitmap

    # Convert bitmap to numpy array
    width = bitmap.width
    height = bitmap.rows

    if width == 0 or height == 0:
        # Empty glyph (e.g., space character)
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        return GlyphData(
            character=character,
            image=canvas,
            bbox=(0, 0, 0, 0),
        )

    # Get bitmap buffer as numpy array
    buffer = np.array(bitmap.buffer, dtype=np.uint8).reshape(height, width)

    # Create canvas and center the glyph
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    # Calculate position to center the glyph
    x_offset = (canvas_size - width) // 2
    y_offset = (canvas_size - height) // 2

    # Adjust for glyph bearing (baseline alignment)
    bearing_y = face.glyph.bitmap_top

    # Center based on visual center, accounting for baseline
    x_offset = (canvas_size - width) // 2
    y_offset = (canvas_size - height) // 2 + (height - bearing_y) // 2

    # Clamp offsets to valid range
    x_offset = max(0, min(x_offset, canvas_size - width))
    y_offset = max(0, min(y_offset, canvas_size - height))

    # Place glyph on canvas
    y_end = y_offset + height
    x_end = x_offset + width
    canvas[y_offset:y_end, x_offset:x_end] = buffer / 255.0

    # Compute bounding box
    bbox = (x_offset, y_offset, x_offset + width, y_offset + height)

    return GlyphData(
        character=character,
        image=canvas,
        bbox=bbox,
    )


def compute_distance_field(
    glyph_data: GlyphData,
    threshold: float = 0.1,
    signed: bool = True,
) -> GlyphData:
    """
    Compute the distance transform of the glyph.

    Creates a field where each pixel contains the distance to the nearest
    glyph boundary. Useful for creating attractive forces toward the glyph.

    Args:
        glyph_data: GlyphData with rasterized image
        threshold: Threshold for binarizing the glyph
        signed: If True, compute signed distance (negative inside)

    Returns:
        GlyphData with distance_field populated
    """
    image = glyph_data.image

    # Binarize the image
    binary = image > threshold

    # Distance transform from outside to boundary
    dist_outside = ndimage.distance_transform_edt(~binary)

    if signed:
        # Distance transform from inside to boundary
        dist_inside = ndimage.distance_transform_edt(binary)
        # Signed distance: negative inside, positive outside
        distance_field = dist_outside - dist_inside
    else:
        distance_field = dist_outside

    # Create new GlyphData with distance field
    return GlyphData(
        character=glyph_data.character,
        image=glyph_data.image,
        distance_field=distance_field.astype(np.float32),
        bbox=glyph_data.bbox,
    )


def compute_gradient_field(glyph_data: GlyphData) -> GlyphData:
    """
    Compute the gradient of the distance field.

    The gradient points in the direction of steepest ascent (away from glyph
    boundary). Negating this gives a force that pulls toward the boundary.

    Args:
        glyph_data: GlyphData with distance_field populated

    Returns:
        GlyphData with gradient_field populated as (grad_y, grad_x)

    Raises:
        ValueError: If distance_field has not been computed
    """
    if glyph_data.distance_field is None:
        raise ValueError(
            "distance_field must be computed first (call compute_distance_field)"
        )

    # Compute gradient (returns dy, dx)
    grad_y, grad_x = np.gradient(glyph_data.distance_field)

    # Normalize gradient vectors
    magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
    grad_x_norm = grad_x / magnitude
    grad_y_norm = grad_y / magnitude

    return GlyphData(
        character=glyph_data.character,
        image=glyph_data.image,
        distance_field=glyph_data.distance_field,
        gradient_field=(
            grad_y_norm.astype(np.float32),
            grad_x_norm.astype(np.float32),
        ),
        bbox=glyph_data.bbox,
    )


def load_glyph(
    font_path: Union[str, Path],
    character: str,
    canvas_size: int = 256,
    font_size: int = 200,
    compute_fields: bool = True,
) -> GlyphData:
    """
    Convenience function to load a font and rasterize a glyph in one call.

    Args:
        font_path: Path to font file
        character: Character to rasterize
        canvas_size: Output canvas size
        font_size: Font rendering size
        compute_fields: If True, also compute distance and gradient fields

    Returns:
        GlyphData with all fields populated
    """
    face = load_font(font_path, font_size)
    glyph_data = rasterize_glyph(face, character, canvas_size)

    if compute_fields:
        glyph_data = compute_distance_field(glyph_data)
        glyph_data = compute_gradient_field(glyph_data)

    return glyph_data
