#!/usr/bin/env python3
"""Highlight visual differences between two images.

This utility loads a reference image and a comparison image, finds regions
that have changed, and draws highlighted circles on top of the comparison
image. The resulting image can be passed to another system (for example, an
LLM) to describe the changes.

The detection pipeline is intentionally dependency-light and relies only on
``numpy`` and ``Pillow``. A simple connected-component analysis is used to
merge neighbouring difference pixels into compact regions so that both icon
and text changes can be highlighted clearly.
"""
from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class DifferenceRegion:
    """Represents a connected area of change in the image."""

    bbox: Tuple[int, int, int, int]
    area: int

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    @property
    def radius(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return math.sqrt(((x1 - x0) / 2.0) ** 2 + ((y1 - y0) / 2.0) ** 2)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Highlight visual differences between two images by drawing circles",
    )
    parser.add_argument("reference", type=Path, help="Path to the reference image")
    parser.add_argument("comparison", type=Path, help="Path to the image to compare")
    parser.add_argument(
        "output",
        type=Path,
        help="Where to save the highlighted comparison image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help=(
            "Intensity difference threshold (0-255). Lower values are more sensitive "
            "but may pick up noise."
        ),
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=60,
        help="Minimum number of pixels for a difference region to be kept",
    )
    parser.add_argument(
        "--circle-padding",
        type=float,
        default=1.25,
        help="Multiplier applied to the detected radius to make the circle more generous",
    )
    parser.add_argument(
        "--stroke-width",
        type=int,
        default=6,
        help="Circle outline thickness in pixels",
    )
    parser.add_argument(
        "--fill-opacity",
        type=int,
        default=72,
        help="Opacity (0-255) for the translucent fill inside each circle",
    )
    parser.add_argument(
        "--no-fill",
        action="store_true",
        help="Disable translucent fills and draw only circle outlines",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help=(
            "Resize the comparison image to match the reference dimensions instead of "
            "raising an error when the sizes differ."
        ),
    )
    return parser.parse_args(argv)


def load_image(path: Path) -> Image.Image:
    try:
        image = Image.open(path)
    except FileNotFoundError as exc:  # pragma: no cover - CLI level safeguard
        raise SystemExit(f"Could not find image: {path}") from exc
    return image.convert("RGB")


def ensure_dimensions(
    reference: Image.Image, comparison: Image.Image, *, allow_resize: bool
) -> Image.Image:
    if reference.size == comparison.size:
        return comparison
    if not allow_resize:
        raise SystemExit(
            "Images have different dimensions. Either provide aligned images or "
            "run with --resize to automatically match the comparison image to the reference size."
        )
    return comparison.resize(reference.size, Image.LANCZOS)


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Return the image data as a float32 RGB array."""

    rgb = image.convert("RGB")
    return np.asarray(rgb, dtype=np.float32)


def compute_difference_mask(
    reference: np.ndarray,
    comparison: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Return a boolean mask where pixels differ beyond the supplied threshold."""

    if reference.shape != comparison.shape:
        raise ValueError("Image arrays must have the same shape")

    # Compute the Euclidean distance in RGB space so that hue shifts with a
    # similar luminance still register as changes (e.g. blue icon swapped for
    # red). This is more robust than operating on a grayscale image alone.
    diff = reference - comparison
    if diff.ndim == 3:
        diff_magnitude = np.linalg.norm(diff, axis=2)
    else:
        diff_magnitude = np.abs(diff)

    # Normalise local contrast: subtract the median difference so that uniform
    # colour or brightness shifts do not trigger false positives.
    diff_magnitude -= np.median(diff_magnitude)
    diff_magnitude = np.clip(diff_magnitude, 0, None)

    mask = diff_magnitude > threshold
    return mask


def neighbours(y: int, x: int, height: int, width: int) -> Iterable[Tuple[int, int]]:
    if y > 0:
        yield y - 1, x
    if y + 1 < height:
        yield y + 1, x
    if x > 0:
        yield y, x - 1
    if x + 1 < width:
        yield y, x + 1


def extract_regions(mask: np.ndarray, min_area: int) -> List[DifferenceRegion]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions: List[DifferenceRegion] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            queue: deque[Tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0

            while queue:
                cy, cx = queue.popleft()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for ny, nx in neighbours(cy, cx, height, width):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if area >= min_area:
                regions.append(DifferenceRegion((min_x, min_y, max_x + 1, max_y + 1), area))

    return regions


def draw_highlights(
    base_image: Image.Image,
    regions: Sequence[DifferenceRegion],
    *,
    circle_padding: float,
    stroke_width: int,
    fill_opacity: int,
    fill_enabled: bool,
) -> Image.Image:
    annotated = base_image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    outline_colour = (255, 0, 0, 255)
    fill_colour = (255, 0, 0, int(np.clip(fill_opacity, 0, 255)))

    for region in regions:
        cx, cy = region.center
        radius = region.radius * circle_padding
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]

        if fill_enabled and fill_colour[3] > 0:
            draw.ellipse(bbox, outline=outline_colour, width=stroke_width, fill=fill_colour)
        else:
            draw.ellipse(bbox, outline=outline_colour, width=stroke_width)

    combined = Image.alpha_composite(annotated, overlay)
    return combined.convert("RGB")


def highlight_differences(args: argparse.Namespace) -> Path:
    reference = load_image(args.reference)
    comparison = load_image(args.comparison)
    comparison = ensure_dimensions(reference, comparison, allow_resize=args.resize)

    ref_rgb = image_to_rgb_array(reference)
    cmp_rgb = image_to_rgb_array(comparison)

    mask = compute_difference_mask(ref_rgb, cmp_rgb, threshold=args.threshold)
    regions = extract_regions(mask, min_area=args.min_area)

    if not regions:
        print("No differences exceeding the configured thresholds were found.")

    annotated = draw_highlights(
        comparison,
        regions,
        circle_padding=args.circle_padding,
        stroke_width=args.stroke_width,
        fill_opacity=args.fill_opacity,
        fill_enabled=not args.no_fill,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(args.output)
    print(f"Saved highlighted image to {args.output}")
    return args.output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    highlight_differences(args)


if __name__ == "__main__":
    main()
