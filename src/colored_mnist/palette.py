"""CIELAB palette generation, contrast verification, and linear-RGB compositing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from colored_mnist.config import DatasetConfig

# CIE D65 reference white implied by the IEC 61966-2-1 sRGB matrix
_D65_XN = 0.95047
_D65_YN = 1.00000
_D65_ZN = 1.08883
_LAB_EPS = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0

# Precomputed sRGB uint8 <-> linear RGB lookup tables for exact, fast compositing
_U8_GRID = np.arange(256, dtype=np.float64) / 255.0
_SRGB_U8_TO_LINEAR = np.where(
    _U8_GRID <= 0.04045,
    _U8_GRID / 12.92,
    ((_U8_GRID + 0.055) / 1.055) ** 2.4,
)


def lab_to_linear_srgb(L: float, a: float, b: float) -> tuple[float, float, float]:
    """Convert CIELAB (D65) coordinates to unclipped linear sRGB floats."""
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    xr = fx**3 if fx**3 > _LAB_EPS else (116.0 * fx - 16.0) / _LAB_KAPPA
    yr = ((L + 16.0) / 116.0) ** 3 if L > _LAB_KAPPA * _LAB_EPS else L / _LAB_KAPPA
    zr = fz**3 if fz**3 > _LAB_EPS else (116.0 * fz - 16.0) / _LAB_KAPPA

    x = xr * _D65_XN
    y = yr * _D65_YN
    z = zr * _D65_ZN

    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return (r, g, bl)


def linear_to_srgb_scalar(c: float) -> float:
    """Encode a single linear RGB component in [0, 1] to sRGB."""
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def linear_to_srgb_array(lin: np.ndarray) -> np.ndarray:
    """Encode linear RGB array in [0, 1] to sRGB float64 array in [0, 1]."""
    return np.where(
        lin <= 0.0031308,
        12.92 * lin,
        1.055 * np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.4) - 0.055,
    )


def srgb_u8_to_linear(rgb_u8: tuple[int, int, int]) -> tuple[float, float, float]:
    """Decode quantized 8-bit sRGB channels to linear sRGB floats."""
    return (
        float(_SRGB_U8_TO_LINEAR[rgb_u8[0]]),
        float(_SRGB_U8_TO_LINEAR[rgb_u8[1]]),
        float(_SRGB_U8_TO_LINEAR[rgb_u8[2]]),
    )


def relative_luminance(rgb_lin: tuple[float, float, float]) -> float:
    """Compute WCAG 2.1 relative luminance from linear sRGB components."""
    return 0.2126 * rgb_lin[0] + 0.7152 * rgb_lin[1] + 0.0722 * rgb_lin[2]


def contrast_ratio_against_white(rgb_lin: tuple[float, float, float]) -> float:
    """Compute WCAG 2.1 contrast ratio of pure white (#FFFFFF) over a background."""
    y_bg = relative_luminance(rgb_lin)
    return 1.05 / (y_bg + 0.05)


@dataclass(frozen=True)
class PaletteColor:
    """Single background color entry with CIELAB source and quantized sRGB values."""

    color_id: int
    hue_deg: float
    lab: tuple[float, float, float]
    srgb_u8: tuple[int, int, int]
    hex_code: str
    linear_rgb_exact: tuple[float, float, float]
    linear_rgb_quantized: tuple[float, float, float]
    relative_luminance: float
    contrast_vs_white: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "color_id": self.color_id,
            "hue_deg": self.hue_deg,
            "lab": list(self.lab),
            "srgb_u8": list(self.srgb_u8),
            "hex_code": self.hex_code,
            "linear_rgb_exact": [round(x, 6) for x in self.linear_rgb_exact],
            "linear_rgb_quantized": [round(x, 6) for x in self.linear_rgb_quantized],
            "relative_luminance": round(self.relative_luminance, 6),
            "contrast_vs_white": round(self.contrast_vs_white, 4),
        }


@dataclass(frozen=True)
class Palette:
    """Fixed 10-color isoluminant palette and class-to-color mappings."""

    colors: tuple[PaletteColor, ...]
    preferred_color_by_digit: tuple[int, ...]
    swapped_target_by_digit: tuple[int, ...]
    color_to_preferred_digit: dict[int, int]

    def __post_init__(self) -> None:
        if len(self.colors) != 10:
            raise ValueError(f"Expected 10 colors, got {len(self.colors)}")
        if sorted(self.preferred_color_by_digit) != list(range(10)):
            raise ValueError("preferred_color_by_digit must be a bijection on 0..9")
        for digit, target in enumerate(self.swapped_target_by_digit):
            if target == digit:
                raise ValueError(f"swapped_target_by_digit[{digit}] must differ from {digit}")
        if sorted(self.swapped_target_by_digit) != list(range(10)):
            raise ValueError("swapped_target_by_digit must be a permutation of 0..9")

    @property
    def bg_linear_table(self) -> np.ndarray:
        """Shape (10, 3) float64 array of quantized-sRGB-backed linear RGB colors."""
        return np.array([c.linear_rgb_quantized for c in self.colors], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "colors": [c.to_dict() for c in self.colors],
            "preferred_color_by_digit": list(self.preferred_color_by_digit),
            "swapped_target_by_digit": list(self.swapped_target_by_digit),
            "color_to_preferred_digit": {str(k): v for k, v in self.color_to_preferred_digit.items()},
        }


def load_palette(path: Path) -> Palette:
    """Load a saved palette, decoding the stored sRGB bytes for compositing."""
    data = json.loads(path.read_text(encoding="utf-8"))
    colors = []
    for row in sorted(data["colors"], key=lambda row: row["color_id"]):
        rgb = tuple(row["srgb_u8"])
        if len(rgb) != 3 or any(type(channel) is not int or not 0 <= channel <= 255 for channel in rgb):
            raise ValueError(f"Invalid sRGB color {row['color_id']} in {path}")
        colors.append(
            PaletteColor(
                **{
                    **row,
                    "lab": tuple(row["lab"]),
                    "srgb_u8": rgb,
                    "linear_rgb_exact": tuple(row["linear_rgb_exact"]),
                    "linear_rgb_quantized": srgb_u8_to_linear(rgb),
                }
            )
        )
    if [color.color_id for color in colors] != list(range(10)):
        raise ValueError(f"Expected palette color IDs 0 through 9 in {path}")
    preferred = tuple(data["preferred_color_by_digit"])
    return Palette(
        colors=tuple(colors),
        preferred_color_by_digit=preferred,
        swapped_target_by_digit=tuple(data["swapped_target_by_digit"]),
        color_to_preferred_digit={color: digit for digit, color in enumerate(preferred)},
    )


def build_palette(config: DatasetConfig) -> Palette:
    """Build and verify the 10-color CIELAB palette without sRGB gamut clipping."""
    colors: list[PaletteColor] = []
    for color_id in range(10):
        hue_deg = config.lab_hue_start_deg + color_id * config.lab_hue_step_deg
        hue_rad = math.radians(hue_deg)
        a = config.lab_chroma * math.cos(hue_rad)
        b = config.lab_chroma * math.sin(hue_rad)
        lin_exact = lab_to_linear_srgb(config.lab_lightness, a, b)

        if min(lin_exact) < 0.0 or max(lin_exact) > 1.0:
            raise ValueError(
                f"Color {color_id} (L={config.lab_lightness}, C={config.lab_chroma}, h={hue_deg}) "
                f"clips sRGB gamut: linear_rgb={lin_exact}"
            )

        srgb_u8 = tuple(int(round(linear_to_srgb_scalar(c) * 255.0)) for c in lin_exact)
        lin_q = srgb_u8_to_linear(srgb_u8)  # type: ignore[arg-type]
        lum = relative_luminance(lin_q)
        cr = contrast_ratio_against_white(lin_q)
        if cr < config.min_contrast_ratio:
            raise ValueError(
                f"Color {color_id} contrast ratio {cr:.2f} below minimum {config.min_contrast_ratio:.2f}"
            )

        hex_code = f"#{srgb_u8[0]:02x}{srgb_u8[1]:02x}{srgb_u8[2]:02x}"
        colors.append(
            PaletteColor(
                color_id=color_id,
                hue_deg=hue_deg,
                lab=(config.lab_lightness, round(a, 6), round(b, 6)),
                srgb_u8=srgb_u8,  # type: ignore[arg-type]
                hex_code=hex_code,
                linear_rgb_exact=lin_exact,
                linear_rgb_quantized=lin_q,
                relative_luminance=lum,
                contrast_vs_white=cr,
            )
        )

    color_to_digit = {
        color_id: digit for digit, color_id in enumerate(config.preferred_color_by_digit)
    }
    return Palette(
        colors=tuple(colors),
        preferred_color_by_digit=config.preferred_color_by_digit,
        swapped_target_by_digit=config.swapped_target_by_digit,
        color_to_preferred_digit=color_to_digit,
    )


def composite_digit_srgb(
    mask_u8: np.ndarray,
    bg_linear_rgb: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Composite a white MNIST foreground mask over a background in linear RGB, then encode to uint8 sRGB.

    Args:
        mask_u8: Grayscale MNIST image of shape (H, W) and dtype uint8 in [0, 255].
        bg_linear_rgb: Background color in linear RGB floats of shape (3,).

    Returns:
        Rendered sRGB uint8 image of shape (H, W, 3).
    """
    if mask_u8.ndim != 2 or mask_u8.dtype != np.uint8:
        raise ValueError(f"Expected 2D uint8 mask, got shape={mask_u8.shape}, dtype={mask_u8.dtype}")

    alpha = mask_u8.astype(np.float64)[..., None] / 255.0
    bg = np.asarray(bg_linear_rgb, dtype=np.float64).reshape(1, 1, 3)
    # White foreground has linear RGB (1.0, 1.0, 1.0)
    out_lin = alpha + (1.0 - alpha) * bg
    out_srgb = linear_to_srgb_array(out_lin)
    return np.clip(np.rint(out_srgb * 255.0), 0, 255).astype(np.uint8)

